from __future__ import annotations
import random
import re
import json
import networkx as nx
import numpy as np
import faiss
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from autograph.rag_server.reafiner_prompt import REAFINER_JUDGEMENT_SYSTEM_PROMPT, REAFINER_JUDGEMENT_USER_PROMPT, \
    REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT, REAFINER_ERROR_ABDUCTION_USER_PROMPT, \
    REAFINER_KG_REFINEMENT_SYSTEM_PROMPT, REAFINER_KG_REFINEMENT_USER_PROMPT, \
    REFINE_SUBGRAPH_SYSTEM_PROMPT, REFINE_SUBGRAPH_USER_PROMPT
from atlas_rag.retriever.simple_retriever import SimpleGraphRetriever
from atlas_rag.evaluation.evaluation import QAJudger
from networkx import DiGraph
from tqdm import tqdm
try:
    import torch
except Exception:
    torch = None


@dataclass
class RetrievalStepResult:
    """
    For single step inference result, for debugging / analysis.
    """
    num_hops: int
    base_top_k: int
    query: str
    retrieved_subgraph: List[Dict[str, str]]
    raw_response: str
    answerable: bool
    answer: Optional[str] = None


class Reafiner:
    """
    - Minimal / K-hop Retrieve: Retrieve a subgraph from KG based on text vector index (support multi-hop iteration expansion).
    - Answerable Judgement: Judge if the current subgraph is enough to answer the query.
    - Error Abduction: If not answerable, let LLM analyze "why not answerable", summarize redundant / incomplete / incorrect information.
    - Subgraph Expanding & Refined KG Generation: Generate new triples based on the reason of the previous step, update the KG incrementally.
    - In-loop Retrieve: Continue retrieving with the updated KG until answerable or reach the step limit.
    """

    def __init__(
        self,
        data: dict,
        sentence_encoder: BaseEmbeddingModel,
        llm_generator: LLMGenerator,
        base_top_k: int = 5,
        increament_hop: int = 1,
        max_hops: int = 4,
        history_horizon_size: int = 2,
        if_gen_answer: bool = True,
        seed: int = 2026,
    ) -> None:
        """
        - data:           Dictionary containing the following keys:
            - KG:             Complete KG (networkx.DiGraph, at least 'id' / 'type' in node attributes, 'relation' in edge attributes).
            - node_list:      List of node ids in the KG.
            - edge_list:      List of edge tuples in the KG.
            - node_faiss_index: Faiss index for node retrieval.
            - edge_faiss_index: Faiss index for edge retrieval.
            - text_node_dict:   Dictionary mapping node ids to text.
        - sentence_encoder: Encoder corresponding to text_faiss_index, for encoding query.
        - llm_generator:    atlas_rag.llm_generator.LLMGenerator instance.
        - base_top_k:       TopK for text vector retrieval for the 1st step.
        - if_gen_answer:    Whether to generate answer in the refinement process.
        - max_hops:         Maximum number of hops for subgraph expansion.
        - history_horizon_size: Size of the interaction history to be considered for error abduction.
        """
        self.data = data
        self.kg = data["KG"]
        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        self.node_faiss_index = data["node_faiss_index"]
        self.edge_faiss_index = data["edge_faiss_index"]
        self.sentence_encoder = sentence_encoder
        self.llm_generator = llm_generator
        self.retriever = SimpleGraphRetriever(
                            llm_generator=self.llm_generator,
                            sentence_encoder=self.sentence_encoder,
                            data=self.data,
                        )
        self.base_top_k = base_top_k
        self.max_hops = max_hops
        self.increament_hop = increament_hop
        self.history_horizon_size = history_horizon_size
        self.if_gen_answer = if_gen_answer
        self.seed = seed
        self._set_seed(seed)
        self._dim = self.data["text_faiss_index"].d
        # Ensure the order of text_node_dict keys matches the order when building text_index (pickle maintains insertion order by default).
        self._text_node_ids: List[str] = list(self.data["text_dict"].keys())

        self.node_id_to_attr_id = {self.kg.nodes[n]['id']: n for n in self.kg.nodes}
        self.qa_judge = QAJudger()

    # ------------------------------------------------------------------
    # Main interface for external use
    # ------------------------------------------------------------------
    def refine(
        self, query: str, return_steps: bool = False
    ) -> Tuple[str, nx.DiGraph, Optional[List[RetrievalStepResult]]]:
        """
        Run the entire REAfiner process for a single query.

        Returns:
        -------
        - answer:       Answer given by LLM on the final refined KG (possibly abstract natural language).
        - refined_kg:    KG after inserting new knowledge (in-place modification, also returned by reference).
        - steps:        If return_steps=True, return a list of intermediate results for each step, for debugging/visualization.
        """
        interaction_history: List[RetrievalStepResult] = []
        final_answer: str = ""
        base_top_k = self.base_top_k

        for step in range(1, self.max_hops + 1):
            print(f"\033[94m [Step: {step}] \033[0m")
            # top-k Retrieve (retrieve a subgraph on the existing KG with vector search)
            if step == 1:
                # base top-k edges retrieval for the 1st step
                sorted_context, sorted_context_ids = self.retriever.retrieve(query, topN=base_top_k)
            else:
                # expand the sub-graph with k-hop retrieval
                # obtain node ids from the previous step
                node_str_list = []
                for triple_str in sorted_context:
                    head_node_str, rel, tail_node_str = triple_str.split("  ")
                    node_str_list.append(head_node_str)
                    node_str_list.append(tail_node_str)
                node_str_list = sorted(set(node_str_list))
                node_id_list = [self.node_id_to_attr_id.get(node_str, node_str) for node_str in node_str_list]
                # retrieve k-hop subgraph with the given node ids
                subgraph = self._construct_subgraph(node_id_list, num_hop=self.increament_hop)
                # convert subgraph to triple strings
                subgraph_edges = len(subgraph.edges)
                subgraph_triples = sorted([(self.kg.nodes[u]['id'], d['relation'], self.kg.nodes[v]['id']) for u, v, d in subgraph.edges(data=True)])
                sorted_context = [f"{s}  {r}  {o}" for s, r, o in subgraph_triples] 
            retrieved_context = "\n".join(sorted_context)
            retrieved_subgraph = [{"subject": f"{x.split('  ')[0]}", "relation": f"{x.split('  ')[1]}", "object": f"{x.split('  ')[2]}"} for x in sorted_context]

            # Answerable Judgement
            answerable, judgement_raw = self._answerable_judgement(query, retrieved_context)
            if answerable:
                if self.if_gen_answer:
                    final_answer = self._generate_answer(query, retrieved_context)
                    short_answer = self.qa_judge.split_answer(final_answer)
                interaction_history.append(
                    RetrievalStepResult(
                        num_hops=(step - 1) * self.increament_hop,
                        base_top_k=base_top_k,
                        query=query,
                        retrieved_subgraph=retrieved_subgraph,
                        raw_response=judgement_raw,
                        answerable=True,
                        answer=short_answer if self.if_gen_answer else None,
                    )
                )
                break
            else:
                if self.if_gen_answer:
                    final_answer = self._generate_answer(query, retrieved_context)
                    short_answer = self.qa_judge.split_answer(final_answer)
                interaction_history.append(
                    RetrievalStepResult(
                        num_hops=(step - 1) * self.increament_hop,
                        base_top_k=base_top_k,
                        query=query,
                        retrieved_subgraph=retrieved_subgraph,
                        raw_response=judgement_raw,
                        answerable=False,
                        answer=short_answer if self.if_gen_answer else None,
                    )
                )
        # Error Abduction
        error_abduction_reason, error_abduction_raw = self._error_abduction(interaction_history)
        # Refined KG Generation
        refined_subgraph, refined_subgraph_raw = self._kg_refinement(retrieved_subgraph, error_abduction_reason)
        # del original smaller subgraph
        self._del_subgraph(interaction_history[-2].retrieved_subgraph)
        # insert refined larger subgraph
        self._insert_subgraph(refined_subgraph)
        return (interaction_history[-1].answer, self.data, interaction_history if return_steps else None)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Retrieval related
    # ------------------------------------------------------------------
    def _encode_query(self, query: str) -> np.ndarray:
        emb = self.sentence_encoder.encode([query], normalize_embeddings=False)[0]
        emb = np.asarray(emb, dtype="float32").reshape(1, -1)
        # Some encoders may return vectors of different dimensions, here we truncate / pad to match the index dimension
        if emb.shape[1] != self._dim:
            if emb.shape[1] > self._dim:
                emb = emb[:, : self._dim]
            else:
                padded = np.zeros((1, self._dim), dtype="float32")
                padded[:, : emb.shape[1]] = emb
                emb = padded
        faiss.normalize_L2(emb)
        return emb

    # ------------------------------------------------------------------
    # LLM interaction part
    # ------------------------------------------------------------------
    def _answerable_judgement(self, query: str, triples_string: str) -> Tuple[bool, str]:
        """
        Judge if the given question is answerable based on the provided KG context.
        
        Returns:
        -------
        - answerable: bool
        - raw_response: str
        """
        system_prompt = REAFINER_JUDGEMENT_SYSTEM_PROMPT
        user_prompt = REAFINER_JUDGEMENT_USER_PROMPT.format(
            question=query,
            triples_string=triples_string,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm_generator.generate_response(
            messages, temperature=0.0
        )
        print(raw)
        # Parse the output: extract <judge> tag
        judge_match = re.search(r'<judge>(.*?)</judge>', raw, re.IGNORECASE | re.DOTALL)
        if judge_match:
            judge_text = judge_match.group(1).strip().lower()
            answerable = judge_text.startswith("yes")
        else:
            # Fallback: try to find Yes/No in the text
            text_lower = raw.lower()
            if "yes" in text_lower[:100]:
                answerable = True
            elif "no" in text_lower[:100]:
                answerable = False
        return answerable, raw
    
    def _error_abduction(self, interaction_history: List[RetrievalStepResult]) -> Tuple[str, str]:
        """
        Analyze the error reasons based on the given interaction history.
        """
        interaction_history_str = "\n".join(
            [f"Step{i+1}:\n['Query': {result.query}, 'Subgraph_hop': {result.num_hops}, 'Subgraph_content': {str(result.retrieved_subgraph)}, 'Answerable': {result.answerable}]\n" for i, result in enumerate(interaction_history[:-self.history_horizon_size])]
            )
        system_prompt = REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT
        user_prompt = REAFINER_ERROR_ABDUCTION_USER_PROMPT.format(
            interaction_history=interaction_history_str,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm_generator.generate_response(
            messages, temperature=0.0
        )
        # Parse the output: extract <abduction> tag
        abduction_match = re.search(r'<abduction>(.*?)</abduction>', raw, re.IGNORECASE | re.DOTALL)
        if abduction_match:
            reason = abduction_match.group(1).strip()
        else:
            raise ValueError(f"Error Abduction Error Format: {raw}")
        return reason, raw
    
    def _kg_refinement(self, triples_string: str, error_abduction_reason: str) -> Tuple[List[Dict[str, str]], str]:
        """
        Refine the knowledge graph based on the given error reasons.
        """
        system_prompt = REAFINER_KG_REFINEMENT_SYSTEM_PROMPT
        user_prompt = REAFINER_KG_REFINEMENT_USER_PROMPT.format(
            triples_string=triples_string,
            error_reasons=error_abduction_reason,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm_generator.generate_response(
            messages, temperature=0.0
        )
        # Parse the output: extract <refinement> tag
        refinement_match = re.search(r'<refinement>(.*?)</refinement>', raw, re.IGNORECASE | re.DOTALL)
        refined_triples = []
        if refinement_match:
            refinement_json = refinement_match.group(1).strip()
            try:
                triples_list = json.loads(refinement_json)
                for triple in triples_list:
                    if isinstance(triple, dict):
                        subject = triple.get("subject", "").strip()
                        relation = triple.get("relation", "").strip()
                        obj = triple.get("object", "").strip()
                        if subject and relation and obj:
                            refined_triples.append({"subject": subject, "relation": relation, "object": obj})
                    else:
                        # Fallback: try to parse as text format (head | relation | tail)
                        for line in refinement_json.splitlines():
                            line = line.strip()
                            if not line or "|" not in line:
                                continue
                            parts = [p.strip() for p in line.split("|")]
                            if len(parts) == 3:
                                h, r, t = parts
                                if h and r and t:
                                    refined_triples.append({"subject": h, "relation": r, "object": t})
            except json.JSONDecodeError:
                raise ValueError(f"KG Refinement Error Format: {raw}")
        else:
            raise ValueError(f"KG Refinement Error Format: {raw}")
        return refined_triples, raw

    def _generate_answer(self, query: str, subgraph_str: str) -> str:
        """
        Generate the final answer on the final refined subgraph (no Yes/No judgment).
        """
        return self.llm_generator.generate_with_context_kg(query, subgraph_str, temperature=0.0)

    # ------------------------------------------------------------------
    # KG update logic
    # ------------------------------------------------------------------
    def _del_subgraph(self, retrieved_subgraph: List[Dict[str, str]]) -> None:
        """
        Delete the retrieved subgraph from the KG and update corresponding indices.
        """
        # collect the edges and nodes to delete
        edges_to_delete = set()
        nodes_to_delete = set()
        for triple in retrieved_subgraph:
            head_id = self._ensure_node(triple["subject"])
            tail_id = self._ensure_node(triple["object"])
            edges_to_delete.add((head_id, tail_id))
            nodes_to_delete.add(head_id)
            nodes_to_delete.add(tail_id)
        # delete the nodes from the KG (will automatically delete the related edges)
        self.kg.remove_nodes_from(nodes_to_delete)
        # find the indices of the edges to delete in the edge_list
        edge_indices_to_delete = []
        for i, edge in enumerate(self.edge_list):
            if edge in edges_to_delete:
                edge_indices_to_delete.append(i)
        # find the indices of the nodes to delete in the node_list
        node_indices_to_delete = []
        for i, node_id in enumerate(self.node_list):
            if node_id in nodes_to_delete:
                node_indices_to_delete.append(i)
        # delete the edges and edge_embeddings from the edge_list and edge_embeddings
        edge_embeddings = self.data["edge_embeddings"]
        for idx in sorted(edge_indices_to_delete, reverse=True):
            del self.edge_list[idx]
            del edge_embeddings[idx]
        # delete the nodes and node_embeddings from the node_list and node_embeddings
        node_embeddings = self.data["node_embeddings"]
        for idx in sorted(node_indices_to_delete, reverse=True):
            del self.node_list[idx]
            del node_embeddings[idx]
        
        # rebuild faiss indices from updated embeddings (simple and reliable method)
        # IndexHNSWFlat doesn't support remove_ids, so we rebuild the index
        def rebuild_index(embeddings_list, original_index):
            """Rebuild index from embeddings list, matching build_faiss_index style"""
            if len(embeddings_list) == 0:
                # return empty index with same dimension
                dim = original_index.d
                if isinstance(original_index, faiss.IndexHNSWFlat):
                    return faiss.IndexHNSWFlat(dim, 64, faiss.METRIC_INNER_PRODUCT)
                else:
                    return faiss.IndexFlatL2(dim)
            
            # rebuild index matching build_faiss_index
            vectors = np.array(embeddings_list, dtype=np.float32)
            dim = vectors.shape[1]
            
            # match build_faiss_index: IndexHNSWFlat with METRIC_INNER_PRODUCT
            if isinstance(original_index, faiss.IndexHNSWFlat):
                new_index = faiss.IndexHNSWFlat(dim, 64, faiss.METRIC_INNER_PRODUCT)
            else:
                new_index = faiss.IndexFlatL2(dim)
            
            # normalize L2 (same as build_faiss_index)
            faiss.normalize_L2(vectors)
            
            # batch add (same as build_faiss_index, batch size 32)
            for i in tqdm(range(0, vectors.shape[0], 32), desc="Rebuilding index"):
                new_index.add(vectors[i:i+32])
            
            return new_index
        
        # rebuild edge index
        if edge_embeddings:
            self.edge_faiss_index = rebuild_index(edge_embeddings, self.edge_faiss_index)
        else:
            dim = self.edge_faiss_index.d
            if isinstance(self.edge_faiss_index, faiss.IndexHNSWFlat):
                self.edge_faiss_index = faiss.IndexHNSWFlat(dim, 64, faiss.METRIC_INNER_PRODUCT)
            else:
                self.edge_faiss_index = faiss.IndexFlatL2(dim)
        
        # rebuild node index
        if node_embeddings:
            self.node_faiss_index = rebuild_index(node_embeddings, self.node_faiss_index)
        else:
            dim = self.node_faiss_index.d
            if isinstance(self.node_faiss_index, faiss.IndexHNSWFlat):
                self.node_faiss_index = faiss.IndexHNSWFlat(dim, 64, faiss.METRIC_INNER_PRODUCT)
            else:
                self.node_faiss_index = faiss.IndexFlatL2(dim)
        # delete text-related data if text_dict exists
        # NOTE: check text nodes BEFORE deleting from KG, since nodes are already deleted above
        if "text_dict" in self.data and self.data["text_dict"]:
            text_dict = self.data["text_dict"]
            text_embeddings = self.data.get("text_embeddings", [])
            text_faiss_index = self.data.get("text_faiss_index")
            # find text nodes to delete (nodes with type "passage" that are in nodes_to_delete)
            # check nodes before they were deleted from KG
            text_node_ids_to_delete = []
            # we need to check nodes before deletion, so we'll check if they exist in text_dict and verify they were in nodes_to_delete
            for text_node_id in list(text_dict.keys()):
                if text_node_id in nodes_to_delete:
                    text_node_ids_to_delete.append(text_node_id)
            # find indices of text embeddings to delete text_embeddings order matches text_dict iteration order
            text_indices_to_delete = []
            for idx, (text_node_id, text_content) in enumerate(text_dict.items()):
                if text_node_id in text_node_ids_to_delete:
                    text_indices_to_delete.append(idx)
            # delete from text_dict
            for text_node_id in text_node_ids_to_delete:
                if text_node_id in text_dict:
                    del text_dict[text_node_id]
            # delete from text_embeddings (from back to front to avoid index shifting)
            for idx in sorted(text_indices_to_delete, reverse=True):
                if idx < len(text_embeddings):
                    del text_embeddings[idx]
            # rebuild text_faiss_index if it exists
            if text_faiss_index is not None:
                text_faiss_index = rebuild_index(text_embeddings, text_faiss_index)
            # update data
            self.data["text_dict"] = text_dict
            self.data["text_embeddings"] = text_embeddings
            if text_faiss_index is not None:
                self.data["text_faiss_index"] = text_faiss_index
        # update the lists in the data
        self.data["KG"] = self.kg
        self.data["node_faiss_index"] = self.node_faiss_index
        self.data["edge_faiss_index"] = self.edge_faiss_index
        self.data["edge_embeddings"] = edge_embeddings
        self.data["node_embeddings"] = node_embeddings
        self.data["edge_list"] = self.edge_list
        self.data["node_list"] = self.node_list        

    def _insert_subgraph(
        self, refined_subgraph: List[Dict[str, str]]
    ) -> None:
        """
        Insert the refined subgraph into the KG and update corresponding indices.
        - If the entity node does not exist, create a new node, type is 'entity' by default.
        - If the edge already exists, only update the relation attribute.
        - Update edge_list, node_list, embeddings, and faiss indices.
        """
        # helper function to ensure index is wrapped with IndexIDMap
        def ensure_idmap(index, embeddings_list):
            """Wrap index with IndexIDMap if not already wrapped, matching build_faiss_index style"""
            if isinstance(index, faiss.IndexIDMap):
                return index
            elif isinstance(index, faiss.IndexIDMap2):
                return index
            else:
                base_index = index
                dim = base_index.d
                if isinstance(base_index, faiss.IndexHNSWFlat):
                    idmap = faiss.IndexIDMap(faiss.IndexHNSWFlat(dim, 64, faiss.METRIC_INNER_PRODUCT))
                else:
                    idmap = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
                if len(embeddings_list) > 0:
                    vectors = np.array(embeddings_list, dtype=np.float32)
                    faiss.normalize_L2(vectors)
                    ids = np.arange(len(embeddings_list), dtype=np.int64)
                    for i in range(0, vectors.shape[0], 32):
                        idmap.add_with_ids(vectors[i:i+32], ids[i:i+32])
                return idmap
        
        # collect new edges and nodes to insert
        new_edges = []
        new_nodes = set()
        new_text_nodes = []  # nodes with type "passage"
        
        for triple in refined_subgraph:
            head_id = self._ensure_node(triple["subject"])
            tail_id = self._ensure_node(triple["object"])
            
            # check if nodes are new
            if head_id not in self.node_list:
                new_nodes.add(head_id)
                # check if it's a text node
                if head_id in self.kg.nodes and "passage" in self.kg.nodes[head_id].get("type", ""):
                    new_text_nodes.append(head_id)
            if tail_id not in self.node_list:
                new_nodes.add(tail_id)
                # check if it's a text node
                if tail_id in self.kg.nodes and "passage" in self.kg.nodes[tail_id].get("type", ""):
                    new_text_nodes.append(tail_id)
            
            # add or update edge in KG
            if self.kg.has_edge(head_id, tail_id):
                self.kg.edges[head_id, tail_id]["relation"] = triple["relation"]
            else:
                self.kg.add_edge(head_id, tail_id, relation=triple["relation"])
                # only add to new_edges if it's truly new
                if (head_id, tail_id) not in self.edge_list:
                    new_edges.append((head_id, tail_id, triple["relation"]))
        
        # generate embeddings for new edges
        if new_edges:
            edge_embeddings = self.data["edge_embeddings"]
            edge_list_string = [
                f"{self.kg.nodes[edge[0]]['id']} {edge[2]} {self.kg.nodes[edge[1]]['id']}"
                for edge in new_edges
            ]
            new_edge_embeddings = self.sentence_encoder.encode(edge_list_string, query_type='edge')
            if isinstance(new_edge_embeddings, torch.Tensor):
                new_edge_embeddings = new_edge_embeddings.cpu().numpy()
            
            # add to edge_list and edge_embeddings
            for edge, emb in zip(new_edges, new_edge_embeddings):
                self.edge_list.append(edge[:2])  # only (head, tail) tuple
                edge_embeddings.append(emb.tolist())
            
            # add new edge embeddings to index (simple method: just use add)
            new_edge_vectors = np.array(new_edge_embeddings, dtype=np.float32)
            faiss.normalize_L2(new_edge_vectors)
            # batch add (batch size 32, matching build_faiss_index)
            for i in tqdm(range(0, new_edge_vectors.shape[0], 32), desc="Adding new edge embeddings to index"):
                self.edge_faiss_index.add(new_edge_vectors[i:i+32])
            
            self.data["edge_embeddings"] = edge_embeddings
            self.data["edge_list"] = self.edge_list
            self.data["edge_faiss_index"] = self.edge_faiss_index
        
        # generate embeddings for new nodes
        if new_nodes:
            node_embeddings = self.data["node_embeddings"]
            # convert set to list to maintain order
            new_nodes_list = list(new_nodes)
            node_list_string = [self.kg.nodes[node_id]["id"] for node_id in new_nodes_list]
            new_node_embeddings = self.sentence_encoder.encode(node_list_string, query_type='node')
            if isinstance(new_node_embeddings, torch.Tensor):
                new_node_embeddings = new_node_embeddings.cpu().numpy()
            
            # add to node_list and node_embeddings
            for node_id, emb in zip(new_nodes_list, new_node_embeddings):
                self.node_list.append(node_id)
                node_embeddings.append(emb.tolist())
            
            # add new node embeddings to index (simple method: just use add)
            new_node_vectors = np.array(new_node_embeddings, dtype=np.float32)
            faiss.normalize_L2(new_node_vectors)
            # batch add (batch size 32, matching build_faiss_index)
            for i in tqdm(range(0, new_node_vectors.shape[0], 32), desc="Adding new node embeddings to index"):
                self.node_faiss_index.add(new_node_vectors[i:i+32])
            
            self.data["node_embeddings"] = node_embeddings
            self.data["node_list"] = self.node_list
            self.data["node_faiss_index"] = self.node_faiss_index
        
        # handle new text nodes
        if new_text_nodes and "text_dict" in self.data:
            text_dict = self.data["text_dict"]
            text_embeddings = self.data.get("text_embeddings", [])
            text_faiss_index = self.data.get("text_faiss_index")
            
            # generate text embeddings for new text nodes
            text_list_string = [self.kg.nodes[node_id]["id"] for node_id in new_text_nodes]
            new_text_embeddings = self.sentence_encoder.encode(text_list_string, query_type='passage')
            if isinstance(new_text_embeddings, torch.Tensor):
                new_text_embeddings = new_text_embeddings.cpu().numpy()
            
            # add to text_dict and text_embeddings
            for node_id, text_content, emb in zip(new_text_nodes, text_list_string, new_text_embeddings):
                text_dict[node_id] = text_content
                text_embeddings.append(emb.tolist())
            
            # add to text_faiss_index if it exists (simple method: just use add)
            if text_faiss_index is not None:
                new_text_vectors = np.array(new_text_embeddings, dtype=np.float32)
                faiss.normalize_L2(new_text_vectors)
                # batch add (batch size 32, matching build_faiss_index)
                for i in tqdm(range(0, new_text_vectors.shape[0], 32), desc="Adding new text embeddings to index"):
                    text_faiss_index.add(new_text_vectors[i:i+32])
                self.data["text_faiss_index"] = text_faiss_index
            
            self.data["text_dict"] = text_dict
            self.data["text_embeddings"] = text_embeddings
        
        # update KG in data
        self.data["KG"] = self.kg

    def _ensure_node(self, entity_str: str) -> str:
        """
        Find / create a node in the graph based on the entity string.
        Simplified strategy: If the node id == entity_str or the node attribute 'id' == entity_str, use it directly;
        Otherwise create a new node with entity_str as id.
        """
        # Directly use id matching
        if entity_str in self.kg:
            return entity_str
        # Match by node attribute 'id'
        for nid, data in self.kg.nodes(data=True):
            if data.get("id") == entity_str:
                return nid

        # If neither exists, create a new node, id is directly entity_str
        node_id = entity_str
        self.kg.add_node(node_id, id=entity_str, type="entity")
        return node_id
    
    def _construct_subgraph(self, initial_nodes, num_hop: int = 1):
        """Construct a multi-hop subgraph around initial nodes up to num_hop."""
        subgraph = DiGraph()
        visited = set()
        queue = [(node, 0) for node in initial_nodes if node in self.node_list]

        # Add initial nodes
        for node, _ in queue:
            subgraph.add_node(node)
            visited.add(node)

        # Breadth-first search to collect neighbors
        while queue:
            current_node, hop_count = queue.pop(0)
            if hop_count >= num_hop:
                continue
            # Add successors (outgoing edges)
            for neighbor in sorted(self.kg.successors(current_node)):
                neighbor_id = self.kg.nodes[neighbor].get('id', None)
                if neighbor_id.isdigit():
                    # Do not further explore this neighbor
                    relation = self.kg.edges[(current_node, neighbor)]["relation"]
                    subgraph.add_edge(current_node, neighbor, relation=relation)
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.kg.edges[(current_node, neighbor)]["relation"]
                subgraph.add_edge(current_node, neighbor, relation=relation)

            # Add predecessors (incoming edges)
            for neighbor in sorted(self.kg.predecessors(current_node)):
                neighbor_id = self.kg.nodes[neighbor].get('id', None)
                if neighbor_id.isdigit():
                    # Do not further explore this neighbor
                    relation = self.kg.edges[(neighbor, current_node)]["relation"]
                    subgraph.add_edge(neighbor, current_node, relation=relation)
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.kg.edges[(neighbor, current_node)]["relation"]
                subgraph.add_edge(neighbor, current_node, relation=relation)

        return subgraph


__all__ = ["Reafiner", "RetrievalStepResult"]