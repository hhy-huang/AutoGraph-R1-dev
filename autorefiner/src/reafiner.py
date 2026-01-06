from __future__ import annotations
import random
import re
import json
import networkx as nx
import numpy as np
import faiss
import json_repair
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

@dataclass
class RefinementResult:
    """
    For single refinement result, for debugging / analysis.
    """
    query: str
    history_horizon_size: int
    interaction_history: List[RetrievalStepResult]
    error_abduction_reason: str
    original_subgraph: List[Dict[str, str]]
    refined_subgraph: List[Dict[str, str]]


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
        max_triple_num: int=25,
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
        - max_triple_num:   Maximum number of triples for subgraph pruning.
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
        self.max_triple_num = max_triple_num
        self.seed = seed
        self._set_seed(seed)
        self._dim = self.data["text_faiss_index"].d
        # Ensure the order of text_node_dict keys matches the order when building text_index (pickle maintains insertion order by default).
        self._text_node_ids: List[str] = list(self.data["text_dict"].keys())

        self.node_id_to_attr_id = {self.kg.nodes[n]['id']: n for n in self.kg.nodes}
        self.qa_judge = QAJudger()
        
        # Initialize ID mapping tables (faiss_id -> list_index) for incremental updates without rebuild
        # If mapping doesn't exist, create identity mapping (initial state: faiss_id == list_index)
        if "edge_faiss_id_to_list_idx" not in self.data:
            self.data["edge_faiss_id_to_list_idx"] = {i: i for i in range(len(self.edge_list))}
        if "node_faiss_id_to_list_idx" not in self.data:
            self.data["node_faiss_id_to_list_idx"] = {i: i for i in range(len(self.node_list))}
        if "text_faiss_id_to_list_idx" not in self.data and "text_dict" in self.data:
            self.data["text_faiss_id_to_list_idx"] = {i: i for i in range(len(self.data["text_dict"]))}
        
        self.edge_faiss_id_to_list_idx = self.data["edge_faiss_id_to_list_idx"]
        self.node_faiss_id_to_list_idx = self.data["node_faiss_id_to_list_idx"]
        if "text_faiss_id_to_list_idx" in self.data:
            self.text_faiss_id_to_list_idx = self.data["text_faiss_id_to_list_idx"]
        else:
            self.text_faiss_id_to_list_idx = {}

    # ------------------------------------------------------------------
    # Main interface for external use
    # ------------------------------------------------------------------
    def refine(
        self, query: str,
    ) -> Tuple[str, nx.DiGraph, Optional[RefinementResult]]:
        """
        Run the entire REAfiner process for a single query.

        Returns:
        -------
        - answer:             Answer given by LLM on the final refined KG (possibly abstract natural language).
        - refined_kg:         KG after inserting new knowledge (in-place modification, also returned by reference).
        - refinement_result:  Refinement result containing interaction history, original subgraph, and refined subgraph.
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
                    if len(triple_str.split("  ")) != 3:
                        print(f"Error: triple string {triple_str} is not in the correct format")
                        continue
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
                if len(subgraph_triples) > self.max_triple_num:
                    sorted_context = self._prune_subgraph(subgraph_triples, query)
            retrieved_context = "\n".join(sorted_context)
            retrieved_subgraph = [{"subject": f"{x.split('  ')[0]}", "relation": f"{x.split('  ')[1]}", "object": f"{x.split('  ')[2]}"} for x in sorted_context]

            # Answerable Judgement
            answerable, judgement_raw = self._answerable_judgement(query, retrieved_context)
            if judgement_raw is None:
                # fallback
                interaction_history.append(
                    RetrievalStepResult(
                        num_hops=(step - 1) * self.increament_hop,
                        base_top_k=base_top_k,
                        query=query,
                        retrieved_subgraph=retrieved_subgraph,
                        raw_response=None,
                        answerable=answerable,
                        answer=None,
                    )
                )
                refinement_result = RefinementResult(
                    query=query,
                    history_horizon_size=self.history_horizon_size,
                    interaction_history=interaction_history,
                    error_abduction_reason=None,
                    original_subgraph=retrieved_subgraph,
                    refined_subgraph=None,
                )
                return (interaction_history[-1].answer, self.data, refinement_result)

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
        if len(interaction_history) <= 1:
            # 1-hop is enough to answer the query
            refinement_result = RefinementResult(
                query=query,
                history_horizon_size=self.history_horizon_size,
                interaction_history=interaction_history,
                error_abduction_reason=None,
                original_subgraph=interaction_history[-1].retrieved_subgraph,
                refined_subgraph=None,
            )
            return (interaction_history[-1].answer, self.data, refinement_result)
        else:
            # Error Abduction
            error_abduction_reason, error_abduction_raw = self._error_abduction(interaction_history)
            if error_abduction_reason is None:
                # fallback
                refinement_result = RefinementResult(
                    query=query,
                    history_horizon_size=self.history_horizon_size,
                    interaction_history=interaction_history,
                    error_abduction_reason=error_abduction_reason,
                    original_subgraph=interaction_history[-1].retrieved_subgraph,
                    refined_subgraph=None,
                )
                return (interaction_history[-1].answer, self.data, refinement_result)
            # Refined KG Generation
            refined_subgraph, refined_subgraph_raw = self._kg_refinement(interaction_history[-1].retrieved_subgraph, error_abduction_reason)
            if refined_subgraph_raw is None:
                # fallback
                refinement_result = RefinementResult(
                    query=query,
                    history_horizon_size=self.history_horizon_size,
                    interaction_history=interaction_history,
                    error_abduction_reason=error_abduction_reason,
                    original_subgraph=interaction_history[-1].retrieved_subgraph,
                    refined_subgraph=refined_subgraph,
                )
                return (interaction_history[-1].answer, self.data, refinement_result)
            # del original smaller subgraph
            self._del_subgraph(interaction_history[-2].retrieved_subgraph)
            # insert refined larger subgraph
            self._insert_subgraph(refined_subgraph)
            # summarize the refinement result
            refinement_result = RefinementResult(
                query=query,
                history_horizon_size=self.history_horizon_size,
                interaction_history=interaction_history,
                error_abduction_reason=error_abduction_reason,
                original_subgraph=interaction_history[-1].retrieved_subgraph,
                refined_subgraph=refined_subgraph,
            )
            return (interaction_history[-1].answer, self.data, refinement_result)

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
    
    def _prune_subgraph(self, subgraph_triples: List[str], query: str) -> List[str]:
        """
        Prune the subgraph based on the given query.
        """
        # convert subgraph triples to edge strings for encoding (same format as simple_retriever)
        subgraph_edge_strings = [f"{s} {r} {o}" for s, r, o in subgraph_triples]
        # encode query and subgraph edges
        query_embedding = self.sentence_encoder.encode([query], query_type='edge')
        edge_embeddings = self.sentence_encoder.encode(subgraph_edge_strings, query_type='edge')
        if isinstance(edge_embeddings, torch.Tensor):
            edge_embeddings = edge_embeddings.cpu().numpy()
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        # compute similarity scores (same as simple_retriever's faiss search)
        query_emb = query_embedding[0]
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        # normalize for cosine similarity (same as build_faiss_index)
        faiss.normalize_L2(query_emb)
        faiss.normalize_L2(edge_embeddings)
        # compute cosine similarity
        similarities = edge_embeddings @ query_emb.T
        # get top-k indices
        topk_indices = np.argsort(similarities.flatten())[-self.max_triple_num:][::-1]
        # filter subgraph_triples and sorted_context
        pruned_subgraph_triples = [f"{subgraph_triples[i][0]}  {subgraph_triples[i][1]}  {subgraph_triples[i][2]}" for i in topk_indices]
        return pruned_subgraph_triples

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
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0.0, max_new_tokens=256
            )
        except Exception as e:
            # fallback
            error_message = {"error": f"Answerable Judgement Generation Error: {e}"}
            print(error_message)
            return error_message['error'], None

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
            else:
                error_message = [{"error": f"Answerable Judgement Error Format: {raw}"}]
                print(error_message)
                return error_message[0]['error'], None
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
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0.0
            )
        except Exception as e:
            # fallback
            error_message = [{"error": f"Abduction Generation Error: {e}"}]
            print(error_message)
            return error_message[0]['error'], None

        # Parse the output: extract <abduction> tag
        abduction_match = re.search(r'<abduction>(.*?)</abduction>', raw, re.IGNORECASE | re.DOTALL)
        if abduction_match:
            reason = abduction_match.group(1).strip()
        else:
            # fallback
            error_message = [{"error": f"Error Abduction Error Format: {raw}"}]
            print(error_message)
            return error_message[0]['error'], None
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
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0.0
            )
        except Exception as e:
            # fallback
            error_message = [{"error": f"Refinement Generation Error: {e}"}]
            print(error_message)
            return error_message, None

        # Parse the output: extract <refinement> tag
        refinement_match = re.search(r'<refinement>(.*?)</refinement>', raw, re.IGNORECASE | re.DOTALL)
        refined_triples = []
        if refinement_match:
            refinement_json = refinement_match.group(1).strip()
            try:
                # try normal json.loads first
                refined_triples = json.loads(refinement_json)
            except json.JSONDecodeError:
                # if fails, try json_repair to fix common JSON issues (like invalid escape sequences)
                try:
                    refined_triples = json_repair.loads(refinement_json)
                except Exception:
                    # fallback
                    error_message = [{"error": f"KG Refinement Error Format: {raw}"}]
                    print(error_message)
                    return error_message, None
        else:
            # try to extract JSON from raw text if <refinement> tag not found
            if '<refinement>' in raw:
                refinement_json = raw.split('<refinement>')[1].split('</refinement>')[0].strip()
            else:
                # fallback
                error_message = [{"error": f"KG Refinement Error Format: {raw}"}]
                print(error_message)
                return error_message, None
            try:
                # try normal json.loads first
                refined_triples = json.loads(refinement_json)
            except json.JSONDecodeError:
                # if fails, try json_repair to fix common JSON issues (like invalid escape sequences)
                try:
                    refined_triples = json_repair.loads(refinement_json)
                except Exception:
                    # fallback
                    error_message = [{"error": f"KG Refinement Error Format: {raw}"}]
                    print(error_message)
                    return error_message, None
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
            head_id = triple["subject"]
            tail_id = triple["object"]
            edges_to_delete.add((head_id, tail_id))
            nodes_to_delete.add(head_id)
            nodes_to_delete.add(tail_id)
        # delete the nodes from the KG (will automatically delete the related edges)
        self.kg.remove_nodes_from(nodes_to_delete)
        
        # find all edges that no longer exist in KG (including those deleted due to node removal)
        # this is important because removing nodes automatically removes all connected edges
        edge_indices_to_delete = []
        edge_embeddings = self.data["edge_embeddings"]
        for i in range(len(self.edge_list) - 1, -1, -1):  # iterate backwards to safely delete
            edge = self.edge_list[i]
            # check if edge still exists in KG (both nodes exist and edge exists)
            if (edge[0] not in self.kg.nodes or 
                edge[1] not in self.kg.nodes or 
                not self.kg.has_edge(edge[0], edge[1]) or
                edge in edges_to_delete):
                edge_indices_to_delete.append(i)
        
        # find the indices of the nodes to delete in the node_list
        node_indices_to_delete = []
        for i, node_id in enumerate(self.node_list):
            if node_id in nodes_to_delete:
                node_indices_to_delete.append(i)
        
        # Use remove_ids for incremental deletion (no rebuild needed)
        # Find FAISS IDs corresponding to list indices to delete
        edge_faiss_ids_to_remove = []
        for list_idx in edge_indices_to_delete:
            # Find FAISS ID that maps to this list index
            for faiss_id, mapped_list_idx in self.edge_faiss_id_to_list_idx.items():
                if mapped_list_idx == list_idx:
                    edge_faiss_ids_to_remove.append(faiss_id)
                    break
        
        node_faiss_ids_to_remove = []
        for list_idx in node_indices_to_delete:
            # Find FAISS ID that maps to this list index
            for faiss_id, mapped_list_idx in self.node_faiss_id_to_list_idx.items():
                if mapped_list_idx == list_idx:
                    node_faiss_ids_to_remove.append(faiss_id)
                    break
        
        # Remove from FAISS index
        if edge_faiss_ids_to_remove:
            edge_ids_to_remove = np.array(edge_faiss_ids_to_remove, dtype=np.int64)
            self.edge_faiss_index.remove_ids(edge_ids_to_remove)
            # Remove from mapping table
            for faiss_id in edge_faiss_ids_to_remove:
                del self.edge_faiss_id_to_list_idx[faiss_id]
            # Update remaining mappings: decrease list_idx for items after deleted ones
            for faiss_id in self.edge_faiss_id_to_list_idx:
                for deleted_list_idx in sorted(edge_indices_to_delete, reverse=True):
                    if self.edge_faiss_id_to_list_idx[faiss_id] > deleted_list_idx:
                        self.edge_faiss_id_to_list_idx[faiss_id] -= 1
        
        if node_faiss_ids_to_remove:
            node_ids_to_remove = np.array(node_faiss_ids_to_remove, dtype=np.int64)
            self.node_faiss_index.remove_ids(node_ids_to_remove)
            # Remove from mapping table
            for faiss_id in node_faiss_ids_to_remove:
                del self.node_faiss_id_to_list_idx[faiss_id]
            # Update remaining mappings: decrease list_idx for items after deleted ones
            for faiss_id in self.node_faiss_id_to_list_idx:
                for deleted_list_idx in sorted(node_indices_to_delete, reverse=True):
                    if self.node_faiss_id_to_list_idx[faiss_id] > deleted_list_idx:
                        self.node_faiss_id_to_list_idx[faiss_id] -= 1
        
        # Delete from lists (from back to front to avoid index shifting)
        for idx in sorted(edge_indices_to_delete, reverse=True):
            del self.edge_list[idx]
            del edge_embeddings[idx]
        
        # delete the nodes and node_embeddings from the node_list and node_embeddings
        node_embeddings = self.data["node_embeddings"]
        for idx in sorted(node_indices_to_delete, reverse=True):
            del self.node_list[idx]
            del node_embeddings[idx]
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
            # Use remove_ids for incremental deletion of text index
            if text_faiss_index is not None and text_indices_to_delete:
                # Find FAISS IDs corresponding to list indices to delete
                text_faiss_ids_to_remove = []
                for list_idx in text_indices_to_delete:
                    for faiss_id, mapped_list_idx in self.text_faiss_id_to_list_idx.items():
                        if mapped_list_idx == list_idx:
                            text_faiss_ids_to_remove.append(faiss_id)
                            break
                
                if text_faiss_ids_to_remove:
                    text_ids_to_remove = np.array(text_faiss_ids_to_remove, dtype=np.int64)
                    text_faiss_index.remove_ids(text_ids_to_remove)
                    # Remove from mapping table
                    for faiss_id in text_faiss_ids_to_remove:
                        del self.text_faiss_id_to_list_idx[faiss_id]
                    # Update remaining mappings: decrease list_idx for items after deleted ones
                    for faiss_id in self.text_faiss_id_to_list_idx:
                        for deleted_list_idx in sorted(text_indices_to_delete, reverse=True):
                            if self.text_faiss_id_to_list_idx[faiss_id] > deleted_list_idx:
                                self.text_faiss_id_to_list_idx[faiss_id] -= 1
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
        # Update mapping tables in data
        self.data["edge_faiss_id_to_list_idx"] = self.edge_faiss_id_to_list_idx
        self.data["node_faiss_id_to_list_idx"] = self.node_faiss_id_to_list_idx
        if hasattr(self, 'text_faiss_id_to_list_idx'):
            self.data["text_faiss_id_to_list_idx"] = self.text_faiss_id_to_list_idx
        
        # update retriever's references to ensure it uses the latest data
        self.retriever.KG = self.kg
        self.retriever.edge_list = self.edge_list
        self.retriever.node_list = self.node_list
        self.retriever.edge_faiss_index = self.edge_faiss_index
        self.retriever.node_faiss_index = self.node_faiss_index        

    def _insert_subgraph(
        self, refined_subgraph: List[Dict[str, str]]
    ) -> None:
        """
        Insert the refined subgraph into the KG and update corresponding indices.
        - If the entity node does not exist, create a new node, type is 'entity' by default.
        - If the edge already exists, only update the relation attribute.
        - Update edge_list, node_list, embeddings, and faiss indices.
        """
        # IndexIDMap is now used by default (from build_faiss_index), no need to wrap
        
        # collect new edges and nodes to insert
        new_edges = []
        new_nodes = set()
        new_text_nodes = []  # nodes with type "passage"
        
        for triple in refined_subgraph:
            if "subject" in triple and "object" in triple and "relation" in triple:
                head_id = self._ensure_node(triple["subject"])
                tail_id = self._ensure_node(triple["object"])
            else:
                print(f"Error: subject or object or relation not in triple {triple}")
                continue
            
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
            
            # calculate start_list_idx and start_faiss_id before adding to lists
            start_list_idx = len(self.edge_list)
            # Use max existing FAISS ID + 1 as start, or 0 if empty
            start_faiss_id = max(self.edge_faiss_id_to_list_idx.keys()) + 1 if self.edge_faiss_id_to_list_idx else 0
            
            # add to edge_list and edge_embeddings
            for edge, emb in zip(new_edges, new_edge_embeddings):
                self.edge_list.append(edge[:2])  # only (head, tail) tuple
                edge_embeddings.append(emb.tolist())
            
            # add new edge embeddings to index with IDs (incremental update)
            new_edge_vectors = np.array(new_edge_embeddings, dtype=np.float32)
            faiss.normalize_L2(new_edge_vectors)
            # batch add with IDs and update mapping table
            for i in range(0, new_edge_vectors.shape[0], 32):
                batch_end = min(i + 32, new_edge_vectors.shape[0])
                faiss_ids = np.arange(start_faiss_id + i, start_faiss_id + batch_end, dtype=np.int64)
                list_indices = range(start_list_idx + i, start_list_idx + batch_end)
                self.edge_faiss_index.add_with_ids(new_edge_vectors[i:batch_end], faiss_ids)
                # Update mapping table
                for faiss_id, list_idx in zip(faiss_ids, list_indices):
                    self.edge_faiss_id_to_list_idx[int(faiss_id)] = list_idx
            
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
            
            # calculate start_list_idx and start_faiss_id before adding to lists
            start_list_idx = len(self.node_list)
            # Use max existing FAISS ID + 1 as start, or 0 if empty
            start_faiss_id = max(self.node_faiss_id_to_list_idx.keys()) + 1 if self.node_faiss_id_to_list_idx else 0
            
            # add to node_list and node_embeddings
            for node_id, emb in zip(new_nodes_list, new_node_embeddings):
                self.node_list.append(node_id)
                node_embeddings.append(emb.tolist())
            
            # add new node embeddings to index with IDs (incremental update)
            new_node_vectors = np.array(new_node_embeddings, dtype=np.float32)
            faiss.normalize_L2(new_node_vectors)
            # batch add with IDs and update mapping table
            for i in range(0, new_node_vectors.shape[0], 32):
                batch_end = min(i + 32, new_node_vectors.shape[0])
                faiss_ids = np.arange(start_faiss_id + i, start_faiss_id + batch_end, dtype=np.int64)
                list_indices = range(start_list_idx + i, start_list_idx + batch_end)
                self.node_faiss_index.add_with_ids(new_node_vectors[i:batch_end], faiss_ids)
                # Update mapping table
                for faiss_id, list_idx in zip(faiss_ids, list_indices):
                    self.node_faiss_id_to_list_idx[int(faiss_id)] = list_idx
            
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
            
            # calculate start_list_idx and start_faiss_id before adding to lists
            start_list_idx = len(text_embeddings)
            # Use max existing FAISS ID + 1 as start, or 0 if empty
            start_faiss_id = max(self.text_faiss_id_to_list_idx.keys()) + 1 if self.text_faiss_id_to_list_idx else 0
            
            # add to text_dict and text_embeddings
            for node_id, text_content, emb in zip(new_text_nodes, text_list_string, new_text_embeddings):
                text_dict[node_id] = text_content
                text_embeddings.append(emb.tolist())
            
            # add to text_faiss_index if it exists (incremental update with IDs)
            if text_faiss_index is not None:
                new_text_vectors = np.array(new_text_embeddings, dtype=np.float32)
                faiss.normalize_L2(new_text_vectors)
                # batch add with IDs and update mapping table
                for i in range(0, new_text_vectors.shape[0], 32):
                    batch_end = min(i + 32, new_text_vectors.shape[0])
                    faiss_ids = np.arange(start_faiss_id + i, start_faiss_id + batch_end, dtype=np.int64)
                    list_indices = range(start_list_idx + i, start_list_idx + batch_end)
                    text_faiss_index.add_with_ids(new_text_vectors[i:batch_end], faiss_ids)
                    # Update mapping table
                    for faiss_id, list_idx in zip(faiss_ids, list_indices):
                        self.text_faiss_id_to_list_idx[int(faiss_id)] = list_idx
                self.data["text_faiss_index"] = text_faiss_index
            
            self.data["text_dict"] = text_dict
            self.data["text_embeddings"] = text_embeddings
        
        # update KG in data
        self.data["KG"] = self.kg
        
        # Update mapping tables in data
        self.data["edge_faiss_id_to_list_idx"] = self.edge_faiss_id_to_list_idx
        self.data["node_faiss_id_to_list_idx"] = self.node_faiss_id_to_list_idx
        if hasattr(self, 'text_faiss_id_to_list_idx'):
            self.data["text_faiss_id_to_list_idx"] = self.text_faiss_id_to_list_idx
        
        # update retriever's references to ensure it uses the latest data
        self.retriever.KG = self.kg
        self.retriever.edge_list = self.edge_list
        self.retriever.node_list = self.node_list
        self.retriever.edge_faiss_index = self.edge_faiss_index
        self.retriever.node_faiss_index = self.node_faiss_index
        # Pass mapping tables to retriever
        self.retriever.edge_faiss_id_to_list_idx = self.edge_faiss_id_to_list_idx
        self.retriever.node_faiss_id_to_list_idx = self.node_faiss_id_to_list_idx

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


__all__ = ["Reafiner", "RetrievalStepResult", "RefinementResult"]