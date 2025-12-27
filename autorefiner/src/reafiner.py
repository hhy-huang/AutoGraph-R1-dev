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
from autograph.rag_server.reafiner_prompt import REFINE_SUBGRAPH_SYSTEM_PROMPT, REFINE_SUBGRAPH_USER_PROMPT
from atlas_rag.evaluation.evaluation import QAJudger
from networkx import DiGraph
try:
    import torch
except Exception:
    torch = None


@dataclass
class ReafinerStepResult:
    """
    For single step inference result, for debugging / analysis.
    """
    step_id: int
    answerable: bool
    reason: str
    raw_response: str
    retrieved_context: str
    refined_triples: List[Tuple[str, str, str]]
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
        kg: nx.DiGraph,
        text_faiss_index: faiss.Index,
        text_node_dict: Dict[str, str],
        node_list: List[str],
        sentence_encoder: BaseEmbeddingModel,
        llm_generator: LLMGenerator,
        retriever: BaseEdgeRetriever,
        base_top_k: int = 5,
        increament_hop: int = 1,
        max_hops: int = 5,
        seed: int = 2026,
    ) -> None:
        """
        - kg:          Complete KG (networkx.DiGraph, at least 'id' / 'type' in node attributes, 'relation' in edge attributes).
        - text_faiss_index: text_faiss_index returned by create_graph_index.
        - text_node_dict:   "text_dict" (node_id -> text) returned by create_graph_index.
        - node_list:        List of node ids in the KG.
        - sentence_encoder: Encoder corresponding to text_faiss_index, for encoding query.
        - llm_generator:    atlas_rag.llm_generator.LLMGenerator instance.
        - retriever:        atlas_rag.retriever.BaseEdgeRetriever instance.
        - base_top_k:       TopK for text vector retrieval for the 1st step.
        - max_hops:         Maximum number of hops for subgraph expansion.
        """
        self.kg = kg
        self.node_list = node_list
        self.text_index = text_faiss_index
        self.text_node_dict = text_node_dict
        self.sentence_encoder = sentence_encoder
        self.llm_generator = llm_generator
        self.retriever = retriever
        
        self.base_top_k = base_top_k
        self.max_hops = max_hops
        self.increament_hop = increament_hop
        self.seed = seed
        self._set_seed(seed)
        self._dim = self.text_index.d
        # Ensure the order of text_node_dict keys matches the order when building text_index (pickle maintains insertion order by default).
        self._text_node_ids: List[str] = list(self.text_node_dict.keys())

        self.node_id_to_attr_id = {self.kg.nodes[n]['id']: n for n in self.kg.nodes}
        self.qa_judge = QAJudger()

    # ------------------------------------------------------------------
    # Main interface for external use
    # ------------------------------------------------------------------
    def refine(
        self, query: str, return_steps: bool = False
    ) -> Tuple[str, nx.DiGraph, Optional[List[ReafinerStepResult]]]:
        """
        Run the entire REAfiner process for a single query.

        Returns:
        -------
        - answer:       Answer given by LLM on the final refined KG (possibly abstract natural language).
        - refined_kg:    KG after inserting new knowledge (in-place modification, also returned by reference).
        - steps:        If return_steps=True, return a list of intermediate results for each step, for debugging/visualization.
        """
        step_results: List[ReafinerStepResult] = []
        final_answer: str = ""
        base_top_k = self.base_top_k

        for step in range(1, self.max_hops + 1):
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
                subgraph = self.construct_subgraph(node_id_list, num_hop=self.increament_hop)
                # convert subgraph to triple strings
                subgraph_edges = len(subgraph.edges)
                subgraph_triples = sorted([(self.kg.nodes[u]['id'], d['relation'], self.kg.nodes[v]['id']) for u, v, d in subgraph.edges(data=True)])
                sorted_context = [f"{s}  {r}  {o}" for s, r, o in subgraph_triples] 

            retrieved_context = "\n".join(sorted_context)

            # Single LLM call: Judge answerable + Error Abduction + Refined KG Generation
            answerable, reason, refined_triples, raw_response = self._refine_in_one_call(
                query, retrieved_context
            )

            if answerable:
                final_answer = self._generate_answer(query, retrieved_context)
                short_answer = self.qa_judge.split_answer(final_answer)
                step_results.append(
                    ReafinerStepResult(
                        step_id=step,
                        answerable=True,
                        raw_response=raw_response,
                        reason=reason,
                        retrieved_context=retrieved_context,
                        refined_triples=[],
                        answer=short_answer,
                    )
                )
                break

            # Insert new triples back into KG
            self._insert_triples_into_kg(refined_triples)
            step_results.append(
                ReafinerStepResult(
                    step_id=step,
                    answerable=False,
                    raw_response=raw_response,
                    reason=reason,
                    retrieved_context=retrieved_context,
                    refined_triples=refined_triples,
                    answer=None,
                )
            )
        if not final_answer:
            final_answer = self._generate_answer(query, retrieved_context)
            short_answer = self.qa_judge.split_answer(final_answer)

        return (short_answer, self.kg, step_results if return_steps else None)

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
    def _refine_in_one_call(
        self, query: str, triples_string: str
    ) -> Tuple[bool, str, str, List[Tuple[str, str, str]]]:
        """
        Single LLM call that performs all three steps in one session:
        1. Answerable Judgement (<judge>Yes/No</judge>)
        2. Error Abduction (<abduction>...</abduction>)
        3. Refined KG Generation (<refinement>[...]</refinement>)
        
        Returns:
        -------
        - answerable: bool
        - reason: str (from abduction if not answerable, empty if answerable)
        - answer: str (if answerable, otherwise empty)
        - refined_triples: List[Tuple[str, str, str]] (from refinement if not answerable, empty if answerable)
        """
        system_prompt = REFINE_SUBGRAPH_SYSTEM_PROMPT
        
        # Format user prompt (note: answer is not available during inference, so we omit it)
        if REFINE_SUBGRAPH_USER_PROMPT is not None:
            # Use the template from reafiner_prompt.py
            # Remove the "True Answer" line since it's not available during inference
            user_template = REFINE_SUBGRAPH_USER_PROMPT.strip()
            if "True Answer:" in user_template:
                # Remove the True Answer line
                lines = [line for line in user_template.split('\n') if 'True Answer:' not in line]
                user_template = '\n'.join(lines)
            user_content = user_template.format(
                question=query,
                triples_string=triples_string,
                answer=""  # Not used if we removed the line, but kept for compatibility
            )
        else:
            # Fallback format
            user_content = (
                f"Question: {query}\n"
                f"Knowledge Graph (KG) context: {triples_string}\n"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = self.llm_generator.generate_response(
            messages, temperature=0.0
        )
        # Parse the output: extract <judge>, <abduction>, <refinement> tags
        answerable = False
        reason = ""
        answer = ""
        refined_triples: List[Tuple[str, str, str]] = []
        
        # Extract <judge> tag
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
        
        if answerable:
            return answerable, None, [], raw
        else:
            # Extract <abduction> tag
            abduction_match = re.search(r'<abduction>(.*?)</abduction>', raw, re.IGNORECASE | re.DOTALL)
            if abduction_match:
                reason = abduction_match.group(1).strip()
            
            # Extract <refinement> tag
            refinement_match = re.search(r'<refinement>(.*?)</refinement>', raw, re.IGNORECASE | re.DOTALL)
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
                                refined_triples.append((subject, relation, obj))
                except json.JSONDecodeError:
                    # Fallback: try to parse as text format (head | relation | tail)
                    for line in refinement_json.splitlines():
                        line = line.strip()
                        if not line or "|" not in line:
                            continue
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) == 3:
                            h, r, t = parts
                            if h and r and t:
                                refined_triples.append((h, r, t))
        
        return answerable, reason, refined_triples, raw

    def _generate_answer(self, query: str, subgraph_str: str) -> str:
        """
        Generate the final answer on the final refined subgraph (no Yes/No judgment).
        """
        return self.llm_generator.generate_with_context_kg(query, subgraph_str, temperature=0.0)

    # ------------------------------------------------------------------
    # KG update logic
    # ------------------------------------------------------------------
    def _insert_triples_into_kg(
        self, triples: Iterable[Tuple[str, str, str]]
    ) -> None:
        """
        Insert the new triples generated by LLM into the KG.

        - If the entity node does not exist, create a new node, type is 'entity' by default.
        - If the edge already exists, only update the relation attribute.
        - Here we only update the KG structure itself, not dynamically update the vector index (simplified implementation).
        """
        for h, r, t in triples:
            head_id = self._ensure_node(h)
            tail_id = self._ensure_node(t)
            if self.kg.has_edge(head_id, tail_id):
                self.kg.edges[head_id, tail_id]["relation"] = r
            else:
                self.kg.add_edge(head_id, tail_id, relation=r)

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
    
    def construct_subgraph(self, initial_nodes, num_hop: int = 1):
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


__all__ = ["Reafiner", "ReafinerStepResult"]