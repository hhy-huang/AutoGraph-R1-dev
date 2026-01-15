from __future__ import annotations
import random
import re
import json
import networkx as nx
import numpy as np
import faiss
import json_repair
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set, Callable
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from autograph.rag_server.reafiner_prompt import REAFINER_JUDGEMENT_SYSTEM_PROMPT, REAFINER_JUDGEMENT_USER_PROMPT, \
    REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT, REAFINER_ERROR_ABDUCTION_USER_PROMPT, \
    REAFINER_KG_REFINEMENT_SYSTEM_PROMPT, REAFINER_KG_REFINEMENT_USER_PROMPT, \
    REFINE_SUBGRAPH_SYSTEM_PROMPT, REFINE_SUBGRAPH_USER_PROMPT, \
    REAFINER_FILTERING_SYSTEM_PROMPT, REAFINER_FILTERING_USER_PROMPT, \
    REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT, REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT
from atlas_rag.retriever.simple_retriever import SimpleGraphRetriever
from atlas_rag.evaluation.evaluation import QAJudger
from networkx import DiGraph
from tqdm import tqdm
try:
    import torch
except Exception:
    torch = None


_ILLEGAL_XML_RE = re.compile(
    "[" +
    "\x00-\x08" +
    "\x0B" +
    "\x0C" +
    "\x0E-\x1F" +
    "\uD800-\uDFFF" +   # Surrogates
    "\uFFFE\uFFFF" +    # Noncharacters
    "]"
)

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
    refinement_action_list: List[Callable[[], None]]
    refinement_action_raw: str


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
        base_top_k: int = 10,
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
        self.node_list = data["node_list"]  # no passage nodes
        self.edge_list = data["edge_list"]  # not include passage nodes
        self.node_faiss_index = data["node_faiss_index"]    # no passage nodes
        self.edge_faiss_index = data["edge_faiss_index"]    # not include passage nodes
        self.node_embeddings = data["node_embeddings"]
        self.edge_embeddings = data["edge_embeddings"]
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

        node_id_to_file_id = {}
        text_id_to_node_name = {}
        for node_id in list(self.kg.nodes):
            if self.kg.nodes[node_id]['type']=="passage":
                text_id_to_node_name[node_id] = self.kg.nodes[node_id]["id"]
            else:
                node_id_to_file_id[node_id] = self.kg.nodes[node_id]["file_id"]
        self.node_id_to_file_id = node_id_to_file_id        # node_id -> text id
        self.text_id_to_node_name = text_id_to_node_name    # text_id -> text string
        # filter out passage nodes - create a new mutable graph instead of a frozen view
        self.kg = DiGraph(self.kg.subgraph(self.node_list))
        self.node_id_to_attr_id = {self.kg.nodes[n]['id']: n for n in self.kg.nodes}
        self.qa_judge = QAJudger()
        self.entity_to_id = {}
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
                # subgraph_triples = sorted([(triple.split("  ")[0], triple.split("  ")[1], triple.split("  ")[2]) for triple in sorted_context])
            else:
                # expand the sub-graph with k-hop retrieval
                # # prune before expanding
                # if len(subgraph_triples) > self.max_triple_num:
                #     pruned_result = self._prune_subgraph_llm(subgraph_triples, query)
                #     pruned_subgraph_triples, prune_raw = pruned_result
                #     if prune_raw is not None:
                #         sorted_context = pruned_subgraph_triples
                #     else:
                #         print(pruned_subgraph_triples)
                #         pass
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
                subgraph_triples = sorted([(self.kg.nodes[u]['id'], d['relation'], self.kg.nodes[v]['id']) for u, v, d in subgraph.edges(data=True)])
                sorted_context = [f"{s}  {r}  {o}" for s, r, o in subgraph_triples] 
                if len(subgraph_triples) > self.max_triple_num:
                    sorted_context = self._prune_subgraph_embd(subgraph_triples, query)
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
                    refinement_action_list=[],
                    refinement_action_raw=None,
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
                refinement_action_list=[],
                refinement_action_raw=None,
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
                    refinement_action_list=[],
                    refinement_action_raw=None,
                )
                return (interaction_history[-1].answer, self.data, refinement_result)
            # Refined KG Generation
            # refined_subgraph, refined_subgraph_raw = self._kg_refinement(interaction_history[-1].retrieved_subgraph, error_abduction_reason)
            refinement_action_list, refinement_action_raw = self._kg_refinement_action(query, interaction_history[-1].retrieved_subgraph, error_abduction_reason)
            if refinement_action_raw is None:
                # fallback
                refinement_result = RefinementResult(
                    query=query,
                    history_horizon_size=self.history_horizon_size,
                    interaction_history=interaction_history,
                    error_abduction_reason=error_abduction_reason,
                    original_subgraph=interaction_history[-1].retrieved_subgraph,
                    refined_subgraph=None,
                    refinement_action_list=refinement_action_list,
                    refinement_action_raw=refinement_action_raw,
                )
                return (interaction_history[-1].answer, self.data, refinement_result)
            # take actions
            for action in refinement_action_list:
                action()
            # summarize the refinement result
            refinement_result = RefinementResult(
                query=query,
                history_horizon_size=self.history_horizon_size,
                interaction_history=interaction_history,
                error_abduction_reason=error_abduction_reason,
                original_subgraph=interaction_history[-1].retrieved_subgraph,
                refined_subgraph=None,
                refinement_action_list=refinement_action_list,
                refinement_action_raw=refinement_action_raw,
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
    
    def _prune_subgraph_llm(self, subgraph_triples: List[str], query: str) -> List[str]:
        """
        Prune the subgraph based on the LLM judgement.
        """
        system_prompt = REAFINER_FILTERING_SYSTEM_PROMPT
        user_prompt = REAFINER_FILTERING_USER_PROMPT.format(
            triples_string=subgraph_triples,
            query=query,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw = self.llm_generator.generate_response(
                messages, temperature=0.0, max_new_tokens=8192
            )
        except Exception as e:
            # fallback
            error_message = {"error": f"Prune Subgraph LLM Error: {e}"}
            print(error_message)
            return error_message['error'], None
        # Parse the output: extract <filtering> tag
        filtering_match = re.search(r'<filtering>(.*?)</filtering>', raw, re.IGNORECASE | re.DOTALL)
        if filtering_match:
            filtering_text = filtering_match.group(1).strip()
            try:
                filtering_triples = json.loads(filtering_text)
            except json.JSONDecodeError:
                try:
                    refined_triples = json_repair.loads(filtering_text)
                except Exception:
                    # fallback
                    error_message = {"error": f"Filtering Subgraph LLM Error Format: {raw}"}
                    print(error_message)
                    return error_message['error'], None
        else:
            if '<filtering>' in raw:
                filtering_text = raw.split('<filtering>')[1].split('</filtering>')[0].strip()
            else:
                # fallback
                error_message = {"error": f"Filtering Subgraph LLM Error Format: {raw}"}
                print(error_message)
                return error_message['error'], None
            try:
                filtering_triples = json.loads(filtering_text)
            except json.JSONDecodeError:
                # if fails, try json_repair to fix common JSON issues (like invalid escape sequences)
                try:
                    filtering_triples = json_repair.loads(filtering_text)
                except Exception:
                    # fallback
                    error_message = {"error": f"Filtering Subgraph LLM Error Format: {raw}"}
                    print(error_message)
                    return error_message['error'], None
        pruned_subgraph_triples = [f"{triple['subject']}  {triple['relation']}  {triple['object']}" for triple in filtering_triples]
        return pruned_subgraph_triples, raw
    
    def _prune_subgraph_embd(self, subgraph_triples: List[str], query: str) -> List[str]:
        """
        Prune the subgraph embedding similarity.
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

    def _parse_action_string(self, action: str) -> Tuple[str, List[str]]:
        """
        Parse an action string like 'insert_edge("subject", "relation", "object")'
        Returns (function_name, [arg1, arg2, ...])
        Handles entity names containing commas, parentheses, and quotes.
        """
        action = action.strip()
        # Match function name and arguments
        # Pattern: function_name("arg1", "arg2", ...) or function_name('arg1', 'arg2', ...)
        pattern = r'(\w+)\s*\((.*)\)\s*$'
        match = re.match(pattern, action)
        if not match:
            raise ValueError(f"Invalid action format: {action}")
        function_name = match.group(1)
        args_str = match.group(2).strip()
        # Parse arguments by finding quoted strings (handles escaped quotes)
        parsed_args = []
        i = 0
        while i < len(args_str):
            # Skip whitespace and commas
            while i < len(args_str) and args_str[i] in ' \t,':
                i += 1
            if i >= len(args_str):
                break
            # Determine quote type (single or double)
            quote_char = args_str[i]
            if quote_char not in ['"', "'"]:
                raise ValueError(f"Expected quoted string at position {i} in: {action}")
            i += 1  # Skip opening quote
            arg_value = []
            # Parse until matching closing quote (handling escaped quotes)
            while i < len(args_str):
                if args_str[i] == '\\' and i + 1 < len(args_str):
                    # Escaped character
                    arg_value.append(args_str[i + 1])
                    i += 2
                elif args_str[i] == quote_char:
                    # Found closing quote
                    parsed_args.append(''.join(arg_value))
                    i += 1
                    break
                else:
                    arg_value.append(args_str[i])
                    i += 1
            else:
                # No closing quote found
                raise ValueError(f"Unclosed quote in: {action}")
        if not parsed_args:
            raise ValueError(f"No valid arguments found in: {action}")
        return function_name, parsed_args

    def _kg_refinement_action(self, query: str, triples_string: str, error_abduction_reason: str) -> Tuple[List[str], str]:
        """
        Generate a series of actions to refine the knowledge graph based on the given error reasons.
        """
        system_prompt = REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT
        user_prompt = REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT.format(
            question=query,
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
            error_message = [{"error": f"KG Refinement Action Generation Error: {e}"}]
            print(error_message)
            return error_message[0]['error'], None
        # Parse the output: extract <refinement> tag
        refinement_match = re.search(r'<refinement>(.*?)</refinement>', raw, re.IGNORECASE | re.DOTALL)
        if refinement_match:
            refinement_actions_str = refinement_match.group(1).strip().strip("|")
            # parse action list and store function objects
            refinement_action_list = []
            for action in refinement_actions_str.split("|"):
                action = action.strip()
                if not action:
                    continue
                try:
                    function_name, args = self._parse_action_string(action)
                    
                    if function_name == "insert_edge":
                        if len(args) != 3:
                            raise ValueError(f"insert_edge requires 3 arguments, got {len(args)}")
                        sub, rel, obj = args[0], args[1], args[2]
                        refinement_action_list.append(lambda s=sub, r=rel, o=obj: self._insert_edge(s, r, o))
                    elif function_name == "delete_edge":
                        if len(args) != 3:
                            raise ValueError(f"delete_edge requires 3 arguments, got {len(args)}")
                        sub, rel, obj = args[0], args[1], args[2]
                        refinement_action_list.append(lambda s=sub, o=obj: self._delete_edge(s, o))
                    elif function_name == "replace_node":
                        if len(args) != 2:
                            raise ValueError(f"replace_node requires 2 arguments, got {len(args)}")
                        old_ent, new_ent = args[0], args[1]
                        refinement_action_list.append(lambda old=old_ent, new=new_ent: self._replace_node(old, new))
                    else:
                        print(f"Error: Unknown action format: {function_name}")
                        continue
                except Exception as e:
                    error_message = [{"error": f"KG Refinement Action Error Format: {action}, Error: {str(e)}"}]
                    print(error_message)
                    return error_message[0]['error'], None
        else:
            error_message = [{"error": f"KG Refinement Error Format: {raw}"}]
            print(error_message)
            return error_message[0]['error'], None
        return refinement_action_list, raw
    
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

    # ------------------------------------------------------------------
    # KG update logic
    # ------------------------------------------------------------------
    def _insert_edge(self, sub: str, rel: str, obj: str) -> None:
        """
        Insert an edge into the KG.
        """
        subject_mapped_id = self._get_node_id(sub, self.entity_to_id)
        object_mapped_id = self._get_node_id(obj, self.entity_to_id)
        new_node_list = []
        # TODO: Add file_id to the node
        if subject_mapped_id not in self.kg.nodes:
            if subject_mapped_id not in self.node_id_to_file_id:
                self.node_id_to_file_id[subject_mapped_id] = None
            self.kg.add_node(
                subject_mapped_id,
                id=self._safe_sanitize(sub),
                type="entity",
                file_id=self.node_id_to_file_id[subject_mapped_id]
            )
            new_node_list.append(subject_mapped_id)
        if object_mapped_id not in self.kg.nodes:
            if object_mapped_id not in self.node_id_to_file_id:
                self.node_id_to_file_id[object_mapped_id] = None
            self.kg.add_node(
                object_mapped_id,
                id=self._safe_sanitize(obj),
                type="entity",
                file_id=self.node_id_to_file_id[object_mapped_id]
            )
            new_node_list.append(object_mapped_id)
        if not self.kg.has_edge(subject_mapped_id, object_mapped_id):
            self.kg.add_edge(
                subject_mapped_id,
                object_mapped_id,
                relation=self._safe_sanitize(rel),
                type="Relation"
            )
        # update the edge_list and edge_embeddings
        if (subject_mapped_id, object_mapped_id) not in self.edge_list:
            new_edge_embeddings = self.sentence_encoder.encode(f"{sub} {rel} {obj}", query_type="edge")
            if isinstance(new_edge_embeddings, torch.Tensor):
                new_edge_embeddings = new_edge_embeddings.cpu().numpy()
            new_edge_faiss_embeddings = np.array(new_edge_embeddings).astype('float32')
            # Reshape to 2D if needed (single vector should be shape [1, dim])
            if new_edge_faiss_embeddings.ndim == 1:
                new_edge_faiss_embeddings = new_edge_faiss_embeddings.reshape(1, -1)
            faiss.normalize_L2(new_edge_faiss_embeddings)
            next_faiss_id = max(self.edge_faiss_id_to_list_idx.keys()) + 1
            self.edge_list.append((subject_mapped_id, object_mapped_id))
            self.edge_embeddings.append(new_edge_embeddings)
            self.edge_faiss_index.add_with_ids(new_edge_faiss_embeddings, np.array([next_faiss_id], dtype=np.int64))
            self.edge_faiss_id_to_list_idx[next_faiss_id] = len(self.edge_list) - 1
        # update the node_list and node_embeddings
        if new_node_list:
            new_node_embeddings = self.sentence_encoder.encode(new_node_list, query_type="node")
            if isinstance(new_node_embeddings, torch.Tensor):
                new_node_embeddings = new_node_embeddings.cpu().numpy()
            new_node_faiss_embeddings = np.array(new_node_embeddings).astype('float32')
            if new_node_faiss_embeddings.ndim == 1:
                new_node_faiss_embeddings = new_node_faiss_embeddings.reshape(1, -1)
            faiss.normalize_L2(new_node_faiss_embeddings)
            next_faiss_id = max(self.node_faiss_id_to_list_idx.keys()) + 1 if self.node_faiss_id_to_list_idx else 0
            start_list_idx = len(self.node_list)
            self.node_list.extend(new_node_list)
            self.node_embeddings.extend(new_node_embeddings)
            faiss_ids = np.array(list(range(next_faiss_id, next_faiss_id + len(new_node_list))), dtype=np.int64)
            self.node_faiss_index.add_with_ids(new_node_faiss_embeddings, faiss_ids)
            for i, faiss_id in enumerate(faiss_ids):
                self.node_faiss_id_to_list_idx[int(faiss_id)] = start_list_idx + i
        # update the data
        self.data["KG"] = self.kg
        self.data["edge_list"] = self.edge_list
        self.data["node_list"] = self.node_list
        self.data["edge_embeddings"] = self.edge_embeddings
        self.data["node_embeddings"] = self.node_embeddings
        self.data["node_faiss_index"] = self.node_faiss_index
        self.data["edge_faiss_index"] = self.edge_faiss_index
        self.data["edge_faiss_id_to_list_idx"] = self.edge_faiss_id_to_list_idx
        self.data["node_faiss_id_to_list_idx"] = self.node_faiss_id_to_list_idx
        # update data in retriever
        self.retriever.KG = self.kg
        self.retriever.edge_list = self.edge_list
        self.retriever.node_list = self.node_list
        self.retriever.edge_faiss_index = self.edge_faiss_index
        self.retriever.node_faiss_index = self.node_faiss_index
        self.retriever.edge_faiss_id_to_list_idx = self.edge_faiss_id_to_list_idx
        self.retriever.node_faiss_id_to_list_idx = self.node_faiss_id_to_list_idx

    def _delete_edge(self, sub: str, obj: str) -> None:
        """
        Delete an edge from the KG.
        """
        subject_mapped_id = self._get_node_id(sub, self.entity_to_id)
        object_mapped_id = self._get_node_id(obj, self.entity_to_id)
        edge_tuple = (subject_mapped_id, object_mapped_id)
        if not self.kg.has_edge(subject_mapped_id, object_mapped_id):
            print("Action Error: Edge not found in KG: ", sub, obj)
            return
        # Find the index of the edge in edge_list
        if edge_tuple not in self.edge_list:
            print("Action Error: Edge not found in edge_list: ", sub, obj)
            return
        list_idx = self.edge_list.index(edge_tuple)
        # Find the corresponding faiss_id by reverse lookup in mapping table
        faiss_id_to_remove = None
        for faiss_id, mapped_list_idx in self.edge_faiss_id_to_list_idx.items():
            if mapped_list_idx == list_idx:
                faiss_id_to_remove = faiss_id
                break
        if faiss_id_to_remove is None:
            print("Action Error: FAISS ID not found for edge: ", sub, obj)
            return
        
        # Remove edge from KG
        self.kg.remove_edge(subject_mapped_id, object_mapped_id)
        # Remove from FAISS index
        self.edge_faiss_index.remove_ids(np.array([faiss_id_to_remove], dtype=np.int64))
        # Remove from mapping table
        del self.edge_faiss_id_to_list_idx[faiss_id_to_remove]
        # Update remaining mappings: decrease list_idx for items after deleted one
        deleted_set = {list_idx}
        for faiss_id in self.edge_faiss_id_to_list_idx:
            current_idx = self.edge_faiss_id_to_list_idx[faiss_id]
            count_deleted_before = sum(1 for idx in deleted_set if idx < current_idx)
            self.edge_faiss_id_to_list_idx[faiss_id] -= count_deleted_before
        # Delete from lists (from back to front to avoid index shifting)
        del self.edge_list[list_idx]
        del self.edge_embeddings[list_idx]
        # update the data
        self.data["KG"] = self.kg
        self.data["edge_faiss_index"] = self.edge_faiss_index
        self.data["edge_list"] = self.edge_list
        self.data["edge_embeddings"] = self.edge_embeddings
        self.data["edge_faiss_id_to_list_idx"] = self.edge_faiss_id_to_list_idx
        # update retriever
        self.retriever.KG = self.kg
        self.retriever.edge_list = self.edge_list
        self.retriever.edge_faiss_index = self.edge_faiss_index
        self.retriever.edge_faiss_id_to_list_idx = self.edge_faiss_id_to_list_idx

    def _replace_node(self, old_entity: str, new_entity: str) -> None:
        """
        Replace a node in the KG.
        """
        old_mapped_id = self._get_node_id(old_entity, self.entity_to_id)
        new_mapped_id = self._get_node_id(new_entity, self.entity_to_id)
        if not self.kg.has_node(old_mapped_id):
            print("Action Error: Node not found in KG: ", old_entity)
            return
        # obtain all edges connected to the old node, preserve for the new node
        edges_to_preserve = []
        edges_to_delete = []
        for neighbor in sorted(self.kg.successors(old_mapped_id)):
            neighbor_id = self.kg.nodes[neighbor].get("id", None)
            if neighbor_id:
                relation = self.kg.edges[old_mapped_id, neighbor]["relation"]
                edges_to_preserve.append((new_entity, relation, neighbor_id))
                edges_to_delete.append((old_entity, neighbor_id))
        for neighbor in sorted(self.kg.predecessors(old_mapped_id)):
            neighbor_id = self.kg.nodes[neighbor].get("id", None)
            if neighbor_id:
                relation = self.kg.edges[neighbor, old_mapped_id]["relation"]
                edges_to_preserve.append((neighbor_id, relation, new_entity))
                edges_to_delete.append((neighbor_id, old_entity))
        # delete the original edges
        for edge in edges_to_delete:
            self._delete_edge(edge[0], edge[1])
        # add the new edges and update the data
        for edge in edges_to_preserve:
            self._insert_edge(edge[0], edge[1], edge[2])
        # delete the original node
        self.kg.remove_node(old_mapped_id)
        # update the node data
        node_idx = self.node_list.index(old_mapped_id)
        faiss_id_to_remove = None
        for faiss_id, mapped_list_idx in self.node_faiss_id_to_list_idx.items():
            if mapped_list_idx == node_idx:
                faiss_id_to_remove = faiss_id
                break
        if faiss_id_to_remove is None:
            print("Action Error: FAISS ID not found for node: ", old_entity)
            return
        # Remove from FAISS index
        self.node_faiss_index.remove_ids(np.array([faiss_id_to_remove], dtype=np.int64))
        # Remove from mapping table
        del self.node_faiss_id_to_list_idx[faiss_id_to_remove]
        # Update remaining mappings: decrease list_idx for items after deleted one
        deleted_set = {node_idx}
        for faiss_id in self.node_faiss_id_to_list_idx:
            current_idx = self.node_faiss_id_to_list_idx[faiss_id]
            count_deleted_before = sum(1 for idx in deleted_set if idx < current_idx)
            self.node_faiss_id_to_list_idx[faiss_id] -= count_deleted_before
        # Delete from lists (from back to front to avoid index shifting)
        del self.node_list[node_idx]
        del self.node_embeddings[node_idx]
        # update the data
        self.data["KG"] = self.kg
        self.data["node_faiss_index"] = self.node_faiss_index
        self.data["node_list"] = self.node_list
        self.data["node_embeddings"] = self.node_embeddings
        self.data["node_faiss_id_to_list_idx"] = self.node_faiss_id_to_list_idx
        # update retriever
        self.retriever.KG = self.kg
        self.retriever.node_list = self.node_list
        self.retriever.node_faiss_index = self.node_faiss_index
        self.retriever.node_faiss_id_to_list_idx = self.node_faiss_id_to_list_idx

    def _generate_answer(self, query: str, subgraph_str: str) -> str:
        """
        Generate the final answer on the final refined subgraph (no Yes/No judgment).
        """
        return self.llm_generator.generate_with_context_kg(query, subgraph_str, temperature=0.0)

    def _get_node_id(self, entity_name, entity_to_id={}):
        """Returns existing or creates new nX ID for an entity using a hash-based approach."""
        if entity_name not in entity_to_id:
            # Use a hash function to generate a unique ID
            entity_name = entity_name+'_entity'
            hash_object = hashlib.sha256(entity_name.encode('utf-8'))
            hash_hex = hash_object.hexdigest()  # Get the hexadecimal representation of the hash
            # Use the first 8 characters of the hash as the ID (you can adjust the length as needed)
            entity_to_id[entity_name] = hash_hex
        return entity_to_id[entity_name]
    
    def _safe_sanitize(self, value):
        """Safely sanitize any value for XML output."""
        def _sanitize_xml_string(s: str) -> str:
            """Remove illegal XML characters from a string."""
            return _ILLEGAL_XML_RE.sub("", s)
        if value is None:
            return ""
        return _sanitize_xml_string(str(value))
    
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