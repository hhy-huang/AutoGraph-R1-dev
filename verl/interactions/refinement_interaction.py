import copy
import hashlib
import logging
import os
import re
import json_repair
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from networkx import DiGraph

from .base import BaseInteraction

try:
    from autograph.rag_server.edge_retriever import EdgeRetriever
except ImportError:
    from autorefiner.src.rag_server.edge_retriever import EdgeRetriever

try:
    from autograph.rag_server.reafiner_prompt import (
        REAFINER_JUDGEMENT_SYSTEM_PROMPT,
        REAFINER_JUDGEMENT_USER_PROMPT,
        REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT,
        REAFINER_ERROR_ABDUCTION_USER_PROMPT,
        REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT,
        REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT,
    )
except ImportError:
    from autorefiner.src.reafiner_prompt import (
        REAFINER_JUDGEMENT_SYSTEM_PROMPT,
        REAFINER_JUDGEMENT_USER_PROMPT,
        REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT,
        REAFINER_ERROR_ABDUCTION_USER_PROMPT,
        REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT,
        REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT,
    )

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]")


class RefinementInteraction(BaseInteraction):
    """Refinement pipeline as three ordered states: ANSWERABLE_JUDGEMENT -> ABDUCTION -> ACTION_GENERATION -> RAG.

    Each state corresponds to one LLM turn via _handle_engine_call; parsing and next-prompt
    construction happen in INTERACTING via generate_response_refinement().
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: Dict[str, dict] = {}
        self.base_top_k = config.get("base_top_k", 10)
        self.max_hops = config.get("max_hops", 5)
        self.increment_hop = config.get("increment_hop", 1)
        self.max_triple_num = config.get("max_triple_num", 60)
        self.history_horizon_size = config.get("history_horizon_size", 3)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        question: Optional[str] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        full_graph_data = kwargs.get("full_graph_data")
        sentence_encoder = kwargs.get("sentence_encoder")
        if not full_graph_data or not sentence_encoder:
            raise ValueError("full_graph_data and sentence_encoder required in interaction_kwargs for refinement")

        kg_orig = full_graph_data["KG"]
        node_list = list(full_graph_data.get("node_list", kg_orig.nodes()))
        edge_list = full_graph_data.get("edge_list", list(kg_orig.edges()))
        edge_faiss_index = full_graph_data["edge_faiss_index"]
        edge_faiss_id_to_list_idx = full_graph_data.get("edge_faiss_id_to_list_idx", {i: i for i in range(len(edge_list))})

        # One KG copy per request so we can mutate it
        kg = DiGraph(copy.deepcopy(kg_orig.subgraph(node_list)))
        node_id_to_attr_id = {kg.nodes[n]["id"]: n for n in kg.nodes}

        # Extract triples from pre-populated messages instead of re-retrieving
        # The data preparation stage already retrieved the first-hop subgraph and built the prompt
        initial_messages = kwargs.get("initial_messages", [])
        sorted_context = []
        prompt_judgement_system = REAFINER_JUDGEMENT_SYSTEM_PROMPT
        prompt_judgement_user = None
        
        if initial_messages:
            # Extract system and user prompts from pre-populated messages
            for msg in initial_messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                else:
                    # Message object with .role and .content attributes
                    role = getattr(msg, "role", "")
                    content = getattr(msg, "content", "")
                
                if role == "system":
                    prompt_judgement_system = content
                elif role == "user":
                    prompt_judgement_user = content
                    # Parse triples from user message (format: "Question: ...\nKnowledge Graph (KG) context: ...")
                    if "Knowledge Graph (KG) context:" in content:
                        kg_section = content.split("Knowledge Graph (KG) context:")[-1].strip()
                        # Parse triples: each line is "subject  relation  object"
                        for line in kg_section.split("\n"):
                            line = line.strip()
                            if line and "  " in line:
                                parts = line.split("  ", 2)
                                if len(parts) == 3:
                                    sorted_context.append(line)
        
        # Fallback: if no initial_messages provided, do retrieval (for backward compatibility)
        if not sorted_context and question:
            import numpy as np
            query_embedding = sentence_encoder.encode([question], query_type="edge")
            q = np.asarray(query_embedding, dtype="float32")
            if q.ndim == 1:
                q = q.reshape(1, -1)
            D, I = edge_faiss_index.search(q, self.base_top_k)
            list_indices = [edge_faiss_id_to_list_idx.get(int(fid), int(fid)) for fid in I[0]]
            topk_edges = [edge_list[i] for i in list_indices if i < len(edge_list)]
            for e in topk_edges:
                if e in kg_orig.edges and e[0] in kg.nodes and e[1] in kg.nodes:
                    s, r = kg_orig.nodes[e[0]].get("id", str(e[0])), kg_orig.edges[e].get("relation", "")
                    o = kg_orig.nodes[e[1]].get("id", str(e[1]))
                    sorted_context.append(f"{s}  {r}  {o}")
            if not prompt_judgement_user:
                triples_string = "\n".join(sorted_context) if sorted_context else "(no triples)"
                prompt_judgement_user = REAFINER_JUDGEMENT_USER_PROMPT.format(question=question, triples_string=triples_string)
        
        # Ensure prompt_judgement_user is set (fallback if neither initial_messages nor retrieval worked)
        if not prompt_judgement_user:
            triples_string = "\n".join(sorted_context) if sorted_context else "(no triples)"
            prompt_judgement_user = REAFINER_JUDGEMENT_USER_PROMPT.format(question=question or "", triples_string=triples_string)

        self._instance_dict[instance_id] = {
            "rag_state": False,
            "ground_truth": ground_truth,
            "question": question,
            "kg": kg,
            "node_id_to_attr_id": node_id_to_attr_id,
            "entity_to_id": {},
            "prompt_judgement_system": prompt_judgement_system,
            "prompt_judgement_user": prompt_judgement_user,
            "refinement_initial_injected": False,
            "refinement_phase": "answerable_judgement",
            "interaction_history": [],
            "sorted_context": sorted_context,
            "error_abduction_reason": None,
        }
        return instance_id

    async def generate_response_refinement(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        current_phase: str,
        **kwargs,
    ) -> Tuple[bool, str, float, dict]:
        """Parse last assistant message and return (should_terminate, next_user_message, reward, extra).
        extra may contain next_system, next_rag_state (enum value string or enum)."""
        inst = self._instance_dict[instance_id]
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "") or ""
                break

        reward = 0.0
        extra: Dict[str, Any] = {}

        if current_phase == "answerable_judgement":
            # Parse current judgement result
            judge_match = re.search(r"<judge>(.*?)</judge>", content, re.IGNORECASE | re.DOTALL)
            if not judge_match:
                # Strictly require <judge>...</judge>, otherwise treat as parse failure.
                print(
                    f"\033[91m [instance {instance_id}] "
                    f"[Failed to parse judgement content: missing <judge> tag]\nContent: {content} \033[0m"
                )
                return True, "Failed to parse judgement content.", 0.0, {}

            answerable_str = judge_match.group(1).strip().lower()
            answerable = answerable_str.startswith("yes")

            # Current judgement step index (1-based)
            prev_steps = sum(1 for h in inst["interaction_history"] if h.get("phase") == "judgement")
            judgement_steps = prev_steps + 1

            # Record structured interaction history, similar to reafiner.Reafiner
            inst["interaction_history"].append(
                {
                    "phase": "judgement",
                    "raw_response": content,
                    "query": inst.get("question", ""),
                    "subgraph_hop": judgement_steps,
                    "subgraph_content": inst.get("sorted_context", []),
                    "answerable": answerable,
                }
            )
            print(
                f"\033[94m [instance {instance_id}] "
                f"[Judgement Steps: {judgement_steps}, Answerable: {answerable}] \033[0m"
            )

            if answerable:
                # inst["rag_state"] = True
                if judgement_steps == 1:
                    return True, "No need to do any refinement.", 1.0, {}
                else:
                    inst["refinement_phase"] = "abduction"
                    # Build interaction history string in the same style as reafiner.py
                    history = inst["interaction_history"]
                    horizon = getattr(self, "history_horizon_size", 0) or 0
                    if horizon > 0 and len(history) > horizon:
                        used_history = history[:-horizon]
                    else:
                        used_history = history
                    hist_str = "\n".join(
                        [
                            "Step{}:\n['Query': {}, 'Subgraph_hop': {}, 'Subgraph_content': {}, 'Answerable': {}]\n".format(
                                i + 1,
                                h.get("query", ""),
                                h.get("subgraph_hop", ""),
                                str(h.get("subgraph_content", "")),
                                h.get("answerable", ""),
                            )
                            for i, h in enumerate(used_history)
                        ]
                    )
                    next_system = REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT
                    next_user = REAFINER_ERROR_ABDUCTION_USER_PROMPT.format(interaction_history=hist_str)
                    extra["next_system"] = next_system
                    extra["next_rag_state"] = "abduction"
                    return False, next_user, 1.0, extra

            # Not answerable: keep doing answerable_judgement up to max_hops.
            # 即使这一轮没法扩张子图，也继续用当前子图再判一轮，而不是立刻进 abduction。
            if judgement_steps < self.max_hops:
                full_graph_data = kwargs.get("full_graph_data")
                if full_graph_data is not None:
                    kg_orig = full_graph_data["KG"]
                    node_list = list(kg_orig.nodes())
                    sorted_ctx = inst.get("sorted_context") or []
                    node_str_list: List[str] = []
                    for triple_str in sorted_ctx:
                        parts = triple_str.split("  ", 2)
                        if len(parts) == 3:
                            node_str_list.append(parts[0].strip())
                            node_str_list.append(parts[2].strip())
                    node_str_list = list(set(node_str_list))
                    id_to_node = {kg_orig.nodes[n].get("id", n): n for n in kg_orig.nodes()}
                    initial_nodes = [id_to_node[ns] for ns in node_str_list if ns in id_to_node]
                    if initial_nodes:
                        subgraph = self._construct_subgraph(
                            kg_orig, node_list, initial_nodes, num_hop=self.increment_hop
                        )
                        subgraph_triples = sorted(
                            [
                                (kg_orig.nodes[u].get("id", u), d.get("relation", ""), kg_orig.nodes[v].get("id", v))
                                for u, v, d in subgraph.edges(data=True)
                            ]
                        )
                        sorted_context = [f"{s}  {r}  {o}" for s, r, o in subgraph_triples]
                        if len(subgraph_triples) > self.max_triple_num:
                            sorted_context = sorted_context[: self.max_triple_num]
                        inst["kg"] = copy.deepcopy(subgraph)
                        inst["node_id_to_attr_id"] = {
                            inst["kg"].nodes[n].get("id", n): n for n in inst["kg"].nodes
                        }
                        inst["sorted_context"] = sorted_context
                        triples_string = "\n".join(sorted_context) if sorted_context else "(no triples)"
                        inst["prompt_judgement_user"] = REAFINER_JUDGEMENT_USER_PROMPT.format(
                            question=inst["question"], triples_string=triples_string
                        )
                # 无论本轮是否成功扩张子图，只要还没到 max_hops，就继续下一轮 judgement
                extra["next_system"] = inst["prompt_judgement_system"]
                extra["next_rag_state"] = "answerable_judgement"
                return False, inst["prompt_judgement_user"], 1.0, extra

            # 达到 max_hops 才进入 abduction
            inst["refinement_phase"] = "abduction"
            history = inst["interaction_history"]
            horizon = getattr(self, "history_horizon_size", 0) or 0
            if horizon > 0 and len(history) > horizon:
                used_history = history[:-horizon]
            else:
                used_history = history
            hist_str = "\n".join(
                [
                    "Step{}:\n['Query': {}, 'Subgraph_hop': {}, 'Subgraph_content': {}, 'Answerable': {}]\n".format(
                        i + 1,
                        h.get("query", ""),
                        h.get("subgraph_hop", ""),
                        str(h.get("subgraph_content", "")),
                        h.get("answerable", ""),
                    )
                    for i, h in enumerate(used_history)
                ]
            )
            next_system = REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT
            next_user = REAFINER_ERROR_ABDUCTION_USER_PROMPT.format(interaction_history=hist_str)
            extra["next_system"] = next_system
            extra["next_rag_state"] = "abduction"
            return False, next_user, 1.0, extra

        if current_phase == "abduction":
            print(f"\033[94m [instance {instance_id}] [Abduction] \033[0m")
            print(f"Raw content:\n {content}")
            inst["interaction_history"].append({"phase": "abduction", "raw_response": content})
            # Abduction phase must output <abduction>...</abduction>, otherwise treat as parse failure.
            abduction_match = re.search(r"<abduction>(.*?)</abduction>", content, re.IGNORECASE | re.DOTALL)
            if not abduction_match:
                print(
                    f"\033[91m [instance {instance_id}] "
                    f"[Failed to parse abduction content: missing <abduction> tag]\nContent: {content} \033[0m"
                )
                inst["refinement_phase"] = "action_generation"
                return True, "Failed to parse abduction content.", 0.0, {}

            error_reason = abduction_match.group(1).strip()
            inst["error_abduction_reason"] = error_reason
            inst["refinement_phase"] = "action_generation"
            triples_string = "\n".join(inst["sorted_context"]) if inst.get("sorted_context") else ""
            next_system = REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT
            next_user = REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT.format(
                original_text=str(inst["sorted_context"]) if inst.get("sorted_context") else "",
                triples_string=triples_string,
                question=inst["question"],
                error_reasons=error_reason,
            )
            extra["next_system"] = next_system
            extra["next_rag_state"] = "action_generation"
            return False, next_user, 1.0, extra

        if current_phase == "action_generation":
            print(f"\033[94m [instance {instance_id}] [Action Generation] \033[0m")
            print(f"Raw content:\n {content}")
            inst["interaction_history"].append({"phase": "action", "raw_response": content})
            # Action generation phase must output <refinement>...</refinement>, otherwise treat as parse failure.
            refinement_match = re.search(r"<refinement>(.*?)</refinement>", content, re.IGNORECASE | re.DOTALL)
            if not refinement_match:
                print(
                    f"\033[91m [instance {instance_id}] "
                    f"[Failed to parse action generation content: missing <refinement> tag]\nContent: {content} \033[0m"
                )
                inst["refinement_phase"] = "rag"
                return True, "Failed to parse action generation content.", 0.0, {}

            try:
                self._apply_refinement_actions(instance_id, refinement_match.group(1).strip())
            except Exception as e:
                print(
                    f"\033[91m [instance {instance_id}] "
                    f"[Failed to apply refinement actions: {e}\nContent: {content}] \033[0m"
                )
                inst["refinement_phase"] = "rag"
                return True, "Failed to apply refinement actions.", 0.0, {}

            inst["rag_state"] = True
            extra["next_rag_state"] = "rag"
            return False, "You will perform graph based RAG based on your constructed knowledge graph.", 1.0, extra

        return True, "Unknown refinement phase.", 0.0, {}

    @staticmethod
    def _construct_subgraph(
        kg: DiGraph, node_list: List, initial_nodes: List, num_hop: int = 1
    ) -> DiGraph:
        """Construct a multi-hop subgraph around initial nodes (BFS). Same logic as reafiner.Reafiner._construct_subgraph."""
        subgraph = DiGraph()
        visited = set()
        node_set = set(node_list)
        queue = [(n, 0) for n in initial_nodes if n in node_set]
        for node, _ in queue:
            subgraph.add_node(node, **dict(kg.nodes[node]))
            visited.add(node)
        while queue:
            current_node, hop_count = queue.pop(0)
            if hop_count >= num_hop:
                continue
            for neighbor in sorted(kg.successors(current_node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor, **dict(kg.nodes[neighbor]))
                    queue.append((neighbor, hop_count + 1))
                rel = kg.edges[(current_node, neighbor)].get("relation", "")
                subgraph.add_edge(current_node, neighbor, relation=rel)
            for neighbor in sorted(kg.predecessors(current_node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor, **dict(kg.nodes[neighbor]))
                    queue.append((neighbor, hop_count + 1))
                rel = kg.edges[(neighbor, current_node)].get("relation", "")
                subgraph.add_edge(neighbor, current_node, relation=rel)
        return subgraph

    @staticmethod
    def _safe_sanitize(value: Any) -> str:
        if value is None:
            return ""
        return _ILLEGAL_XML_RE.sub("", str(value))

    @staticmethod
    def _get_node_id(entity_name: str, entity_to_id: dict) -> str:
        if entity_name not in entity_to_id:
            entity_to_id[entity_name] = hashlib.sha256((entity_name + "_entity").encode()).hexdigest()
        return entity_to_id[entity_name]

    def _insert_edge(self, instance_id: str, sub: str, rel: str, obj: str) -> None:
        inst = self._instance_dict[instance_id]
        kg = inst["kg"]
        eid = inst["entity_to_id"]
        sid, oid = self._get_node_id(sub, eid), self._get_node_id(obj, eid)
        if sid not in kg.nodes:
            kg.add_node(sid, id=self._safe_sanitize(sub), type="entity")
        if oid not in kg.nodes:
            kg.add_node(oid, id=self._safe_sanitize(obj), type="entity")
        if not kg.has_edge(sid, oid):
            kg.add_edge(sid, oid, relation=self._safe_sanitize(rel))

    def _delete_edge(self, instance_id: str, sub: str, obj: str) -> None:
        inst = self._instance_dict[instance_id]
        kg, eid = inst["kg"], inst["entity_to_id"]
        sid, oid = self._get_node_id(sub, eid), self._get_node_id(obj, eid)
        if kg.has_edge(sid, oid):
            kg.remove_edge(sid, oid)

    def _replace_node(self, instance_id: str, old_entity: str, new_entity: str) -> None:
        inst = self._instance_dict[instance_id]
        kg, eid = inst["kg"], inst["entity_to_id"]
        old_id = self._get_node_id(old_entity, eid)
        if old_id not in kg.nodes:
            return
        edges_add = []
        for _u, v, d in list(kg.edges(old_id, data=True)):
            edges_add.append((new_entity, d.get("relation", ""), kg.nodes[v].get("id", str(v))))
        for u, _v, d in list(kg.in_edges(old_id, data=True)):
            edges_add.append((kg.nodes[u].get("id", str(u)), d.get("relation", ""), new_entity))
        kg.remove_node(old_id)
        for s, r, o in edges_add:
            self._insert_edge(instance_id, s, r, o)

    def _apply_refinement_actions(self, instance_id: str, raw_actions: str) -> None:
        for action in (a.strip() for a in raw_actions.strip().strip("|").split("|") if a.strip()):
            try:
                fn_name, args = self._parse_action_string(action)
                if fn_name == "insert_edge" and len(args) == 3:
                    self._insert_edge(instance_id, args[0], args[1], args[2])
                elif fn_name == "delete_edge" and len(args) >= 2:
                    self._delete_edge(instance_id, args[0], args[-1])
                elif fn_name == "replace_node" and len(args) == 2:
                    self._replace_node(instance_id, args[0], args[1])
            except Exception as e:
                logger.warning("Refinement action parse error: %s – %s", action, e)

    def _parse_action_string(self, action: str) -> Tuple[str, List[str]]:
        """
        Parse an action string like 'insert_edge("subject", "relation", "object")'
        Returns (function_name, [arg1, arg2, ...])
        Handles entity names containing commas, parentheses, and quotes.
        """
        action = action.strip()
        # Match function name and arguments
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
    
    async def generate_response_simple(
        self, instance_id: str, retriever: EdgeRetriever, query: str, KG: nx.DiGraph, base_top_k: int = 10, sampling_params: dict = None, **kwargs
    ) -> Tuple[bool, str, float, dict]:
        """
        Generate the user response based on the assistant's output.
        Simple version.
        Just need to handle the case between refinement and RAG.
        """
        reward = 0.0
        should_terminate_sequence = False
        self._instance_dict[instance_id]["rag_state"] = False
        try:
            # need try because
            #TODO: the retrieved can be more advanced
            # perform RAG on the updated KG
            retriever = EdgeRetriever(self.retriever_config, self.llm_generator, self.reranker)
            retrieved_context = await retriever.retrieve_context(
                question=query,
                kg=KG,
                sampling_params=sampling_params,
                reward_function=self.reward_function
            )
            output = retrieved_context
            self._instance_dict[instance_id]["rag_state"] = True
            reward = 1.0
        except Exception as e:
            logger.warning(f"Failed to retrieve context: {e}\nQuery: {query}")
            should_terminate_sequence = True
            output = "Failed to retrieve context."
            reward = 0.0
            return should_terminate_sequence, output, reward, {}
        return should_terminate_sequence, output, reward, {}
    
    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, str, float, dict]:
        """
        Generate the user response based on the assistant's output.

        If the assistant's response includes <plan>...</plan>, ask the assistant to rewrite the query.
        Otherwise, ask the assistant to generate a response based on the current retrieved context.
        """
        should_terminate_sequence = False
        iterative = kwargs.get("iterative", True)
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
        # the hierarchy is 
        # plan > answer > no plan or answer
        reward = 0.0
        # Check if the assistant's response includes <plan>...</plan>
        # give format reward
        self._instance_dict[instance_id]["rag_state"] = False
        try:
            result_json = json_repair.loads(content)
        except Exception as e:
            logger.warning(f"Failed to parse assistant content as JSON: {e}\nContent: {content}")
            should_terminate_sequence = True
            response = "The response is not in the correct format."
            reward = 0.0
            return should_terminate_sequence, response, reward, {}

        if isinstance(result_json, dict) and "answer" in result_json:
            # answer generation
            answer = result_json.get("answer", "")
            should_terminate_sequence = True
            response = "<answer>" + answer + "</answer>"
        else:
            should_terminate_sequence = True
            response = "The response is not in the correct format."
            reward = 0.0
        return should_terminate_sequence, response, reward, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """
        Finalize the interaction by cleaning up the instance data.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
