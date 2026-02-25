"""
prepare refinement RL training data for graph_refinement
Read queries from dataset under KGs path (e.g. KGs/hotpotqa/*.json or *.jsonl),
use original_kg.pkl in the same dataset dir for retrieval, output parquet for RL.
"""
import argparse
import random
import pandas as pd
import json
import sys
import os
import pickle
import asyncio
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# add project path (go up to project root: autorefiner/scripts -> autorefiner -> project_root)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from autograph.rag_server.reafiner_prompt import (
    REAFINER_JUDGEMENT_SYSTEM_PROMPT,
    REAFINER_JUDGEMENT_USER_PROMPT,
    REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT,
    REAFINER_ERROR_ABDUCTION_USER_PROMPT,
    REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT,
    REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT,
)

# Import retriever and encoder
from openai import OpenAI, AsyncOpenAI
from atlas_rag.vectorstore.embedding_model import Qwen3Emb
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.retriever.simple_retriever import SimpleGraphRetriever
from atlas_rag.vectorstore.create_graph_index import create_embeddings_and_index

# token-level F1 used for filtering (same as training reward)
from verl.third_party.autograph_r1.f1_reward import compute_f1 as token_f1

# Same retriever as rollout for draft_answer (EdgeRetriever)
from autograph.rag_server.base_retriever import RetrieverConfig
from autograph.rag_server.edge_retriever import EdgeRetriever
from autograph.rag_server.reranker_api import Reranker
from autograph.rag_server.llm_api import LLMGenerator as AutographLLMGenerator


# Default KGs base path: dataset dir = KGS_BASE / dataset_name, contains original_kg.pkl and dataset json/jsonl
KGS_BASE = "/data/haoyuhuang/data/AtlasTune/data/KGs"
# Key in dataset JSON/JSONL for the query text (HotpotQA/MuSiQue use "question")
QUERY_KEY = "question"


def load_queries_from_dataset(dataset_dir, query_key=QUERY_KEY, split=None):
    """
    Load query records from a dataset directory under KGs.
    Scans for *.json and *.jsonl; each record should have query_key (e.g. "question").
    Other keys (e.g. answer, id) are kept in the record for extra_info/ground_truth.

    Args:
        dataset_dir: Path to dataset dir, e.g. KGs/hotpotqa
        query_key: Key for query string in each record (default "question")
        split: If set, only load files whose name contains this (e.g. "dev", "train")

    Yields:
        dict per record: at least {query_key: str}, plus any other keys from json
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    for ext in ("*.json", "*.jsonl"):
        for path in sorted(dataset_dir.glob(ext)):
            if split and split not in path.name.lower():
                continue
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if query_key not in rec or not rec[query_key]:
                            continue
                        yield rec
                elif path.suffix == ".json":
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, list):
                        items = data
                    elif isinstance(data, dict):
                        items = data.get("data", data.get("questions", data.get("instances", [data])))
                        if not isinstance(items, list):
                            items = [items]
                    else:
                        continue
                    for rec in items:
                        if not isinstance(rec, dict) or query_key not in rec or not rec[query_key]:
                            continue
                        yield rec


def find_original_kg_pkl(dataset_dir):
    """Return path to original_kg.pkl under dataset_dir (direct or in one subdir)."""
    dataset_dir = Path(dataset_dir)
    direct = dataset_dir / "original_kg.pkl"
    if direct.exists():
        return str(direct)
    for sub in dataset_dir.iterdir():
        if sub.is_dir():
            p = sub / "original_kg.pkl"
            if p.exists():
                return str(p)
    return str(direct)


def format_triples_string(triples):
    """format triples list to string"""
    if isinstance(triples, list):
        return json.dumps(triples, ensure_ascii=False, indent=2)
    elif isinstance(triples, str):
        try:
            # try to parse as JSON, if successful, format it
            parsed = json.loads(triples)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except:
            return triples
    return str(triples)


def build_judgement_prompt(question, triples_string):
    """
    Build the first answerable judgement prompt with retrieved subgraph
    
    Args:
        question: The question string
        triples_string: Formatted triples string from retrieved subgraph
    
    Returns:
        List of messages for the judgement prompt
    """
    prompt = [
        {
            "role": "system",
            "content": REAFINER_JUDGEMENT_SYSTEM_PROMPT.strip()
        },
        {
            "role": "user",
            "content": REAFINER_JUDGEMENT_USER_PROMPT.format(
                question=question,
                triples_string=triples_string
            ).strip()
        }
    ]
    return prompt


async def retrieve_subgraph(retriever, question, base_top_k=10):
    """
    Retrieve subgraph for the first hop (step 0) - async version
    
    Args:
        retriever: SimpleGraphRetriever instance
        question: Query string
        base_top_k: Top K edges to retrieve
    
    Returns:
        triples_string: Formatted string of retrieved triples
        retrieved_subgraph: List of triple dicts
    """
    try:
        # Run synchronous retrieve in thread pool to avoid blocking
        # Use functools.partial to properly pass keyword argument
        import functools
        retrieve_func = functools.partial(retriever.retrieve, question, topN=base_top_k)
        loop = asyncio.get_event_loop()
        sorted_context, sorted_context_ids = await loop.run_in_executor(None, retrieve_func)
        
        # Format triples as newline-separated string (same as reafiner.py line 219)
        # sorted_context is already a list of strings like ["subject1  relation1  object1", ...]
        triples_string = "\n".join(sorted_context)
        
        # Convert to triple dict format for potential future use
        retrieved_subgraph = []
        for triple_str in sorted_context:
            parts = triple_str.split("  ")
            if len(parts) == 3:
                retrieved_subgraph.append({
                    "subject": parts[0],
                    "relation": parts[1],
                    "object": parts[2]
                })
        
        return triples_string, retrieved_subgraph
    except Exception as e:
        print(f"Error retrieving subgraph for question '{question}': {e}")
        # Return empty subgraph on error
        return "", []


async def process_row(record, row_index, retriever, base_top_k=10, full_graph_data_path=None, query_key=QUERY_KEY,
                     edge_retriever=None, full_kg=None):
    """
    Process a single sample: retrieve subgraph, build judgement prompt, build extra_info.
    If edge_retriever and full_kg are provided, use same config as rollout to get answer and set interaction_kwargs["draft_answer"].
    """
    # Support both parquet-style (extra_info.question) and dataset-style (top-level question/query_key)
    question = record.get(query_key) or (isinstance(record.get("extra_info"), dict) and record["extra_info"].get("question")) or ""
    if isinstance(record.get("extra_info"), str):
        try:
            ei = json.loads(record["extra_info"])
            question = question or ei.get("question", "")
        except Exception:
            pass
    if not question:
        return None

    # Retrieve subgraph for the first hop (step 0) - async
    triples_string, retrieved_subgraph = await retrieve_subgraph(retriever, question, base_top_k)
    judgement_prompt = build_judgement_prompt(question, triples_string)

    interaction_kwargs = {
        "name": "graph_refinement",
        "question": question,
    }
    if full_graph_data_path:
        interaction_kwargs["full_graph_data_path"] = full_graph_data_path

    # Draft answer: use EdgeRetriever over the full KG, but reuse a single
    # precomputed KG index for all questions (only query embeddings vary).
    if edge_retriever is not None:
        try:
            if hasattr(edge_retriever, "_cached_triple_embeddings"):
                draft_answer = await draft_answer_with_cached_index(
                    edge_retriever=edge_retriever,
                    question=question,
                )
                interaction_kwargs["draft_answer"] = draft_answer
        except Exception as e:
            print(f"Warning: draft_answer failed for row {row_index}: {e}")
            interaction_kwargs["draft_answer"] = ""

    # ground_truth: from parquet extra_info, or dataset json (answer / answers). Normalize to list for interaction_kwargs and reward.
    old_interaction_kwargs = {}
    if isinstance(record.get("extra_info"), dict):
        old_interaction_kwargs = record["extra_info"].get("interaction_kwargs", {})
    if isinstance(old_interaction_kwargs, dict) and "supporting_context" in old_interaction_kwargs:
        interaction_kwargs["supporting_context"] = old_interaction_kwargs["supporting_context"]

    gt_raw = None
    if isinstance(old_interaction_kwargs, dict) and "ground_truth" in old_interaction_kwargs:
        gt_raw = old_interaction_kwargs["ground_truth"]
    if gt_raw is None and "answer" in record:
        gt_raw = record["answer"]
    if gt_raw is None and "answers" in record:
        gt_raw = record["answers"]
    if isinstance(gt_raw, list):
        gt_list = [str(x).strip() for x in gt_raw if x is not None]
    elif gt_raw is not None:
        gt_list = [str(gt_raw).strip()]
    else:
        gt_list = []
    if not gt_list:
        gt_list = [""]  # avoid rollout .get("ground_truth")[0] IndexError
    interaction_kwargs["ground_truth"] = gt_list

    prompt_templates = {
        "prompt_template_judgement": {
            "system": REAFINER_JUDGEMENT_SYSTEM_PROMPT.strip(),
            "user": REAFINER_JUDGEMENT_USER_PROMPT.strip()
        },
        "prompt_template_abduction": {
            "system": REAFINER_ERROR_ABDUCTION_SYSTEM_PROMPT.strip(),
            "user": REAFINER_ERROR_ABDUCTION_USER_PROMPT.strip()
        },
        "prompt_template_action": {
            "system": REAFINER_KG_REFINEMENT_ACTION_SYSTEM_PROMPT.strip(),
            "user": REAFINER_KG_REFINEMENT_ACTION_USER_PROMPT.strip()
        }
    }
    interaction_kwargs.update(prompt_templates)

    new_extra_info = {
        "index": str(row_index),
        "need_tools_kwargs": record.get("need_tools_kwargs", False),
        "question": question,
        "split": record.get("split", "train"),
        "interaction_kwargs": interaction_kwargs,
        **prompt_templates
    }

    reward_model = record.get("reward_model")
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except Exception:
            reward_model = {}
    if not isinstance(reward_model, dict):
        reward_model = {}
    # Reward manager reads reward_model["ground_truth"]; f1_reward expects ground_truth["target"] as list
    reward_model["ground_truth"] = {"target": gt_list}

    return {
        "data_source": record.get("data_source", "graph_refinement"),
        "prompt": judgement_prompt,
        "ability": "graph_refinement",
        "reward_model": reward_model,
        "extra_info": new_extra_info,
        "metadata": record.get("metadata"),
    }


def initialize_retriever(pkl_path, encoder_base_url="http://0.0.0.0:8128/v1", 
                         llm_base_url="http://0.0.0.0:8129/v1", 
                         encoder_model_name="Qwen/Qwen3-Embedding-0.6B",
                         llm_model_name="Qwen/Qwen2.5-7B-Instruct",
                         working_directory=None,
                         keyword=None):
    """
    Initialize retriever with KG data from pkl file or create embeddings if needed
    Following the same pattern as benchmarking_graph_refiner.py
    
    Args:
        pkl_path: Path to the pkl file containing KG data
        encoder_base_url: Base URL for encoder API
        llm_base_url: Base URL for LLM API
        encoder_model_name: Model name for encoder
        llm_model_name: Model name for LLM
        working_directory: Working directory for creating embeddings (if pkl doesn't exist)
        keyword: Keyword for creating embeddings (if pkl doesn't exist)
    
    Returns:
        retriever: SimpleGraphRetriever instance
    """
    # Initialize sentence encoder (same as benchmarking_graph_refiner.py)
    print("Initializing sentence encoder...")
    sentence_model = OpenAI(base_url=encoder_base_url, api_key="EMPTY KEY")
    sentence_encoder = Qwen3Emb(sentence_model)
    
    # Load or create KG data
    if os.path.exists(pkl_path):
        print(f"Loading KG data from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    else:
        if working_directory is None or keyword is None:
            raise ValueError(f"Pkl file {pkl_path} not found. Must provide working_directory and keyword to create embeddings.")
        print(f"Pkl file not found. Creating embeddings and index...")
        data = create_embeddings_and_index(
            sentence_encoder=sentence_encoder,
            model_name=encoder_model_name,
            working_directory=working_directory,
            keyword=keyword,
            include_concept=False,
            include_events=False,
            normalize_embeddings=False,
            text_batch_size=512,
            node_and_edge_batch_size=512,
            use_flat_index=True
        )
    
    # Initialize LLM generator (same as benchmarking_graph_refiner.py)
    print("Initializing LLM generator...")
    client = OpenAI(base_url=llm_base_url, api_key="EMPTY KEY")
    llm_generator = LLMGenerator(client=client, model_name=llm_model_name)
    
    # Initialize retriever (same as benchmarking_graph_refiner.py)
    print("Initializing retriever...")
    retriever = SimpleGraphRetriever(
        llm_generator=llm_generator,
        sentence_encoder=sentence_encoder,
        data=data,
    )
    
    print("Retriever initialized successfully!")
    return retriever


def load_full_graph_and_edge_retriever(pkl_path, encoder_base_url="http://0.0.0.0:8128/v1",
                                       llm_base_url="http://0.0.0.0:8129/v1",
                                       llm_model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Load full_graph_data from pkl (must have "KG" key) and create EdgeRetriever with same config as rollout.
    Returns (full_kg, edge_retriever) or (None, None) if pkl has no KG.
    """
    if not os.path.exists(pkl_path):
        return None, None
    with open(pkl_path, "rb") as f:
        full_graph_data = pickle.load(f)
    kg = full_graph_data.get("KG")
    if kg is None:
        return None, None
    emb_client = AsyncOpenAI(base_url=encoder_base_url, api_key="EMPTY KEY")
    reranker = Reranker(emb_client)
    llm_client = AsyncOpenAI(base_url=llm_base_url, api_key="EMPTY KEY")
    llm_generator = AutographLLMGenerator(llm_client, llm_model_name, backend="openai")
    config = RetrieverConfig("re_edge")
    edge_retriever = EdgeRetriever(config, llm_generator, reranker)
    return kg, edge_retriever


async def precompute_triple_embeddings_for_edge_retriever(edge_retriever: EdgeRetriever, kg, batch_size: int = 100):
    """
    Precompute triple embeddings for the full KG once, and cache them on edge_retriever.
    This mirrors EdgeRetriever.index_kg but only does the KG part (no per-query work).
    """
    if kg is None or kg.number_of_edges() == 0:
        edge_retriever._cached_triple_embeddings = None
        edge_retriever._cached_edge_list = []
        edge_retriever._cached_kg = kg
        return

    edge_list = list(kg.edges)
    triples = []
    for src, dst in edge_list:
        rel = kg.edges[src, dst].get("relation", "")
        triples.append(f"{src} {rel} {dst}")

    triple_emb_batches = []
    for i in range(0, len(triples), batch_size):
        batch_triples = triples[i : i + batch_size]
        emb = await edge_retriever.reranker.embed(batch_triples)
        triple_emb_batches.append(emb)

    if triple_emb_batches:
        triple_embeddings = torch.cat(triple_emb_batches, dim=0)
    else:
        triple_embeddings = torch.zeros((0, 1024))  # fallback, should not be used

    edge_retriever._cached_triple_embeddings = triple_embeddings  # shape: (num_edges, dim)
    edge_retriever._cached_edge_list = edge_list
    edge_retriever._cached_kg = kg


async def draft_answer_with_cached_index(
    edge_retriever: EdgeRetriever,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Given a question, compute draft answer using:
    - cached triple embeddings / KG (precomputed once)
    - per-question query embedding + same scoring & answer generation logic
      as EdgeRetriever.retrieve.
    """
    if not hasattr(edge_retriever, "_cached_triple_embeddings"):
        return ""

    triple_embeddings = edge_retriever._cached_triple_embeddings
    edge_list = getattr(edge_retriever, "_cached_edge_list", [])
    kg = getattr(edge_retriever, "_cached_kg", None)

    if triple_embeddings is None or triple_embeddings.size(0) == 0 or kg is None:
        return ""

    # Build query instruct in the same way as EdgeRetriever.index_kg
    task = "Given a question, retrieve the most relevant knowledge graph triple."
    query_instruct = f"Instruct: {task}\nQuery: {question}"
    query_embedding = await edge_retriever.reranker.embed([query_instruct])  # (1, dim)

    # Score edges and select top N
    edge_scores = query_embedding @ triple_embeddings.T  # (1, num_edges)
    edge_scores_np = edge_scores.detach().cpu().numpy().flatten()
    top_edge_count = min(edge_retriever.config.topN_retrieval_edges, len(edge_scores_np))
    if top_edge_count == 0:
        return ""
    top_edge_indices = np.argsort(edge_scores_np)[-top_edge_count:][::-1]

    edge_str_lst = []
    for idx in top_edge_indices:
        src, dst = edge_list[int(idx)]
        relation = kg.edges[src, dst].get("relation", "")
        edge_str_lst.append(f"({kg.nodes[src]['id']}-{relation}->{kg.nodes[dst]['id']})")
    edge_str = "\n".join(edge_str_lst)

    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": temperature, "frequency_penalty": 0.0}
    edge_retriever.sampling_params = sampling_params
    answer = await edge_retriever.generate_answer(question, edge_str)
    return answer


def filter_rows_by_f1(rows, threshold: float = 0.9):
    """
    Filter out rows whose draft_answer is too close to ground truth
    in token-level F1 (max over all targets) with score > threshold.
    """
    if not rows:
        return rows, 0

    kept = []
    removed = 0
    for r in rows:
        extra = r.get("extra_info", {}) or {}
        ik = extra.get("interaction_kwargs", {}) or {}
        draft = ik.get("draft_answer", "")
        targets = ik.get("ground_truth", [])

        # Ensure types
        if not isinstance(draft, str) or not draft.strip():
            kept.append(r)
            continue
        if isinstance(targets, str):
            targets = [targets]
        if not isinstance(targets, list) or not targets:
            kept.append(r)
            continue

        best_f1 = 0.0
        for t in targets:
            if not isinstance(t, str) or not t.strip():
                continue
            f1 = token_f1(draft, t)
            if f1 > best_f1:
                best_f1 = f1
        if best_f1 > threshold:
            removed += 1
            continue
        kept.append(r)

    return kept, removed


async def process_batch(rows_batch, retriever, base_top_k=10, semaphore=None, full_graph_data_path=None, query_key=QUERY_KEY,
                        edge_retriever=None, full_kg=None):
    """Process a batch of records asynchronously. Each item in rows_batch is (idx, record)."""
    tasks = []
    indices = []
    for idx, row in rows_batch:
        indices.append(idx)
        if semaphore:
            async def process_with_semaphore(r=row, i=idx):
                async with semaphore:
                    return await process_row(r, i, retriever, base_top_k, full_graph_data_path, query_key,
                                            edge_retriever=edge_retriever, full_kg=full_kg)
            tasks.append(process_with_semaphore())
        else:
            tasks.append(process_row(row, idx, retriever, base_top_k, full_graph_data_path, query_key,
                                    edge_retriever=edge_retriever, full_kg=full_kg))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_rows = []
    failed_count = 0
    for idx, result in zip(indices, results):
        if isinstance(result, Exception):
            print(f"Error processing row {idx}: {result}")
            import traceback
            traceback.print_exc()
            failed_count += 1
        elif result is None:
            failed_count += 1
        else:
            processed_rows.append(result)
    return processed_rows, failed_count


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare refinement RL data from KGs dataset dir")
    parser.add_argument("--kgs-base", type=str, default=KGS_BASE, help="Base path for KGs (e.g. .../data/KGs)")
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name under kgs_base (e.g. hotpotqa)")
    parser.add_argument("--split", type=str, default=None, help="Only load files containing this (e.g. dev, train). If not set, load all json/jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path. Default: {kgs_base}/{dataset}/{dataset}_{split}_refinement.parquet")
    parser.add_argument("--query-key", type=str, default=QUERY_KEY, help="Key for query text in dataset json (default: question)")
    parser.add_argument("--base-top-k", type=int, default=10, help="Top K for first hop retrieval")
    parser.add_argument("--batch-size", type=int, default=50, help="Process batch size")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent retrieval requests")
    parser.add_argument("--no-draft-answer", action="store_true", help="Do not compute draft_answer via EdgeRetriever")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for train (rest for valid). Set to 0 to disable train/valid split (single output).",
    )
    parser.add_argument(
        "--draft-f1-threshold",
        type=float,
        default=0.9,
        help="Filter out items whose draft_answer vs ground_truth token-level F1 is higher than this threshold.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/valid split")
    return parser.parse_args()


async def main_async():
    args = parse_args()
    dataset_dir = Path(args.kgs_base) / args.dataset
    pkl_path = find_original_kg_pkl(dataset_dir)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"original_kg.pkl not found under {dataset_dir}. Expected: {pkl_path}")

    out_dir = Path(args.kgs_base) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ratio = getattr(args, "train_ratio", 0.8)
    if args.output and train_ratio <= 0:
        output_file = args.output
    elif not args.output:
        suffix = f"_{args.split}" if args.split else ""
        output_file = str(out_dir / f"{args.dataset}{suffix}_refinement.parquet")
    else:
        output_file = None

    base_top_k = args.base_top_k
    batch_size = args.batch_size
    max_concurrent = args.max_concurrent

    # Load query records from dataset dir (json/jsonl)
    print(f"Loading queries from dataset dir: {dataset_dir} (split={args.split})")
    records = list(load_queries_from_dataset(dataset_dir, query_key=args.query_key, split=args.split))
    print(f"Loaded {len(records)} query records")

    if not records:
        print("No records found. Exiting.")
        return

    encoder_base_url = os.getenv("ENCODER_BASE_URL", "http://0.0.0.0:8128/v1")
    llm_base_url = os.getenv("LLM_BASE_URL", "http://0.0.0.0:8129/v1")
    encoder_model_name = os.getenv("ENCODER_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    retriever = initialize_retriever(
        pkl_path=pkl_path,
        encoder_base_url=encoder_base_url,
        llm_base_url=llm_base_url,
        encoder_model_name=encoder_model_name,
        llm_model_name=llm_model_name
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    full_kg, edge_retriever = None, None
    if not getattr(args, "no_draft_answer", False):
        print("Loading full KG and EdgeRetriever for draft_answer...")
        full_kg, edge_retriever = load_full_graph_and_edge_retriever(
            pkl_path, encoder_base_url=encoder_base_url,
            llm_base_url=llm_base_url, llm_model_name=llm_model_name
        )
        if full_kg is not None and edge_retriever is not None:
            print("Precomputing KG index for draft_answer (single index, many queries)...")
            await precompute_triple_embeddings_for_edge_retriever(edge_retriever, full_kg)
            print("Draft answers will be computed with EdgeRetriever (same as rollout) using cached KG index.")
        else:
            print("Could not load KG for draft_answer; skipping (use --no-draft-answer to suppress).")
            full_kg, edge_retriever = None, None

    rows_list = [(i, rec) for i, rec in enumerate(records)]
    batches = [rows_list[i:i + batch_size] for i in range(0, len(rows_list), batch_size)]
    print(f"Processing {len(batches)} batches with batch_size={batch_size}, max_concurrent={max_concurrent}")

    all_processed_rows = []
    total_failed = 0
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        processed_rows, failed_count = await process_batch(
            batch, retriever, base_top_k, semaphore,
            full_graph_data_path=pkl_path,
            query_key=args.query_key,
            edge_retriever=edge_retriever,
            full_kg=full_kg,
        )
        all_processed_rows.extend(processed_rows)
        total_failed += failed_count
        print(f"Batch {batch_idx + 1}/{len(batches)}: Processed {len(processed_rows)} rows, Failed {failed_count} rows")

    print(f"\nProcessed {len(all_processed_rows)} rows successfully")
    print(f"Failed {total_failed} rows")

    train_ratio = getattr(args, "train_ratio", 0.8)
    seed = getattr(args, "seed", 42)
    f1_threshold = getattr(args, "draft_f1_threshold", 0.9)
    if train_ratio > 0 and train_ratio < 1 and all_processed_rows:
        random.seed(seed)
        shuffled = list(all_processed_rows)
        random.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_ratio))
        train_rows = shuffled[:n_train]
        valid_rows = shuffled[n_train:]

        # 先在 train / valid 上分别按 draft_answer vs ground_truth 的 token-level F1 过滤
        train_rows, removed_train = filter_rows_by_f1(train_rows, threshold=f1_threshold)
        valid_rows, removed_valid = filter_rows_by_f1(valid_rows, threshold=f1_threshold)

        for r in train_rows:
            r["extra_info"]["split"] = "train"
        for r in valid_rows:
            r["extra_info"]["split"] = "valid"
        out_dir = Path(args.kgs_base) / args.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        train_file = out_dir / f"{args.dataset}_train_refinement.parquet"
        valid_file = out_dir / f"{args.dataset}_valid_refinement.parquet"
        pd.DataFrame(train_rows).to_parquet(str(train_file), index=False)
        pd.DataFrame(valid_rows).to_parquet(str(valid_file), index=False)
        print(f"Saved train ({len(train_rows)} rows, removed {removed_train} by F1>{f1_threshold}) to {train_file}")
        print(f"Saved valid ({len(valid_rows)} rows, removed {removed_valid} by F1>{f1_threshold}) to {valid_file}")
        print(f"Train/valid ratio: {train_ratio:.0%} / {1 - train_ratio:.0%} (seed={seed})")
    else:
        df_processed = pd.DataFrame(all_processed_rows)
        out_path = output_file or str(out_dir / f"{args.dataset}_refinement.parquet")
        df_processed.to_parquet(out_path, index=False)
        print(f"Saved processed data to {out_path}")
    print("\nStatistics:")
    print(f"Total rows: {len(all_processed_rows)}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
