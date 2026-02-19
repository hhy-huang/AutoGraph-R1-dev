"""
prepare refinement RL training data for graph_refinement
read data from existing parquet file, replace prompt with refinement related prompt
"""
import pandas as pd
import json
import sys
import os
import pickle
import asyncio
from pathlib import Path
from tqdm import tqdm

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
from openai import OpenAI
from atlas_rag.vectorstore.embedding_model import Qwen3Emb
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.retriever.simple_retriever import SimpleGraphRetriever
from atlas_rag.vectorstore.create_graph_index import create_embeddings_and_index


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


async def process_row(row, row_index, retriever, base_top_k=10):
    """process single row data - async version"""
    # get extra_info
    extra_info = row.get("extra_info", {})
    if isinstance(extra_info, str):
        extra_info = json.loads(extra_info)
    elif not isinstance(extra_info, dict):
        extra_info = {}
    
    # get question
    question = extra_info.get("question", "")
    
    if not question:
        print(f"Warning: Row {row_index} has no question, skipping...")
        return None
    
    # Retrieve subgraph for the first hop (step 0) - async
    triples_string, retrieved_subgraph = await retrieve_subgraph(retriever, question, base_top_k)
    
    # Build judgement prompt (first answerable judgement prompt)
    judgement_prompt = build_judgement_prompt(question, triples_string)
    
    # build new interaction_kwargs
    interaction_kwargs = {
        "name": "graph_refinement",
        "question": question,
    }
    
    # keep necessary information from original interaction_kwargs
    old_interaction_kwargs = extra_info.get("interaction_kwargs", {})
    if isinstance(old_interaction_kwargs, dict):
        if "ground_truth" in old_interaction_kwargs:
            interaction_kwargs["ground_truth"] = old_interaction_kwargs["ground_truth"]
        if "supporting_context" in old_interaction_kwargs:
            interaction_kwargs["supporting_context"] = old_interaction_kwargs["supporting_context"]
    
    # add full_graph_data_path (pkl path)
    interaction_kwargs["full_graph_data_path"] = "/data/haoyuhuang/data/AtlasTune/data/train_full_kg.pkl"
    
    # Store prompt templates in extra_info
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

    # build new extra_info
    new_extra_info = {
        "index": str(row_index),
        "need_tools_kwargs": extra_info.get("need_tools_kwargs", False),
        "question": question,
        "split": extra_info.get("split", "train"),
        "interaction_kwargs": interaction_kwargs,
        **prompt_templates  # Add prompt templates
    }
    
    # process reward_model (maybe dict or str)
    reward_model = row.get("reward_model")
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except:
            pass  # keep original
    
    # return processed data
    return {
        "data_source": row.get("data_source", "graph_refinement"),
        "prompt": judgement_prompt,  # Only first judgement prompt
        "ability": "graph_refinement",
        "reward_model": reward_model,  # keep original
        "extra_info": new_extra_info,
        "metadata": row.get("metadata"),
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


async def process_batch(rows_batch, retriever, base_top_k=10, semaphore=None):
    """Process a batch of rows asynchronously"""
    tasks = []
    indices = []
    for idx, row in rows_batch:
        indices.append(idx)
        if semaphore:
            # Use semaphore to limit concurrent requests
            # Create a closure to capture the current row and idx
            async def process_with_semaphore(r=row, i=idx):
                async with semaphore:
                    return await process_row(r, i, retriever, base_top_k)
            tasks.append(process_with_semaphore())
        else:
            tasks.append(process_row(row, idx, retriever, base_top_k))
    
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


async def main_async():
    input_file = "/data/haoyuhuang/data/AtlasTune/data/mixed_hotpot_musique_valid_doc_size_15_distract_False_iterate.parquet"
    output_file = "/data/haoyuhuang/data/AtlasTune/data/mixed_hotpot_musique_valid_doc_size_15_distract_False_iterate_refinement.parquet"
    # input_file = "/data/haoyuhuang/data/AtlasTune/data/mixed_hotpot_musique_train_doc_size_15_distract_False_iterate.parquet"
    # output_file = "/data/haoyuhuang/data/AtlasTune/data/mixed_hotpot_musique_train_doc_size_15_distract_False_iterate_refinement.parquet"
    pkl_path = "/data/haoyuhuang/data/AtlasTune/checkpoints/20251211_234743_qwen2.5-3B-autograph-easy-docsize15-textlinkingFalse-loose/global_step_350/actor/huggingface/constructed_kg/hotpotqa_output/original_kg.pkl"
    base_top_k = 10  # Top K for first hop retrieval
    batch_size = 50  # Process rows in batches
    max_concurrent = 20  # Maximum concurrent requests
    
    # Configuration for encoder and LLM (can be modified via environment variables)
    encoder_base_url = os.getenv("ENCODER_BASE_URL", "http://0.0.0.0:8128/v1")
    llm_base_url = os.getenv("LLM_BASE_URL", "http://0.0.0.0:8129/v1")
    encoder_model_name = os.getenv("ENCODER_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    
    print(f"Reading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Initialize retriever (using same pattern as benchmarking_graph_refiner.py)
    retriever = initialize_retriever(
        pkl_path=pkl_path,
        encoder_base_url=encoder_base_url,
        llm_base_url=llm_base_url,
        encoder_model_name=encoder_model_name,
        llm_model_name=llm_model_name
    )
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process rows in batches
    all_processed_rows = []
    total_failed = 0
    
    # Create batches
    rows_list = [(idx, row) for idx, row in df.iterrows()]
    batches = [rows_list[i:i + batch_size] for i in range(0, len(rows_list), batch_size)]
    
    print(f"Processing {len(batches)} batches with batch_size={batch_size}, max_concurrent={max_concurrent}")
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        processed_rows, failed_count = await process_batch(batch, retriever, base_top_k, semaphore)
        all_processed_rows.extend(processed_rows)
        total_failed += failed_count
        print(f"Batch {batch_idx + 1}/{len(batches)}: Processed {len(processed_rows)} rows, Failed {failed_count} rows")
    
    # save processed data
    print(f"\nProcessed {len(all_processed_rows)} rows successfully")
    print(f"Failed {total_failed} rows")
    df_processed = pd.DataFrame(all_processed_rows)
    df_processed.to_parquet(output_file, index=False)
    print(f"Saved processed data to {output_file}")
    
    # print some statistics
    print("\nStatistics:")
    print(f"Total rows: {len(df_processed)}")
    print(f"Output file: {output_file}")


def main():
    """Main entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
