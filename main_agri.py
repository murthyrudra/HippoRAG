import os
from typing import List
import json
import nltk
from nltk.tokenize import sent_tokenize

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

import argparse

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("RITS_API_KEY")

nltk.download("punkt")


def chunk_by_sentences(text, max_chars=250):
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # If adding this sentence exceeds limit,
        # finalize current chunk and start a new one
        if current_chunk and current_len + sent_len + 1 > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len + (1 if current_chunk else 0)

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument(
        "--dataset", type=str, default="ap_agri_telugu", help="Dataset name"
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1/",
        help="LLM base URL",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="meta-llama/llama-3-3-70b-instruct",
        help="LLM name",
    )
    parser.add_argument(
        "--embedding_name",
        type=str,
        default="ibm-granite/granite-embedding-english-r2",
        help="embedding model name",
    )
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-english-r2/v1",
        help="Embedding base URL",
    )
    parser.add_argument(
        "--force_index_from_scratch",
        type=str,
        default="false",
        help="If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.",
    )
    parser.add_argument(
        "--force_openie_from_scratch",
        type=str,
        default="false",
        help="If set to False, will try to first reuse openie results for the corpus if they exist.",
    )
    parser.add_argument(
        "--openie_mode",
        choices=["online", "offline"],
        default="online",
        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes",
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs", help="Save directory"
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name

    if save_dir == "outputs":
        save_dir = save_dir + "/" + dataset_name
    else:
        save_dir = save_dir + "_" + dataset_name

    corpus_path = (
        f"/Users/rudramurthy/Documents/GitHub/IRL-Indic-RAG/data/seed_data.jsonl"
    )
    with open(corpus_path, "r") as f:
        docs = []
        for each_line in f:
            text = json.loads(each_line)

            if "chunk" in text:
                text = text["chunk"]
            else:
                text = text["document"]
            docs.extend(chunk_by_sentences(text))

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
        force_index_from_scratch=force_index_from_scratch,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
        force_openie_from_scratch=force_openie_from_scratch,
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode=args.openie_mode,
    )

    logging.basicConfig(level=logging.INFO)

    hipporag = HippoRAG(
        global_config=config,
        extra_headers={"RITS_API_KEY": api_key},
    )

    hipporag.index(docs)

    # Retrieval and QA
    # hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)


if __name__ == "__main__":
    main()
