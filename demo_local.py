import os
from typing import List
import json
import argparse
import logging

from nltk.tokenize import sent_tokenize
import nltk

from src.hipporag import HippoRAG
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("RITS_API_KEY")

nltk.download("punkt")


def chunk_by_sentences(text, max_chars=1000):
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

    # Prepare datasets and evaluation
    docs = []
    with open(
        "/Users/rudramurthy/Documents/GitHub/PageIndex/data/sarvam_output_md_orientation_corrected_translated_summaries.json",
        "r",
        errors="ignore",
        encoding="utf8",
    ) as f:
        data = json.load(f)

        for each_data in data:
            text = each_data.get("text", "")
            docs.extend(chunk_by_sentences(text))

    save_dir = "outputs/ap_agri_gpt_oss_120b"  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = "openai/gpt-oss-120b"  # Any OpenAI model name

    # Startup a HippoRAG instance
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name="ibm-granite/granite-embedding-english-r2",
        llm_base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b/v1",
        extra_headers={"RITS_API_KEY": api_key},
        embedding_base_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-english-r2/v1",
    )

    # Run indexing
    hipporag.index(docs=docs)

    # # Separate Retrieval & QA
    # queries = [
    #     "What is George Rankin's occupation?",
    #     "How did Cinderella reach her happy ending?",
    #     "What county is Erik Hort's birthplace a part of?",
    # ]

    # # For Evaluation
    # answers = [["Politician"], ["By going to the ball."], ["Rockland County"]]

    # print(hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers))


if __name__ == "__main__":
    main()
