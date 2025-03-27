import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc
import numpy as np

setup_logger("lightrag", level="INFO")


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("V3_API_KEY"),
        base_url="https://api.gpt.ge/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="text-embedding-3-small",
        api_key=os.getenv("V3_API_KEY"),
        base_url="https://api.gpt.ge/v1",
    )


async def initialize_rag():
    rag = LightRAG(
        working_dir="input",
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=4096, max_token_size=8192, func=embedding_func
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    # Insert text
    rag.insert("Your text")

    # Perform naive search
    mode = "naive"
    # Perform local search
    mode = "local"
    # Perform global search
    mode = "global"
    # Perform hybrid search
    mode = "hybrid"
    # Mix mode Integrates knowledge graph and vector retrieval.
    mode = "mix"

    rag.query("What are the top themes in this story?", param=QueryParam(mode=mode))


if __name__ == "__main__":
    main()
