import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import asyncio
from typing import Literal
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

load_dotenv()

# 检测并创建文件夹
WORKING_DIR = os.getenv("WORKING_DIR", "./dickens")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    model = os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o"
    return await openai_complete_if_cache(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_API_BASE", ""),
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        llm_model_max_async=5,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # Initialize RAG instance
        results = pd.DataFrame(columns=["QueryParam", "Result"])
        rag = await initialize_rag()
        print("RAG initialized successfully.")

        query_modes: list[Literal["naive", "local", "global", "hybrid"]] = [
            "naive",
            "local",
            "global",
            "hybrid",
        ]
        for mode in query_modes:
            result = await rag.aquery(
                "这个故事的核心主题是什么？", param=QueryParam(mode)
            )
            results = pd.concat(
                [
                    results,
                    pd.DataFrame({"QueryParam": [mode], "Result": [result]}),
                ],
                ignore_index=True,
            )
            print(f"Result for {mode}: {result}")

        # Ensure the result directory exists
        os.makedirs("./result", exist_ok=True)
        results.to_csv("./result/query_results.csv", index=False)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
