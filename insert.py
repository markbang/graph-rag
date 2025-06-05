import os
from dotenv import load_dotenv
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status
from graph_analysis import get_info

from lightrag.prompt import PROMPTS

print(f"使用实体提取提示词: {PROMPTS['entity_extraction']}")
print(f"使用继续实体提取提示词: {PROMPTS['entity_continue_extraction']}")


load_dotenv()

WORKING_DIR = os.getenv("WORKING_DIR", "./dickens")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model=os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
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
        rag = await initialize_rag()
        print("RAG instance initialized.")
        insert_dir = "inputs"
        for file in os.listdir(insert_dir):
            if file.endswith(".txt"):
                print(f"Inserting {file} into RAG...")
                with open(f"{insert_dir}/{file}", "r", encoding="utf-8") as f:
                    await rag.ainsert(f.read())
        print("All files inserted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    print(get_info())


if __name__ == "__main__":
    asyncio.run(main())
