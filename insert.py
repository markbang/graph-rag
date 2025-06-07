import os
from dotenv import load_dotenv
import asyncio
import logging
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

# 设置简单的日志记录 - 同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("insert.log"),
        logging.StreamHandler(),  # 添加控制台输出
    ],
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Detected embedding dimension: {embedding_dimension}")

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
    logger.info("RAG instance initialized successfully")

    return rag


async def main():
    try:
        logger.info("Starting RAG insertion process")

        # Initialize RAG instance
        rag = await initialize_rag()

        insert_dir = "inputs"
        logger.info(f"Processing files from directory: {insert_dir}")

        files_processed = 0
        for file in os.listdir(insert_dir):
            if file.endswith(".txt"):
                logger.info(f"Inserting {file} into RAG...")
                with open(f"{insert_dir}/{file}", "r", encoding="utf-8") as f:
                    content = f.read()
                    logger.info(f"File {file} has {len(content)} characters")
                    await rag.ainsert(content)
                files_processed += 1
                logger.info(f"Successfully inserted {file}")

        logger.info(
            f"✅ Insert completed successfully! Processed {files_processed} files"
        )

        # 检查生成的文件
        logger.info("Checking generated files in working directory:")
        for item in os.listdir(WORKING_DIR):
            item_path = os.path.join(WORKING_DIR, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                logger.info(f"  - {item}: {size} bytes")

        # 显示成功完成的消息到标准输出（供 source_modifier.py 捕获）
        print("INSERT_COMPLETED_SUCCESSFULLY")

        # 确保所有异步任务完成并清理资源
        logger.info("Cleaning up resources...")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"INSERT_FAILED: {e}")
        raise
    finally:
        # 强制退出以确保进程结束
        logger.info("Process completed, exiting...")
        import sys

        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Process interrupted")
        import sys

        sys.exit(1)
