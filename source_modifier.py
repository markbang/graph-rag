#!/usr/bin/env python3
"""
LightRAG源码提示词修改管理器
直接修改LightRAG源码中的提示词，每次修改后重新运行insert.py测试
确保提示词修改真正生效
"""

import os
import sys
import shutil
import logging
import subprocess
import pandas as pd
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LightRAGSourceModifier:
    def __init__(self):
        # LightRAG安装路径
        self.lightrag_path = (
            "/root/miniconda3/envs/dev/lib/python3.12/site-packages/lightrag"
        )
        self.prompt_file = os.path.join(self.lightrag_path, "prompt.py")

        # 配置文件
        self.prompt_versions_file = "prompt_versions.xlsx"
        self.results_file = "source_modification_results.csv"

        # 备份文件
        self.backup_prompt_file = "original_prompt.py.backup"

        # 工作目录
        self.working_dir = "./dickens"
        self.tobe_dir = "./tobe"

        # 确保目录存在
        os.makedirs(self.tobe_dir, exist_ok=True)

    def backup_original_prompt(self):
        """备份原始的prompt.py文件"""
        if not os.path.exists(self.backup_prompt_file):
            try:
                shutil.copy2(self.prompt_file, self.backup_prompt_file)
                logging.info(f"✓ 已备份原始prompt.py到: {self.backup_prompt_file}")
                return True
            except Exception as e:
                logging.error(f"✗ 备份原始prompt.py失败: {e}")
                return False
        else:
            logging.info(f"✓ 备份文件已存在: {self.backup_prompt_file}")
            return True

    def restore_original_prompt(self):
        """恢复原始的prompt.py文件"""
        if os.path.exists(self.backup_prompt_file):
            try:
                shutil.copy2(self.backup_prompt_file, self.prompt_file)
                logging.info("✓ 已恢复原始prompt.py文件")
                return True
            except Exception as e:
                logging.error(f"✗ 恢复原始prompt.py失败: {e}")
                return False
        else:
            logging.error("✗ 备份文件不存在，无法恢复")
            return False

    def read_prompt_file(self):
        """读取prompt.py文件内容"""
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.error(f"✗ 读取prompt.py文件失败: {e}")
            return None

    def write_prompt_file(self, content):
        """写入prompt.py文件内容"""
        try:
            with open(self.prompt_file, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info("✓ 已更新prompt.py源码文件")
            return True
        except Exception as e:
            logging.error(f"✗ 写入prompt.py文件失败: {e}")
            return False

    def modify_source_prompts(
        self, entity_extraction_prompt, entity_continue_prompt, version_name
    ):
        """直接修改源码中的提示词"""
        content = self.read_prompt_file()
        if not content:
            return False

        marked_entity_extraction = entity_extraction_prompt
        marked_entity_continue = entity_continue_prompt

        try:
            # 替换entity_extraction
            entity_start = content.find('PROMPTS["entity_extraction"] = """')
            if entity_start == -1:
                entity_start = content.find("PROMPTS['entity_extraction'] = '''")
                quote_type = "'''"
            else:
                quote_type = '"""'

            if entity_start != -1:
                # 找到开始位置后的第一个三引号
                start_pos = content.find(quote_type, entity_start) + len(quote_type)
                # 找到对应的结束三引号
                end_pos = content.find(quote_type, start_pos)

                if start_pos != -1 and end_pos != -1:
                    # 替换entity_extraction内容
                    new_content = (
                        content[:start_pos]
                        + marked_entity_extraction
                        + content[end_pos:]
                    )
                    content = new_content
                    logging.info(
                        f"✓ 成功替换entity_extraction提示词 (版本: {version_name})"
                    )
                else:
                    logging.error("✗ 无法找到entity_extraction的结束位置")
                    return False
            else:
                logging.error("✗ 无法找到entity_extraction的开始位置")
                return False

            # 替换entity_continue_extraction
            continue_start = content.find('PROMPTS["entity_continue_extraction"] = """')
            if continue_start == -1:
                continue_start = content.find(
                    "PROMPTS['entity_continue_extraction'] = '''"
                )
                quote_type_continue = "'''"
            else:
                quote_type_continue = '"""'

            if continue_start != -1:
                # 找到开始位置后的第一个三引号
                start_pos = content.find(quote_type_continue, continue_start) + len(
                    quote_type_continue
                )
                # 找到对应的结束三引号
                end_pos = content.find(quote_type_continue, start_pos)

                if start_pos != -1 and end_pos != -1:
                    # 替换entity_continue_extraction内容
                    new_content = (
                        content[:start_pos] + marked_entity_continue + content[end_pos:]
                    )
                    content = new_content
                    logging.info(
                        f"✓ 成功替换entity_continue_extraction提示词 (版本: {version_name})"
                    )
                else:
                    logging.error("✗ 无法找到entity_continue_extraction的结束位置")
                    return False
            else:
                logging.error("✗ 无法找到entity_continue_extraction的开始位置")
                return False

            # 写入修改后的内容
            return self.write_prompt_file(content)

        except Exception as e:
            logging.error(f"✗ 修改源码提示词时出错: {e}")
            return False

    def clean_working_dir(self):
        """彻底清理工作目录"""
        if os.path.exists(self.working_dir):
            try:
                # 删除整个工作目录
                shutil.rmtree(self.working_dir)
                logging.info(f"  - 已彻底删除工作目录: {self.working_dir}")

                # 重新创建空的工作目录
                os.makedirs(self.working_dir, exist_ok=True)
                logging.info(f"  - 已重新创建工作目录: {self.working_dir}")

            except Exception as e:
                logging.warning(f"  - 彻底清理工作目录失败: {e}")
                # 如果删除整个目录失败，回退到原来的逐文件删除方式
                files_to_clean = [
                    "graph_chunk_entity_relation.graphml",
                    "graph_data.json",
                    "kv_store_doc_status.json",
                    "kv_store_full_docs.json",
                    "kv_store_text_chunks.json",
                    "vdb_chunks.json",
                    "vdb_entities.json",
                    "vdb_relationships.json",
                    "kv_store_llm_response_cache.json",
                ]

                for filename in files_to_clean:
                    file_path = os.path.join(self.working_dir, filename)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logging.info(f"  - 已删除: {filename}")
                        except Exception as e:
                            logging.warning(f"  - 无法删除 {filename}: {e}")
        else:
            # 如果工作目录不存在，直接创建
            os.makedirs(self.working_dir, exist_ok=True)
            logging.info(f"  - 已创建工作目录: {self.working_dir}")

    def save_working_dir_to_tobe(self, version_name):
        """完整保存工作目录到tobe文件夹"""
        tobe_version_dir = os.path.join(self.tobe_dir, f"{version_name}_源码修改版")

        # 如果目标目录已存在，先删除
        if os.path.exists(tobe_version_dir):
            try:
                shutil.rmtree(tobe_version_dir)
                logging.info(f"  - 已删除已存在的目录: {tobe_version_dir}")
            except Exception as e:
                logging.warning(f"  - 删除已存在目录失败: {e}")

        try:
            # 确保工作目录存在
            if not os.path.exists(self.working_dir):
                logging.warning(f"  - 工作目录不存在: {self.working_dir}")
                return False

            # 完整复制工作目录
            shutil.copytree(self.working_dir, tobe_version_dir)

            # 验证复制结果
            copied_files = []
            for root, dirs, files in os.walk(tobe_version_dir):
                for file in files:
                    rel_path = os.path.relpath(
                        os.path.join(root, file), tobe_version_dir
                    )
                    copied_files.append(rel_path)

            if copied_files:
                logging.info(f"✓ 已完整保存工作目录到: {tobe_version_dir}")
                logging.info(f"  - 共保存 {len(copied_files)} 个文件:")
                for file in sorted(copied_files):
                    logging.info(f"    * {file}")
            else:
                logging.warning("  - 工作目录为空，未保存任何文件")

            return True

        except Exception as e:
            logging.error(f"✗ 保存工作目录失败: {e}")
            return False

    def run_insert_script(self, version_name):
        """运行insert.py脚本"""
        logging.info(f"🚀 开始运行insert.py (版本: {version_name})")

        try:
            # 使用Popen进行实时监控
            process = subprocess.Popen(
                [sys.executable, "insert.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            stdout_lines = []
            stderr_lines = []

            # 实时读取输出
            while True:
                stdout_line = process.stdout.readline()  # type: ignore
                if stdout_line:
                    stdout_lines.append(stdout_line)
                    # 检测完成标志
                    if "INSERT_COMPLETED_SUCCESSFULLY" in stdout_line:
                        logging.info("✓ 检测到插入完成信号")
                        process.terminate()  # 优雅终止进程
                        return True, "\n".join(stdout_lines)
                    elif "INSERT_FAILED" in stdout_line:
                        logging.error("✗ 检测到插入失败信号")
                        process.terminate()
                        return False, "\n".join(stdout_lines + stderr_lines)

                # 检查进程是否已经结束
                if process.poll() is not None:
                    # 进程已经结束，读取剩余输出
                    remaining_stdout, remaining_stderr = process.communicate()
                    if remaining_stdout:
                        stdout_lines.append(remaining_stdout)
                    if remaining_stderr:
                        stderr_lines.append(remaining_stderr)
                    break

            # 检查最终结果
            full_stdout = "\n".join(stdout_lines)
            full_stderr = "\n".join(stderr_lines)

            if "INSERT_COMPLETED_SUCCESSFULLY" in full_stdout:
                logging.info(f"✓ 版本 {version_name} 运行成功")
                return True, full_stdout
            elif process.returncode == 0:
                logging.info(f"✓ 版本 {version_name} 运行成功")
                return True, full_stdout
            else:
                logging.error(f"✗ 版本 {version_name} 运行失败")
                logging.error(f"错误输出: {full_stderr}")
                return False, full_stderr

        except Exception as e:
            logging.error(f"✗ 运行版本 {version_name} 时出错: {e}")
            return False, str(e)

    def load_prompt_versions(self):
        """加载提示词版本配置"""
        try:
            df = pd.read_excel(self.prompt_versions_file)
            return df.to_dict("records")
        except Exception as e:
            logging.error(f"✗ 加载提示词版本失败: {e}")
            return []

    def run_all_versions(self):
        """运行所有版本的测试"""
        # 备份原始文件
        if not self.backup_original_prompt():
            logging.error("无法备份原始文件，停止执行")
            return []

        # 加载版本配置
        prompt_versions = self.load_prompt_versions()
        if not prompt_versions:
            logging.error("没有找到提示词版本配置")
            return []

        results = []

        try:
            for i, version_info in enumerate(prompt_versions, 1):
                version_name = version_info.get("name", f"Version_{i}")
                entity_extraction = version_info.get("entity_extraction")
                entity_continue = version_info.get("entity_continue_extraction")

                if not entity_extraction or not entity_continue:
                    logging.warning(f"⚠️ 跳过版本 {version_name}，缺少提示词定义")
                    results.append(
                        {
                            "version_name": version_name,
                            "success": False,
                            "error": "Missing prompt definition",
                        }
                    )
                    continue

                logging.info(f"\n{'=' * 60}")
                logging.info(f"📝 处理版本 {i}/{len(prompt_versions)}: {version_name}")
                logging.info(f"{'=' * 60}")

                # 1. 修改源码
                logging.info("🔧 修改LightRAG源码中的提示词...")
                if not self.modify_source_prompts(
                    entity_extraction, entity_continue, version_name
                ):
                    logging.error(f"✗ 修改源码失败，跳过版本 {version_name}")
                    results.append(
                        {
                            "version_name": version_name,
                            "success": False,
                            "error": "Failed to modify source code",
                        }
                    )
                    continue

                # 2. 清理工作目录
                logging.info("🧹 清理工作目录...")
                self.clean_working_dir()

                # 3. 运行测试
                success, output = self.run_insert_script(version_name)

                # 4. 保存结果
                if success:
                    self.save_working_dir_to_tobe(version_name)

                results.append(
                    {
                        "version_name": version_name,
                        "success": success,
                        "output": output[:500] if output else "",  # 限制输出长度
                    }
                )

                logging.info(
                    f"📊 版本 {version_name} 测试完成: {'✅ 成功' if success else '❌ 失败'}"
                )

                # 等待一下，确保文件操作完成
                time.sleep(2)

        finally:
            # 恢复原始文件
            logging.info("🔄 恢复原始prompt.py文件...")
            self.restore_original_prompt()

        # 保存结果
        self.save_results(results)
        return results

    def save_results(self, results):
        """保存测试结果"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(self.results_file, index=False, encoding="utf-8")
            logging.info(f"💾 测试结果已保存到: {self.results_file}")
        except Exception as e:
            logging.error(f"✗ 保存结果失败: {e}")


def main():
    modifier = LightRAGSourceModifier()

    logging.info("🎯 开始LightRAG源码提示词修改测试")
    logging.info("📋 策略：直接修改源码 → 运行insert.py → 保存结果")

    results = modifier.run_all_versions()

    logging.info(f"\n{'=' * 60}")
    logging.info("🏁 所有测试完成！结果摘要:")
    logging.info(f"{'=' * 60}")

    if results:
        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)

        logging.info(f"📈 成功: {success_count}/{total_count}")

        for result in results:
            status = "✅ 成功" if result["success"] else "❌ 失败"
            logging.info(f"  {result['version_name']}: {status}")
            if not result["success"] and "error" in result:
                logging.info(f"    错误: {result['error']}")
    else:
        logging.info("❌ 没有执行任何测试")


if __name__ == "__main__":
    main()
