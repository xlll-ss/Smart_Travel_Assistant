"""检查智能旅行助手的运行环境、存储和本地 Ollama 配置。"""

from __future__ import annotations

import argparse
import importlib
import os
import py_compile
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


class CheckReport:
    def __init__(self) -> None:
        self.failed = 0
        self.warnings = 0

    def ok(self, name: str, detail: str) -> None:
        print(f"[通过] {name}: {detail}")

    def warn(self, name: str, detail: str) -> None:
        self.warnings += 1
        print(f"[提示] {name}: {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.failed += 1
        print(f"[失败] {name}: {detail}")


def run_check(
    report: CheckReport,
    name: str,
    function: Callable[[], str],
    required: bool = True,
) -> None:
    try:
        report.ok(name, function())
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        report.fail(name, detail) if required else report.warn(name, detail)


def check_python() -> str:
    if sys.version_info < (3, 10):
        raise RuntimeError("需要 Python 3.10 或更高版本")
    return f"Python {sys.version.split()[0]}"


def check_imports() -> str:
    for module_name in (
        "hello_agents",
        "openai",
        "dotenv",
        "gradio",
        "pypdf",
        "docling.document_converter",
        "sentence_transformers",
    ):
        importlib.import_module(module_name)
    return "核心依赖均可导入"


def check_source_compile() -> str:
    source_paths = [
        PROJECT_DIR / "travel_assistant.py",
        PROJECT_DIR / "travel_evaluation.py",
        PROJECT_DIR / "wise_douctment.py",
    ]
    source_paths.extend((PROJECT_DIR / "hello_agents").rglob("*.py"))
    for path in source_paths:
        py_compile.compile(str(path), doraise=True)
    return f"{len(source_paths)} 个 Python 源文件语法检查通过"


def check_configuration() -> str:
    model = os.getenv("LLM_MODEL_ID")
    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    if not model:
        raise RuntimeError(".env 中缺少 LLM_MODEL_ID")
    return f"模型={model}，base_url={base_url}，本地 Ollama 不需要 API Key"


def check_llm_initialization() -> str:
    from travel_assistant import create_llm

    llm = create_llm()
    if llm.provider != "ollama":
        raise RuntimeError(f"预期 provider=ollama，实际为 {llm.provider}")
    return f"HelloAgentsLLM 初始化成功，模型={llm.model}"


def check_embedding_model() -> str:
    from wise_douctment import BGEEmbedding

    embedding = BGEEmbedding()
    query_vector = embedding.embed_query("推荐杭州安静的室内地点")
    document_vector = embedding.embed_documents(["杭州有适合安静阅读的书店。"])[0]
    if not query_vector or len(query_vector) != len(document_vector):
        raise RuntimeError("查询向量和文档向量维度不一致")
    return (
        f"模型={embedding.model_id}，设备={embedding.device}，"
        f"维度={len(query_vector)}"
    )


def check_sqlite_databases() -> str:
    database_paths = [PROJECT_DIR / "travel_assistant.db"]
    database_paths.extend((PROJECT_DIR / "knowledge_base").rglob("*.db"))
    checked = []
    for path in database_paths:
        if not path.exists():
            continue
        connection = sqlite3.connect(path)
        try:
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise RuntimeError(f"{path.name} 完整性检查失败：{result}")
        finally:
            connection.close()
        checked.append(str(path.relative_to(PROJECT_DIR)))
    if not checked:
        return "尚无运行数据库，首次启动时会自动创建"
    return "数据库完整性正常：" + ", ".join(checked)


def check_storage_isolation() -> str:
    from travel_assistant import (
        TravelPreferenceStore,
        TravelPreferences,
        UserKnowledgeBaseManager,
    )

    class CheckEmbedding:
        model_id = "environment-check:v1"

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    with tempfile.TemporaryDirectory(prefix="travel_check_") as temp_dir:
        root = Path(temp_dir)
        store = TravelPreferenceStore(root / "travel.db")
        preferences = TravelPreferences(
            city="杭州",
            mood="想安静",
            activity_preference="室内为主",
            budget="经济",
            duration_hours=4,
            companions="独自出行",
            departure="市中心",
        )
        store.save("user-a", preferences)
        if store.load("user-b") is not None:
            raise RuntimeError("偏好数据发生跨用户读取")

        manager = UserKnowledgeBaseManager(
            root / "users",
            embedding=CheckEmbedding(),
        )
        try:
            user_a = manager.for_user("user-a")
            user_b = manager.for_user("user-b")
            user_a.add_guide("检查攻略", "这是一条用户隔离检查资料。")
            if user_b.list_documents():
                raise RuntimeError("私人知识库发生跨用户读取")
        finally:
            manager.close_all()
    return "偏好和私人知识库用户隔离正常"


def check_live_llm() -> str:
    from travel_assistant import create_llm

    answer = create_llm().invoke(
        [
            {"role": "system", "content": "只回复 TRAVEL_CHECK_OK"},
            {"role": "user", "content": "执行连通性检查"},
        ],
        temperature=0,
        max_tokens=30,
    )
    if "TRAVEL_CHECK_OK" not in answer:
        raise RuntimeError(f"模型返回内容不符合预期：{answer[:100]}")
    return "Ollama 调用成功"


def main() -> int:
    parser = argparse.ArgumentParser(description="智能旅行助手环境检查")
    parser.add_argument("--live", action="store_true", help="额外调用一次本地 Ollama")
    args = parser.parse_args()

    report = CheckReport()
    print("智能旅行助手环境检查")
    print(f"项目目录：{PROJECT_DIR}")
    print("=" * 68)
    run_check(report, "Python", check_python)
    run_check(report, "依赖", check_imports)
    run_check(report, "源码", check_source_compile)
    run_check(report, "模型配置", check_configuration)
    run_check(report, "LLM 客户端", check_llm_initialization)
    run_check(report, "BGE 向量模型", check_embedding_model)
    run_check(report, "SQLite", check_sqlite_databases)
    run_check(report, "用户隔离", check_storage_isolation)
    if args.live:
        run_check(report, "真实模型", check_live_llm)
    print("=" * 68)
    print(f"检查完成：失败 {report.failed} 项，提示 {report.warnings} 项")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
