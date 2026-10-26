"""本地智能文档问答服务。

文档被解析、分块并存入 SQLite。查询时先检索相关片段，再由本机
Ollama 模型根据片段生成带来源的回答。整个流程不需要云端 API Key。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
SUPPORTED_TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".py",
    ".json",
    ".csv",
    ".html",
    ".xml",
}

load_dotenv(PROJECT_DIR / ".env")


class HashEmbedding:
    """无需外部服务的确定性文本向量。"""

    def __init__(self, dimensions: int = 128):
        if dimensions <= 0:
            raise ValueError("向量维度必须大于 0")
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> list[str]:
        ##把文本切分成：**连续英文 / 数字块，或者单个汉字**；符号、空格、标点全部丢弃
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            ##它用 md5 哈希把每个词（token）映射到 128 维空间里的一个位置，出现一次就加 1，最后做 L2 归一化
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0 else [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个已归一化向量的余弦相似度。"""
    return sum(a * b for a, b in zip(left, right))

##记忆工具持久化记忆库
class MemoryTool:
    """使用 SQLite 保存用户笔记和文档问答记录。"""

    def __init__(
        self,
        user_id: str = "default_user",
        database_path: str | Path | None = None,
    ):
        self.user_id = user_id.strip() or "default_user"
        self.database_path = Path(
            database_path
            or PROJECT_DIR / f"memory_{self._safe_name(self.user_id)}.db"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        ##查询返回 `Row` 对象，可以用 `row["colname"]` 按列名读取。
        self.connection.row_factory = sqlite3.Row
        ##同一个线程可以多次 `acquire()` 不会死锁（普通 Lock 不行）
        ##所有读写数据库的操作，都要包上
        ##with self._lock:
            ##cur = self.connection.execute(...)
        self._lock = RLock()
        with self.connection:
##INTEGER PRIMARY KEY AUTOINCREMENT
##整数类型主键，**自增 ID**，插入记录时不指定这个字段，SQLite 自动分配一个递增的整数
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_created
                ON memories(user_id, created_at DESC)
                """
            )

    @staticmethod
##把 `value` 里面**所有不是字母、数字、`_`、`-` 的连续字符，统一替换成单个下划线**；如果处理完是空字符串，就返回 `default_user`
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", value) or "default_user"

    def execute(self, action: str, **kwargs: Any) -> str:
        actions = {
            "add": self._add,
            "search": self._search,
            "summary": self._summary,
            "stats": self._stats,
            "important": self._important,
            "clear_all": self._clear_all,
        }
        handler = actions.get(action)
        if handler is None:
            return f"不支持的记忆操作：{action}"
        return handler(**kwargs)

    def run(self, parameters: dict[str, Any] | str, **kwargs: Any) -> str:
        if isinstance(parameters, dict):
            values = dict(parameters)
            action = str(values.pop("action", "summary"))
            return self.execute(action, **values, **kwargs)
        return self.execute(parameters, **kwargs)

    def _add(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        **metadata: Any,
    ) -> str:
        clean_content = content.strip()
        clean_type = memory_type.strip() or "working"
        if not clean_content:
            return "记忆内容不能为空。"
        score = float(importance)
        if not 0 <= score <= 1:
            return "记忆重要性必须在 0 到 1 之间。"
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO memories
                    (user_id, content, memory_type, importance, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.user_id,
                    clean_content,
                    clean_type,
                    score,
                    datetime.now().astimezone().isoformat(),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        return "记忆已保存。"

    def _search(self, query: str = "", limit: int = 5, **kwargs: Any) -> str:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT memory_type, content, importance
                FROM memories
                WHERE user_id = ? AND content LIKE ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
                """,
                (self.user_id, f"%{query.strip()}%", max(1, int(limit))),
            ).fetchall()
        if not rows:
            return "没有找到相关记忆。"
        return "\n".join(
            f"[{row['memory_type']}] {row['content']} "
            f"(重要性={row['importance']:.1f})"
            for row in rows
        )

    def _summary(self, limit: int = 10, **kwargs: Any) -> str:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT memory_type, content
                FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (self.user_id, max(1, int(limit))),
            ).fetchall()
        if not rows:
            return "当前没有记忆。"
        return "\n".join(
            f"[{row['memory_type']}] {row['content']}" for row in rows
        )

    def _stats(self, **kwargs: Any) -> str:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT memory_type, COUNT(*) AS total
                FROM memories
                WHERE user_id = ?
                GROUP BY memory_type
                """,
                (self.user_id,),
            ).fetchall()
        return json.dumps(
            {row["memory_type"]: int(row["total"]) for row in rows},
            ensure_ascii=False,
        )

    def _important(self, limit: int = 2, **kwargs: Any) -> str:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT memory_type, content, importance
                FROM memories
                WHERE user_id = ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
                """,
                (self.user_id, max(1, int(limit))),
            ).fetchall()
        if not rows:
            return "当前没有记忆。"
        return "\n".join(
            f"[{row['memory_type']}] {row['content']} "
            f"(重要性={row['importance']:.1f})"
            for row in rows
        )

    def _clear_all(self, **kwargs: Any) -> str:
        with self._lock, self.connection:
            self.connection.execute(
                "DELETE FROM memories WHERE user_id = ?",
                (self.user_id,),
            )
        return "当前用户的记忆已清空。"

    def close(self) -> None:
        with self._lock:
            self.connection.close()


class RAGTool:
    """文档解析、分块、向量存储、检索与问答工具。"""

    def __init__(
        self,
        knowledge_base_path: str | Path = KNOWLEDGE_BASE_DIR,
        rag_namespace: str = "default",
        llm: HelloAgentsLLM | None = None,
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.rag_namespace = rag_namespace.strip() or "default"
        self.database_path = (
            self.knowledge_base_path / f"{self._safe_name(self.rag_namespace)}.db"
        )
        self.connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.embedding = HashEmbedding()
        self.llm = llm
        self._lock = RLock()
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_document
                ON rag_chunks(document_id, chunk_index)
                """
            )

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", value) or "default"

    def execute(self, action: str, **kwargs: Any) -> Any:
        actions = {
            "add_document": self._add_document,
            "add_text": self._add_text,
            "search": self._search_action,
            "ask": self._ask,
            "stats": self._stats,
            "list_documents": self._list_documents,
            "delete_document": self._delete_document,
            "clear": self._clear,
        }
        handler = actions.get(action)
        if handler is None:
            return {"success": False, "error": f"不支持的 RAG 操作：{action}"}
        return handler(**kwargs)

    def run(self, parameters: dict[str, Any] | str, **kwargs: Any) -> Any:
        if isinstance(parameters, dict):
            values = dict(parameters)
            action = str(values.pop("action", "search"))
            return self.execute(action, **values, **kwargs)
        return self.execute(parameters, **kwargs)

    @staticmethod
    def _convert_to_markdown(path: Path) -> str:
        ##获取文件**后缀名（扩展名）**
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT_SUFFIXES:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("读取 PDF 需要安装 pypdf。") from exc
            reader = PdfReader(str(path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise RuntimeError(f"暂不支持文档格式：{suffix or '未知格式'}") from exc
        result = MarkItDown().convert(str(path))
        return getattr(result, "text_content", "") or ""

    @staticmethod
    def _split_markdown(
        text: str,
        chunk_size: int,##单个块最大字符长度
        chunk_overlap: int,##块之间重叠字符数
##滑动步长：比如 chunk_size=500，overlap=100 → step=400；
##0~500，400~900，800~1300…… 每两块之间尾部重叠 100 字符。
    ) -> list[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")
        ##Windows 换行转 Unix 换行，统一分割
        lines = text.replace("\r\n", "\n").split("\n")
        ##存放【绑定好标题路径】的完整段落单元（还没做滑动切分）
        ##在进行切块前进行转Markdown语义段落单元**按标题、空行天然分割出来**，优先保证语义完整
        blocks: list[str] = []
        ##`heading_stack`：**标题栈，维护当前大纲路径**
        heading_stack: list[str] = []
        ##缓存当前正在收集的普通正文行（非标题、非空行）
        paragraph: list[str] = []
##把积攒在 `paragraph` 的正文打包成一个 block，带上当前标题栈作为前缀
##拼完清空 paragraph，准备收下一段文字
##- 触发 flush 的时机：
##1. 遇到新标题 2. 遇到空行（markdown 段落分隔）3. 全文循环结束最后兜底 flush
        def flush() -> None:
            if not paragraph:
                return
            body = " ".join(line.strip() for line in paragraph).strip()
            if body:
                prefix = " > ".join(heading_stack)
                blocks.append(f"{prefix}\n{body}" if prefix else body)
            paragraph.clear()

        for line in lines:
            ##匹配标准 ATX 标题 `# 标题` ~ `###### 标题`
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
##- group1：`###` → `len=3` → level=3（三级标题）
##- group2：标题文字

            if heading:
                flush()
##新标题是第几级，栈就只保留前面「level-1 个标题」，然后把当前标题放进去
##为什么要这样？ prefix = " > ".join(heading_stack)
##正文会带上完整标题链。 比如在 `### C` 下面的段落，prefix = `"A > B > C"`
##检索的时候，这段文字就知道它属于哪一层大纲
                level = len(heading.group(1))
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading.group(2))
            elif line.strip():
                paragraph.append(line)
            else:
                flush()
        flush()
##if not blocks场景：全文没有任何 `#` 标题，也没有换行分段，解析完 blocks 为空。
##把所有空白（换行、多个空格）压缩成单个空格，变成一整块文本。
        if not blocks:
            normalized = re.sub(r"\s+", " ", text).strip()
            blocks = [normalized] if normalized else []
##滑动窗口切分 block → 最终 chunks
#举个例子
# 大模型
## RAG 检索增强生成。 可以提升问答准确性。

## 微调 微调需要数据集。
# 大模型` → flush 空，stack:["大模型"]
## RAG` → flush 空，stack:["大模型", "RAG"]
#得到 blocks = ["大模型 > RAG\n 检索增强生成。可以提升问答准确性。","大模型 > 微调 \n 微调需要数据集。"]
#这里就对每一个bolck进行切分切块 长度 ≤ chunk_size：直接作为一个 chunk
        chunks: list[str] = []
        step = chunk_size - chunk_overlap
        for block in blocks:
            if len(block) <= chunk_size:
                chunks.append(block)
                continue
            for start in range(0, len(block), step):
                piece = block[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
        return chunks
#前端加载document
    def _add_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        document_id: str | None = None,
        source_label: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.is_file():
            return {"success": False, "error": f"文件不存在：{path}"}
        try:
            text = self._convert_to_markdown(path)
            chunks = self._split_markdown(text, chunk_size, chunk_overlap)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        if not chunks:
            return {
                "success": False,
                "error": "文档中没有可提取的文本；扫描版 PDF 需要先进行 OCR。",
            }
        return self._store_chunks(
            document_id=document_id or path.name,
            source=source_label or str(path),
            chunks=chunks,
        )
#前端加载text
    def _add_text(
        self,
        text: str,
        document_id: str = "text_document",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        source_label: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        clean_id = document_id.strip()
        if not clean_id:
            return {"success": False, "error": "文档标识不能为空"}
        chunks = self._split_markdown(text, chunk_size, chunk_overlap)
        if not chunks:
            return {"success": False, "error": "文档内容不能为空"}
        return self._store_chunks(
            document_id=clean_id,
            source=source_label or clean_id,
            chunks=chunks,
        )

    def _store_chunks(
        self,
        document_id: str,
        source: str,
        chunks: list[str],
    ) -> dict[str, Any]:
        created_at = datetime.now().astimezone().isoformat()
        rows = [
            (
                document_id,
                source,
                index,
                chunk,
                json.dumps(self.embedding.embed(chunk)),
                created_at,
            )
            for index, chunk in enumerate(chunks)
        ]
        ##先按 document_id 删除旧块，再批量插入新块。所以同名文档重新上传会覆盖。
        with self._lock, self.connection:
            self.connection.execute(
                "DELETE FROM rag_chunks WHERE document_id = ?",
                (document_id,),
            )
            self.connection.executemany(
                """
                INSERT INTO rag_chunks
                    (document_id, source, chunk_index, content, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return {
            "success": True,
            "document_id": document_id,
            "chunks": len(chunks),
            "message": f"已导入文档，共 {len(chunks)} 个文本块。",
        }

    def _search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        ##调用 embedding 模型，把用户输入文本 `query` 转换成向量
        query_vector = self.embedding.embed(query)
        with self._lock:
            rows = self.connection.execute("SELECT * FROM rag_chunks").fetchall()
        scored = [
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "source": row["source"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "score": cosine_similarity(
                    query_vector,
                    json.loads(row["embedding"]),
                ),
            }
            for row in rows
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        ##**把库里面所有 chunk 全部查出来，在 Python 内存里逐个算余弦相似度，按相似度从高到低排序，返回 top limit 条**。
        return scored[: max(1, int(limit))]
##**程序内部调用**：你自己拿到结果后，可以自定义渲染、前端展示、自己写 prompt、自定义过滤阈值
    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        clean_query = query.strip()
        return self._search(clean_query, limit) if clean_query else []

    def _search_action(self, query: str, limit: int = 5, **kwargs: Any) -> str:
        results = self.search(query, limit)
        if not results:
            return "知识库中没有找到相关内容。"
        return self._format_context(results)

    @staticmethod
    def _format_context(results: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{index}] 来源：{item['source']}，片段：{item['chunk_index'] + 1}，"
            f"相似度：{item['score']:.3f}\n{item['content']}"
            for index, item in enumerate(results, start=1)
        )

    def _llm_text(self, prompt: str, system: str) -> str:
        if self.llm is None:
            raise RuntimeError("RAGTool 未配置本地 LLM")
        answer = self.llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        if not answer or not answer.strip():
            raise RuntimeError("本地模型没有返回有效内容")
        return answer.strip()
    
##MQE（多查询扩展）：解决用户提问太短（“杭州怎么玩”）导致召回不足的问题
    def _generate_multi_queries(self, question: str) -> list[str]:
        if self.llm is None:
            return [question]
        try:
            answer = self._llm_text(
                "请把问题改写成最多 3 个适合文档检索的查询。"
                "每行只输出一个查询。\n问题：" + question,
                "你是检索查询改写器。",
            )
        except Exception:
            return [question]
        queries = [line.strip(" -•") for line in answer.splitlines() if line.strip()]
        return list(dict.fromkeys([question, *queries]))[:4]
##HyDE（假设文档嵌入）：解决 query 和文档用词不一致（用户说“住宿”，文档里叫“酒店”）的问题。
# 让 LLM 先“脑补”一段答案，用假答案去搜，命中率往往更高
    def _generate_hyde(self, question: str) -> str:
        if self.llm is None:
            return ""
        try:
            return self._llm_text(
                "写一段简短的假设性答案，用于检索相关文档。\n问题：" + question,
                "你是文档检索助手。",
            )
        except Exception:
            return ""

    def _ask(
        self,
        question: str,
        limit: int = 5,
        enable_advanced_search: bool = False,
        enable_mqe: bool = True,
        enable_hyde: bool = True,
        **kwargs: Any,
    ) -> str:
        clean_question = question.strip()
        if not clean_question:
            return "问题不能为空。"
        if not self._list_documents():
            return "知识库为空，请先导入文档。"

        queries = [clean_question]
        if enable_advanced_search and enable_mqe:
            queries = self._generate_multi_queries(clean_question)
        if enable_advanced_search and enable_hyde:
            hyde = self._generate_hyde(clean_question)
            if hyde:
                queries.append(hyde)
##- 字典 `merged` 的 key = chunk 的数据库主键 `id`，用来**去重**：同一个 chunk，多个 query 都搜到了，只保留一条
##- 规则：如果同一个 chunk 被多次召回，**保留最高的那个相似度 score**
##- 每个 query 检索时取至少 5 条，保证召回充足，后面再统一筛选
        merged: dict[int, dict[str, Any]] = {}
        for query in queries:
            for item in self._search(query, limit=max(int(limit), 5)):
                if item["id"] not in merged or item["score"] > merged[item["id"]]["score"]:
                    merged[item["id"]] = item
        results = sorted(
            merged.values(),
            key=lambda item: item["score"],
            reverse=True,
        )[: max(1, int(limit))]
        if not results:
            return "知识库中没有找到相关内容。"

        context = self._format_context(results)
        return self._llm_text(
            f"用户问题：\n{clean_question}\n\n参考资料：\n{context}\n\n"
            "只依据参考资料回答。资料不足时明确说明。"
            "回答末尾列出使用的来源和片段编号。",
            "你是严谨的本地文档问答助手，不得编造资料中不存在的事实。",
        )

    def _stats(self, **kwargs: Any) -> str:
        with self._lock:
            chunks = self.connection.execute(
                "SELECT COUNT(*) AS total FROM rag_chunks"
            ).fetchone()
            documents = self.connection.execute(
                "SELECT COUNT(DISTINCT document_id) AS total FROM rag_chunks"
            ).fetchone()
        return json.dumps(
            {
                "namespace": self.rag_namespace,
                "documents": int(documents["total"]),
                "chunks": int(chunks["total"]),
            },
            ensure_ascii=False,
        )

    def _list_documents(self, **kwargs: Any) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT document_id, source, COUNT(*) AS chunks,
                       MAX(created_at) AS created_at
                FROM rag_chunks
                GROUP BY document_id, source
                ORDER BY created_at DESC, document_id
                """
            ).fetchall()
        return [
            {
                "document_id": row["document_id"],
                "source": row["source"],
                "chunks": int(row["chunks"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _delete_document(self, document_id: str, **kwargs: Any) -> dict[str, Any]:
        clean_id = document_id.strip()
        if not clean_id:
            return {"success": False, "error": "文档标识不能为空"}
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "DELETE FROM rag_chunks WHERE document_id = ?",
                (clean_id,),
            )
        if cursor.rowcount == 0:
            return {"success": False, "error": f"没有找到文档：{clean_id}"}
        return {"success": True, "document_id": clean_id}

    def _clear(self, **kwargs: Any) -> str:
        with self._lock, self.connection:
            self.connection.execute("DELETE FROM rag_chunks")
        return "知识库已清空。"

    def close(self) -> None:
        with self._lock:
            self.connection.close()

##把 MemoryTool 和 RAGTool 组合起来，给外部提供一个统一的 API
class DocumentAssistant:
    """组合本地 LLM、RAG 和持久化记忆的文档问答服务。"""

    def __init__(
        self,
        user_id: str = "default_user",
        llm: HelloAgentsLLM | None = None,
        knowledge_base_path: str | Path = KNOWLEDGE_BASE_DIR,
        memory_database_path: str | Path | None = None,
    ):
        self.user_id = user_id.strip() or "default_user"
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.llm = llm or create_llm()
        self.memory_tool = MemoryTool(
            user_id=self.user_id,
            database_path=memory_database_path,
        )
        self.rag_tool = RAGTool(
            knowledge_base_path=knowledge_base_path,
            rag_namespace=f"pdf_{self.user_id}",
            llm=self.llm,
        )
        self.session_started_at = datetime.now().astimezone()
        self.documents_loaded = 0
        self.questions_asked = 0
        self.notes_saved = 0
##导入文件，同时往记忆库写一条“导入了文档”的事件
    def load_document(self, file_path: str) -> dict[str, Any]:
        start = time.perf_counter()
        result = self.rag_tool.execute("add_document", file_path=file_path)
        if not result.get("success"):
            return {
                "success": False,
                "message": f"加载失败：{result.get('error', '未知错误')}",
            }
        self.documents_loaded += 1
        self.memory_tool.execute(
            "add",
            content=f"导入文档《{result['document_id']}》",
            memory_type="episodic",
            importance=0.8,
            session_id=self.session_id,
        )
        return {
            "success": True,
            "message": f"加载成功，耗时 {time.perf_counter() - start:.1f} 秒。",
            "document": result["document_id"],
            "chunks": result["chunks"],
        }
##粘贴文本导入
    def add_text_document(self, title: str, content: str) -> dict[str, Any]:
        clean_title = title.strip()
        if not clean_title:
            return {"success": False, "message": "文档标题不能为空。"}
        result = self.rag_tool.execute(
            "add_text",
            text=content,
            document_id=clean_title,
            source_label=clean_title,
        )
        if not result.get("success"):
            return {
                "success": False,
                "message": f"保存失败：{result.get('error', '未知错误')}",
            }
        self.documents_loaded += 1
        return {
            "success": True,
            "message": result["message"],
            "document": result["document_id"],
            "chunks": result["chunks"],
        }
##提问，并记录“问了什么问题”。
    def ask(self, question: str, use_advanced_search: bool = False) -> str:
        clean_question = question.strip()
        if not clean_question:
            return "问题不能为空。"
        answer = self.rag_tool.execute(
            "ask",
            question=clean_question,
            limit=5,
            enable_advanced_search=use_advanced_search,
            enable_mqe=use_advanced_search,
            enable_hyde=use_advanced_search,
        )
        self.questions_asked += 1
        self.memory_tool.execute(
            "add",
            content=f"提问：{clean_question}",
            memory_type="working",
            importance=0.6,
            session_id=self.session_id,
        )
        return str(answer)
##写学习笔记
    def add_note(self, content: str, concept: str | None = None) -> str:
        result = self.memory_tool.execute(
            "add",
            content=content,
            memory_type="semantic",
            importance=0.8,
            concept=(concept or "general").strip(),
            session_id=self.session_id,
        )
        if result == "记忆已保存。":
            self.notes_saved += 1
        return result

    def recall(self, query: str, limit: int = 5) -> str:
        return self.memory_tool.execute("search", query=query, limit=limit)

    def list_documents(self) -> list[dict[str, Any]]:
        return self.rag_tool.execute("list_documents")

    def delete_document(self, document_id: str) -> dict[str, Any]:
        return self.rag_tool.execute("delete_document", document_id=document_id)
##生成运行统计
    def get_stats(self) -> dict[str, Any]:
        duration = datetime.now().astimezone() - self.session_started_at
        rag_stats = json.loads(self.rag_tool.execute("stats"))
        return {
            "session_id": self.session_id,
            "duration_seconds": round(duration.total_seconds()),
            "documents_loaded_this_session": self.documents_loaded,
            "questions_asked": self.questions_asked,
            "notes_saved": self.notes_saved,
            "knowledge_base": rag_stats,
            "model": self.llm.model,
            "base_url": self.llm.base_url,
        }
##生成JSON 报告
    def generate_report(self, save_to_file: bool = True) -> dict[str, Any]:
        report = {
            "session": self.get_stats(),
            "memory_summary": self.memory_tool.execute("summary", limit=10),
            "documents": self.list_documents(),
        }
        if save_to_file:
            report_file = PROJECT_DIR / f"document_report_{self.session_id}.json"
            report_file.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report["report_file"] = str(report_file)
        return report

    def close(self) -> None:
        self.memory_tool.close()
        self.rag_tool.close()


PDFLearningAssistant = DocumentAssistant


def create_llm(
    model: str | None = None,
    base_url: str | None = None,
) -> HelloAgentsLLM:
    """创建不需要云端密钥的本地 Ollama 客户端。"""
    return HelloAgentsLLM(
        provider="ollama",
        model=model
        or os.getenv("OLLAMA_MODEL")
        or os.getenv("LLM_MODEL_ID")
        or DEFAULT_OLLAMA_MODEL,
        base_url=base_url
        or os.getenv("OLLAMA_HOST")
        or os.getenv("LLM_BASE_URL")
        or DEFAULT_OLLAMA_URL,
        timeout=int(os.getenv("LLM_TIMEOUT", "180")),
    )

##web ui
def create_gradio_ui(assistant: DocumentAssistant):
    """创建文档导入、问答和记忆管理页面。"""
    import gradio as gr

    def document_rows() -> list[list[Any]]:
        return [
            [
                item["document_id"],
                item["source"],
                item["chunks"],
                item["created_at"],
            ]
            for item in assistant.list_documents()
        ]

    def load_file(file_path: str | None):
        if not file_path:
            raise gr.Error("请先选择文档。")
        result = assistant.load_document(str(file_path))
        if not result["success"]:
            raise gr.Error(result["message"])
        return result["message"], document_rows(), None

    def save_text(title: str, content: str):
        result = assistant.add_text_document(title, content)
        if not result["success"]:
            raise gr.Error(result["message"])
        return result["message"], document_rows(), "", ""

    def chat(
        message: str,
        history: list[dict[str, str]] | None,
        advanced_search: bool,
    ):
        if not message.strip():
            raise gr.Error("问题不能为空。")
        response = assistant.ask(message, use_advanced_search=advanced_search)
        updated = [
            *(history or []),
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": response},
        ]
        return "", updated

    def save_note(content: str, concept: str):
        if not content.strip():
            raise gr.Error("笔记不能为空。")
        return assistant.add_note(content, concept or None), "", ""

    def stats_text() -> str:
        return json.dumps(assistant.get_stats(), ensure_ascii=False, indent=2)

    def report_text() -> str:
        return json.dumps(
            assistant.generate_report(),
            ensure_ascii=False,
            indent=2,
        )

    with gr.Blocks(title="本地智能文档问答") as app:
        gr.Markdown("# 本地智能文档问答")
        gr.Markdown(
            f"当前模型：`{assistant.llm.model}` · 本地服务：`{assistant.llm.base_url}`"
        )
        with gr.Tab("知识库"):
            with gr.Row():
                file_input = gr.File(
                    label="上传文档",
                    type="filepath",
                    file_types=[".pdf", ".md", ".markdown", ".txt"],
                )
                with gr.Column():
                    load_button = gr.Button("导入文档", variant="primary")
                    load_output = gr.Textbox(label="导入状态", interactive=False)
            with gr.Row():
                text_title = gr.Textbox(label="文本标题")
                text_content = gr.Textbox(label="文本内容", lines=6)
            save_text_button = gr.Button("保存文本")
            document_table = gr.Dataframe(
                headers=["文档", "来源", "文本块", "导入时间"],
                value=document_rows(),
                interactive=False,
                wrap=True,
                label="知识库文档",
            )
        with gr.Tab("问答"):
            chatbot = gr.Chatbot(label="文档问答", height=500)
            advanced_search = gr.Checkbox(
                label="使用多查询和 HyDE 增强检索",
                value=False,
            )
            message = gr.Textbox(label="问题", lines=2)
            with gr.Row():
                send = gr.Button("发送", variant="primary")
                clear_chat = gr.Button("清空对话")
        with gr.Tab("记忆与统计"):
            note = gr.Textbox(label="笔记", lines=3)
            concept = gr.Textbox(label="主题")
            save_note_button = gr.Button("保存笔记")
            note_output = gr.Textbox(label="保存状态", interactive=False)
            with gr.Row():
                stats_button = gr.Button("查看统计")
                report_button = gr.Button("生成报告")
            report_output = gr.Code(label="结果", language="json")

        load_button.click(
            load_file,
            inputs=file_input,
            outputs=[load_output, document_table, file_input],
        )
        save_text_button.click(
            save_text,
            inputs=[text_title, text_content],
            outputs=[load_output, document_table, text_title, text_content],
        )
        send.click(
            chat,
            inputs=[message, chatbot, advanced_search],
            outputs=[message, chatbot],
        )
        message.submit(
            chat,
            inputs=[message, chatbot, advanced_search],
            outputs=[message, chatbot],
        )
        clear_chat.click(lambda: ("", []), outputs=[message, chatbot])
        save_note_button.click(
            save_note,
            inputs=[note, concept],
            outputs=[note_output, note, concept],
        )
        stats_button.click(stats_text, outputs=report_output)
        report_button.click(report_text, outputs=report_output)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="本地智能文档问答")
    parser.add_argument("--document", help="要导入的 PDF、Markdown 或 TXT 文件")
    parser.add_argument("--question", help="导入文档后要询问的问题")
    parser.add_argument("--user-id", default="default_user", help="用户数据隔离标识")
    parser.add_argument("--advanced-search", action="store_true", help="启用 MQE 和 HyDE")
    parser.add_argument("--web", action="store_true", help="启动 Web 页面")
    parser.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=7860, help="Web 服务端口")
    parser.add_argument("--model", help="Ollama 模型名称")
    parser.add_argument("--base-url", help="Ollama OpenAI 兼容接口地址")
    args = parser.parse_args()

    assistant = DocumentAssistant(
        user_id=args.user_id,
        llm=create_llm(model=args.model, base_url=args.base_url),
    )
    try:
        if args.document:
            result = assistant.load_document(args.document)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.web:
            create_gradio_ui(assistant).queue(default_concurrency_limit=1).launch(
                server_name=args.host,
                server_port=args.port,
                share=False,
            )
            return
        if args.question:
            print(
                assistant.ask(
                    args.question,
                    use_advanced_search=args.advanced_search,
                )
            )
        elif not args.document:
            parser.print_help()
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
