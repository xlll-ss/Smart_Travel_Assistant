"""本地文档知识库：结构解析、BGE 向量、SQLite RAG 和记忆工具。

旅行应用统一从此模块导入。PDF 保留物理页码、章节和表格，查询时传回出处；
独立运行仍支持文档问答 CLI/Web，通过本地 Ollama 生成回答，无需云端密钥。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Protocol

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM

PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
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

# ------------------------- 文档结构与切块 -------------------------

#PDFBlockParser所以在这类函数里面逻辑是判断是不是需要用到Docling解析树（懒加载当真正上传了PDF时调用）然后把前端接受一个PDF文件路劲 
#然后给Docling解析为一个 Docling 自己极其复杂的 Document 树（里面包含了几十种不同的节点类型、坐标、属性）
#这个对象树里面已经帮我们分析好了多级标题和正文，只需要我们进行通过栈判断转化为1->2->3这种，最后返回一个带有标题页码正文等等的一个列表

##这是整个系统内部流通的“标准货物”。无论进来的是 PDF 还是 Markdown，第一步都要切成这样的小块。
@dataclass(frozen=True)#返回一个带有 正文 页码 表格的字典
class ContentBlock:
    """正文和出处一起传递，防止切块后无法定位原文。"""

    content: str  #正文纯文本
    page_numbers: tuple[int, ...] = ()  # 多页原块保留页集合。
    section: tuple[str, ...] = () #属于哪个大纲（如 ("第一章", "1.1节")）
    block_type: str = "paragraph"
    block_id: str = ""
    table: dict[str, Any] | None = None  # 表格行、表头、单元格和出处。
    parser: str = "text"
##把除了正文以外的信息打包成字典，方便后续存进数据库
    def metadata(self) -> dict[str, Any]:
        return {
            "page_numbers": list(self.page_numbers),
            "section": list(self.section),
            "block_type": self.block_type,
            "block_id": self.block_id,
            "table": self.table,
            "parser": self.parser,
        }

class PDFBlockParser:
    """全进程共享一个懒加载转换器，并串行解析以控制模型内存和并发。"""

    def __init__(self) -> None:
        self._converter: Any = None
        self._lock = RLock()
##懒加载：只有真的上传了 PDF，才会去 import 庞大的 Docling 库。
    def _get_converter(self) -> Any:
        if self._converter is None:
            try:
                from docling.datamodel.accelerator_options import (
                    AcceleratorDevice,
                    AcceleratorOptions,
                )
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    PdfPipelineOptions,
                    RapidOcrOptions,
                )
                from docling.document_converter import DocumentConverter, PdfFormatOption
            except ImportError as exc:
                raise RuntimeError(
                    "结构化 PDF 解析需要 Docling，请执行 python -m pip install -r requirements.txt"
                ) from exc

            options = PdfPipelineOptions()
            # 版面识别保留表格结构；扫描件按需启用 OCR，避免文本 PDF 额外开销。
            options.do_table_structure = True
            options.do_ocr = os.getenv("PDF_ENABLE_OCR", "false").lower() in {
                "1",
                "true",
                "yes",
            }
            if options.do_ocr:
                options.ocr_options = RapidOcrOptions(backend="onnxruntime")
            # CPU 默认适配当前机器；PDF 解析设备与 Ollama/BGE 独立配置。
            options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice(os.getenv("PDF_DEVICE", "cpu")),
            )
            artifacts = os.getenv("PDF_ARTIFACTS_PATH")
            if artifacts:
                options.artifacts_path = Path(artifacts)
            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
            )
        return self._converter
##接收一个 PDF 文件路径，把它喂给底层强大的 Docling 解析引擎
#Docling 解析出来的不是 ContentBlock，而是 Docling 自己极其复杂的 Document 树（里面包含了几十种不同的节点类型、坐标、属性）
    def parse(self, path: Path) -> list[ContentBlock]:
        with self._lock:
            result = self._get_converter().convert(path)
            status = getattr(result.status, "value", str(result.status))
            # 不覆盖原文档为一个仅包含部分页面的新版本。
            if status != "success":
                raise RuntimeError(f"PDF 未完整解析（状态：{status}），请检查文件或 OCR 配置")
            blocks = self.from_document(result.document)
        if not any(block.content.strip() for block in blocks):
            raise ValueError(
                "PDF 没有可提取的正文；扫描件请在 .env 设置 PDF_ENABLE_OCR=true 后重新导入"
            )
        return blocks

    @staticmethod
    #遍历 Docling 解析树。 PDF 解析 把这个复杂的 Document 树，降维翻译成了我们自己定义的 ContentBlock 列表
    def from_document(document: Any) -> list[ContentBlock]:
        """遍历 Docling 的阅读顺序；标题栈跨页保留，页眉页脚不进入知识库。"""
        blocks: list[ContentBlock] = []
        headings: list[tuple[int, str]] = []
        for item, _tree_depth in document.iterate_items():
            label = getattr(item.label, "value", str(item.label))
            if label in {"page_header", "page_footer", "picture"}:
                continue  #如果遇到页眉页脚，直接 continue 丢弃
            # 采用从 1 开始的 PDF 物理页码，不猜印刷页码。
            pages = tuple(sorted({int(prov.page_no) for prov in item.prov}))
            text = str(getattr(item, "text", "")).strip()
            # 标题栈保留章节层级，使跨页正文仍能追溯所在章节。
#如果是标题，用 headings 栈维护层级（实现跨页章节追踪） 遇到同级或更高级标题时，把旧的“弹出去（pop）"，实现#->##->###
            #headings[-1] 代表栈顶（当前最近的一个标题）  headings[-1][0] 就是栈顶标题的级别
            if label in {"title", "section_header"}:
                level = 0 if label == "title" else int(getattr(item, "level", 1))
                while headings and headings[-1][0] >= level:
                    headings.pop()
                if text:
                    headings.append((level, text))
                # 标题本身也可检索；不会让只有标题的 PDF 被误判为空。
            section = tuple(title for _, title in headings)
            table = None
            # 同时保留可检索的表格正文和原始单元格结构。
#如果是表格，提取网格结构，并用 render_table 转成 Markdown 格式的文本
            if label == "table":
                grid = item.data.grid  ##Docling 会把表格解析成网格
                cells = [cell.model_dump(mode="json") for cell in item.data.table_cells]
                rows = [[cell.text.strip() for cell in row] for row in grid]
                header_count = 0
                for row in grid:
                    if any(cell.column_header for cell in row):
                        header_count += 1
                    else:
                        break
                table = {
                    "rows": rows,
                    "header_rows": header_count,
                    "cells": cells,  # 包括 row_span / col_span 和原始坐标。
                    "caption": item.caption_text(document),
                }
                text = render_table(rows, header_count)
            if not text:
                continue
            blocks.append(
                ContentBlock(
                    content=text,
                    page_numbers=pages,
                    section=section,
                    block_type=label,
                    block_id=str(item.self_ref),
                    table=table,
                    parser="docling",
                )
            )
        return blocks

DEFAULT_PDF_PARSER = PDFBlockParser()##用于后续主函数直接调用

#（轻量文本解析器）
def markdown_blocks(text: str) -> list[ContentBlock]:
    """纯文本/Markdown 也保存标题路径；不为它们虚构 PDF 页码。"""
    blocks: list[ContentBlock] = []
    headings: list[tuple[int, str]] = []
    paragraph: list[str] = []
##flush把当前积攒的正文打包成一个内容块，然后把缓存清空，准备迎接下一段内容
    def flush() -> None:
        if paragraph:
            blocks.append(
                ContentBlock(
                    content="\n".join(paragraph).strip(),
                    section=tuple(title for _, title in headings),
                    block_id=f"text:{len(blocks)}",
                )
            )
            paragraph.clear()
##用正则 ^#{1,6} 识别 Markdown 标题，维护同样的 headings 栈
    for line in text.replace("\r\n", "\n").splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush()## <--- 先把上一段正文打包
            level, title = len(heading[1]), heading[2]
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
        elif line.strip():
            paragraph.append(line)
        else:
            flush() ## <--- 空行代表段落结束，立刻打包
    flush()
#你必须自己写逻辑：把第一句和第二句攒起来，遇到空行时，触发 flush()，把它们打包成一个完整的段落，再发给 ContentBlock
    if not blocks and headings:
        blocks.append(
            ContentBlock(
                content=headings[-1][1],
                section=tuple(title for _, title in headings),
                block_type="heading",
            )
        )
    return blocks

##是把“二维数组”变成“大模型能看懂的 Markdown 表格”的翻译官。
#如果把 | 直接传进去：大模型生成的表格会产生 | 某某|某某 |，直接导致 Markdown 的列错位。
#如果把 \n 直接传进去：Markdown 的表格是不允许跨行的，一换行就断。
#所以这里把换行替换成了 HTML 标签 <br>，保证渲染到网页或大模型解析时不会变形。
def render_table(rows: list[list[str]], header_count: int) -> str:
    """保留单元格边界；无表头时使用中性的列号，避免把首行数据误当表头。"""
    if not rows:
        return ""
## 1. 找出最长的一行，防止有的行短了一截
    width = max(len(row) for row in rows)

    def render(row: list[str]) -> str:
        #把每一个单元格里的 | 替换成 \|，防止把表格截断
        # 把单元格里的换行替换成 <br>，保证表格不会变形
        values = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        # 补齐缺失的列，并前后加上 | 符号
        return "| " + " | ".join(values + [""] * (width - len(values))) + " |"
# 3. 处理表头
    header_count = min(header_count, len(rows))
    if header_count:
    # # 如果有多行表头，用 " / " 把它们拼起来（比如 "基本信息 / 名称"）
        header = [
            " / ".join(
                dict.fromkeys(
                    row[col] for row in rows[:header_count] if col < len(row) and row[col]
                )
            )
            for col in range(width)
        ]
    # # 如果没识别出表头，就自动生成 "列 1" "列 2"
    else:
        header = [f"列 {index + 1}" for index in range(width)]
    # # 4. 拼接最终的 Markdown 字符串
    return "\n".join(
        [
            render(header),# 表头
            render(["---"] * width),# 分割线
            *(render(row) for row in rows[header_count:]),# 剩余的数据行
        ]
    )


def chunk_blocks(
    blocks: list[ContentBlock],
    chunk_size: int,
    chunk_overlap: int,
    fits_text: Callable[[str], bool] | None = None,
#fits_text 回调：这个传入的 fits_text 会直接调 BGE 模型去算真实的 Token 长度。如果字符没超，但 Token 超了，它依然会继续切短，防止向量化时被静默截断
) -> list[ContentBlock]:
    """每个块独立切分；页码/章节不丢失，表格分组时重复表头而不切断行。"""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size 必须大于 0，chunk_overlap 必须大于等于 0 且小于 chunk_size")
    chunks: list[ContentBlock] = []
##因为这里的 fits() 函数（在底层调用了 BGE 模型真实的 Token 计算器）。
# 有时候 1000 个中文字符可能只有 500 个 Token，但如果是 1000 个英文单词，可能就超了。
# 为了做到绝对不超，它用二分法去试探：切一半能不能行？不行再切一半，直到找到 BGE 模型能吃下的最大极限长度。
    def fits(text: str) -> bool:
        return len(text) <= chunk_size and (fits_text is None or fits_text(text))
##当一段文本（或者表格的一行）实在太长时，把它切成大小合适的碎片，同时绝对不能把上下文（标题、页码）弄丢
    def split_text(block: ContentBlock, body: str, prefix: str) -> None:
        start = 0# # 二分法查找切分位置
        while start < len(body):
            # 二分确定长度，给每一片保留章节/表格字段，避免 embedding 静默截断。
            low, high = 0, min(len(body) - start, chunk_size)
            while low < high:
                middle = (low + high + 1) // 2
                if fits(prefix + body[start : start + middle]):
                    low = middle
                else:
                    high = middle - 1
            if low == 0:
                raise ValueError("章节或表头过长，无法在当前 chunk_size/向量模型长度内切块")
            end = start + low
            chunks.append(replace(block, content=prefix + body[start:end]))
            if end == len(body):
                break
            # overlap 仅发生在同一个结构块内，不把相邻栏/页/表格混合。
            start += max(1, low - min(chunk_overlap, low // 2))
##prefix 强制绑定：每切出来的一片，都会在开头加上 "第一章 > 第一节\n"，保证切碎后上下文依然完整
    for block in blocks:
        prefix = " > ".join(block.section)
        prefix = prefix + "\n" if prefix else ""
        if block.table is None:
            split_text(block, block.content, prefix)
            continue

        table = block.table
        rows = table["rows"]
        header_count = int(table["header_rows"])
        headers = rows[:header_count]
        caption = str(table.get("caption", ""))
        table_prefix = prefix + (caption + "\n" if caption else "")
        group: list[list[str]] = []
        group_start = header_count
##当表格的数据行积攒到一定数量，再塞一行就会超长（超过 BGE 模型的 Token 限制）时，
# emit_group 就把当前积攒的这些行，加上原本的表头，打包成一个完整的 ContentBlock 塞进结果列表里，然后清空缓存，准备迎接下一批数据。
        def emit_group() -> None:
            if group:
                metadata = dict(
                    table,
                    rows=headers + list(group),
                    row_start=group_start,
                    row_end=group_start + len(group),
                )
                # 原始单元格信息只保留当前片段的行及表头，避免每片复制整张大表。
                metadata["cells"] = [
                    cell
                    for cell in table.get("cells", [])
                    if cell["start_row_offset_idx"] < header_count
                    or (
                        cell["start_row_offset_idx"] < metadata["row_end"]
                        and cell["end_row_offset_idx"] > group_start
                    )
                ]
                chunks.append(
                    replace(
                        block,
                        content=table_prefix + render_table(headers + group, header_count),
                        table=metadata,
                    )
                )
                group.clear()

        for row_index, row in enumerate(rows[header_count:], start=header_count):
            candidate = table_prefix + render_table(headers + group + [row], header_count)
            if group and not fits(candidate):
                emit_group()
            if not group:
                group_start = row_index
            if fits(table_prefix + render_table(headers + group + [row], header_count)):
                group.append(row)
            else:
                # 超长单行逐单元格切开；每一片重复列名，并保存原行号/列号。
                for col, value in enumerate(row):
                    if not value:
                        continue
                    label = (
                        " / ".join(
                            dict.fromkeys(
                                header[col]
                                for header in headers
                                if col < len(header) and header[col]
                            )
                        )
                        or f"列 {col + 1}"
                    )
                    metadata = dict(
                        rows=headers + [row],
                        header_rows=header_count,
                        row_start=row_index,
                        row_end=row_index + 1,
                        column_index=col,
                        column_header=label,
                        caption=caption,
                        cells=[
                            cell
                            for cell in table.get("cells", [])
                            if cell["start_row_offset_idx"] < header_count
                            or (
                                cell["start_row_offset_idx"]
                                <= row_index
                                < cell["end_row_offset_idx"]
                            )
                        ],
                    )
                    split_text(
                        replace(block, table=metadata),
                        value,
                        f"{table_prefix}表格第 {row_index + 1} 行，{label}：",
                    )
        emit_group()
        if rows and len(rows) == header_count:
            split_text(block, block.content, table_prefix)
    return chunks


def format_location(metadata: dict[str, Any]) -> str:
    """LLM 上下文与页面使用同样的定位信息，旧数据没有元数据时不编造页码。"""
    parts = []
    if metadata.get("page_numbers"):
        parts.append("第 " + "、".join(map(str, metadata["page_numbers"])) + " 页")
    if metadata.get("section"):
        parts.append("章节：" + " > ".join(metadata["section"]))
    if metadata.get("block_type") == "table":
        parts.append("表格")
    return "；".join(parts)


# ------------------------- 本地语义向量 ---------------------------


class EmbeddingBackend(Protocol):
    """RAGTool 依赖的最小向量接口，便于测试和替换模型。"""

    model_id: str

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class BGEEmbedding:
    """使用 BAAI BGE 中文语义模型生成归一化查询和文档向量。

    模型采用懒加载：仅在首次导入文档或执行检索时加载。多个 RAGTool 可以共享
    同一个实例，避免每个用户知识库分别占用一份模型内存。
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        query_instruction: str | None = None,
        batch_size: int | None = None,
    ):
        self.model_name = (
            model_name or os.getenv("EMBEDDING_MODEL_ID") or DEFAULT_EMBEDDING_MODEL
        ).strip()
        self.device = (device or os.getenv("EMBEDDING_DEVICE", "cpu")).strip()
        self.query_instruction = (
            query_instruction
            if query_instruction is not None
            else os.getenv("EMBEDDING_QUERY_INSTRUCTION", DEFAULT_QUERY_INSTRUCTION)
        )
        self.batch_size = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        if not self.model_name:
            raise ValueError("embedding 模型名称不能为空")
        if self.batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE 必须大于 0")
        self.model_id = f"sentence-transformers:{self.model_name}"
        self._model: Any | None = None
        self._lock = RLock()

    def _get_model(self) -> Any:
        """线程安全地懒加载 SentenceTransformer 模型。"""
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "BGE 向量模型需要 sentence-transformers，"
                    "请执行 python -m pip install -r requirements.txt"
                ) from exc

            cache_folder = os.getenv("EMBEDDING_CACHE_DIR") or None
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device or None,
                cache_folder=cache_folder,
            )
            return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        with self._lock:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                # 归一化后，检索时的点积等价于余弦相似度。
                normalize_embeddings=True,
            )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """为检索问题添加 BGE 中文指令并生成单条查询向量。"""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("查询文本不能为空")
        return self._encode([f"{self.query_instruction}{clean_text}"])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档向量，减少逐文本调用模型的额外开销。"""
        clean_texts = [text.strip() for text in texts]
        if any(not text for text in clean_texts):
            raise ValueError("文档文本不能为空")
        return self._encode(clean_texts)

    def fits_text(self, text: str) -> bool:
        """给切块器提供实际 token 上限，避免 BGE 编码时静默截断。"""
        model = self._get_model()
        with self._lock:
            return len(model.tokenizer.encode(text, truncation=False)) <= model.max_seq_length


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个已归一化向量的余弦相似度。"""
    if len(left) != len(right):
        raise ValueError(f"向量维度不一致：{len(left)} != {len(right)}")
    return sum(a * b for a, b in zip(left, right))


# ------------------------- SQLite 记忆 ---------------------------

##MemoryTool 处理的是用户个人的零碎记忆（比如学习笔记、最近的提问记录）
class MemoryTool:
    """使用 SQLite 保存用户笔记和文档问答记录。"""

    def __init__(
        self,
        user_id: str = "default_user",
        database_path: str | Path | None = None,
    ):
        self.user_id = user_id.strip() or "default_user"
        self.database_path = Path(
            database_path or PROJECT_DIR / f"memory_{self._safe_name(self.user_id)}.db"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        # 按列名读取记录；锁保护跨线程共享的 SQLite 连接。
        self.connection.row_factory = sqlite3.Row

        self._lock = RLock()
        with self.connection:
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
##防止用户输入奇怪的字符（比如 ../）导致路径穿越漏洞，生成安全的文件名。
    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", value) or "default_user"
##命令分发机制（路由层）
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
##run 方法的唯一职责就是：把外部的“标准调用格式”，适配并转换成内部好用的 execute 格式。 它就是个兼容层（Adapter）。
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
            f"[{row['memory_type']}] {row['content']} " f"(重要性={row['importance']:.1f})"
            for row in rows
        )
##（看最近记忆）
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
        return "\n".join(f"[{row['memory_type']}] {row['content']}" for row in rows)
##（统计）
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
##（看最重要记忆）
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
            f"[{row['memory_type']}] {row['content']} " f"(重要性={row['importance']:.1f})"
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


# ------------------------- SQLite RAG ----------------------------

#负责文档解析 → 文本分块 → 向量化存储 → 相似度检索 → 结合 LLM 生成回答的完整闭环
class RAGTool:
    """文档解析、分块、向量存储、检索与问答工具。"""

    def __init__(
        self,
        knowledge_base_path: str | Path = KNOWLEDGE_BASE_DIR,
        rag_namespace: str = "default",
        llm: HelloAgentsLLM | None = None,
        embedding: EmbeddingBackend | None = None,
        pdf_parser: PDFBlockParser | None = None,
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.rag_namespace = rag_namespace.strip() or "default"
        self.database_path = self.knowledge_base_path / f"{self._safe_name(self.rag_namespace)}.db"
        self.connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.embedding = embedding or BGEEmbedding()
        # 旅行页面的文件导入最终也会经过这里，所有用户共享懒加载的解析模型。
        self.pdf_parser = pdf_parser or DEFAULT_PDF_PARSER
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
                    embedding_model TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in self.connection.execute("PRAGMA table_info(rag_chunks)").fetchall()
            }
            if "embedding_model" not in columns:
                self.connection.execute(
                    "ALTER TABLE rag_chunks " "ADD COLUMN embedding_model TEXT NOT NULL DEFAULT ''"
                )
            if "metadata" not in columns:
                self.connection.execute(
                    "ALTER TABLE rag_chunks ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'"
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
##execute 负责内部调度
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
##run 负责兼容外部字典传参。
    def run(self, parameters: dict[str, Any] | str, **kwargs: Any) -> Any:
        if isinstance(parameters, dict):
            values = dict(parameters)
            action = str(values.pop("action", "search"))
            return self.execute(action, **values, **kwargs)
        return self.execute(parameters, **kwargs)

    @staticmethod#一个纯粹的工具函数
    def _convert_to_markdown(path: Path) -> str:
        #.md、.txt、.py、.json 直接读
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_TEXT_SUFFIXES:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            # PDF 必须走结构化入口，避免丢失页码与表格。
            raise ValueError("PDF 必须经过结构化解析，不能转换成单个长字符串")
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise RuntimeError(f"暂不支持文档格式：{suffix or '未知格式'}") from exc
        result = MarkItDown().convert(str(path))
        return getattr(result, "text_content", "") or ""
##它是 RAG 系统处理文件的唯一入口
#PDF：直接喂给 pdf_parser.parse，提取出带物理页码和表格结构的 list[ContentBlock]
#文本/Markdown/文档：先调 _convert_to_markdown 提取纯文本，然后喂给 markdown_blocks，用正则识别标题层级，打包成 list[ContentBlock]。
    def _parse_document(self, path: Path) -> list[ContentBlock]:
        """PDF 保留页码/章节/表格；其他文件也转换成统一的内容块。"""
        if path.suffix.lower() == ".pdf":
            return self.pdf_parser.parse(path)
        return markdown_blocks(self._convert_to_markdown(path))

    def _chunk_blocks(
        self,
        blocks: list[ContentBlock],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[ContentBlock]:
        """正文分片和向量长度检查同时进行，每片继承原文定位信息。"""
        return chunk_blocks(
            blocks,
            chunk_size,
            chunk_overlap,
            fits_text=getattr(self.embedding, "fits_text", None),
        )
##_add_document（处理文件）
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
            # 网页上传 -> 本方法 -> 结构解析 -> 切块 -> BGE/SQLite。
            blocks = self._parse_document(path)
            chunks = self._chunk_blocks(blocks, chunk_size, chunk_overlap)
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
##处理复制粘贴的文本
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
        chunks = self._chunk_blocks(markdown_blocks(text), chunk_size, chunk_overlap)
        if not chunks:
            return {"success": False, "error": "文档内容不能为空"}
        return self._store_chunks(
            document_id=clean_id,
            source=source_label or clean_id,
            chunks=chunks,
        )
##_store_chunks（向量化并入库）
    def _store_chunks(
        self,
        document_id: str,
        source: str,
        chunks: list[ContentBlock],
    ) -> dict[str, Any]:
        created_at = datetime.now().astimezone().isoformat()
        # 向量模型只编码正文；页码等元数据另存，检索命中后一起取回。# 批量向量化（只编码正文）
        vectors = self.embedding.embed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("embedding 返回的向量数量与文本块数量不一致")
        ## 组装行数据（包含 metadata）
        rows = [
            (
                document_id,
                source,
                index,
                chunk.content,
                json.dumps(vector),
                self.embedding.model_id,
                json.dumps(chunk.metadata(), ensure_ascii=False),
                created_at,
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
         # 事务保护：先删旧文档，再批量插入新块。先 DELETE 旧的，再插新的。上传同名文件，就相当于更新。
        with self._lock, self.connection:
            self.connection.execute(
                "DELETE FROM rag_chunks WHERE document_id = ?",
                (document_id,),
            )
            self.connection.executemany(
                """
                INSERT INTO rag_chunks
                    (document_id, source, chunk_index, content, embedding,
                     embedding_model, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return {
            "success": True,
            "document_id": document_id,
            "chunks": len(chunks),
            "message": f"已导入文档，共 {len(chunks)} 个文本块。",
        }
#_search（余弦相似度计算）
    def _search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            # # 2. 从数据库里把所有块捞出来
            rows = self.connection.execute("SELECT * FROM rag_chunks").fetchall()
        if not rows:
            return []

        # 小型库全量计算相似度；大规模数据可替换成向量索引。
        # # 1. 把用户问题变成向量
        query_vector = self.embedding.embed_query(query)
        row_vectors: dict[int, list[float]] = {}
        stale_rows: list[sqlite3.Row] = []
        for row in rows:
            try:
                vector = [float(value) for value in json.loads(row["embedding"])]
            except (TypeError, ValueError, json.JSONDecodeError):
                vector = []
            ## # 3. 超强防呆：检查向量是否过期
            if row["embedding_model"] != self.embedding.model_id or len(vector) != len(
                query_vector## # 发现是旧模型算的，或者维度不对
            ):
                stale_rows.append(row)
            else:
                row_vectors[int(row["id"])] = vector

        # 模型升级或导入旧数据库后，直接利用已保存的正文批量重建失效向量。
        if stale_rows:
            refreshed_vectors = self.embedding.embed_documents(
                [str(row["content"]) for row in stale_rows]
            )
            if len(refreshed_vectors) != len(stale_rows):
                raise RuntimeError("embedding 返回的重建向量数量不正确")
            updates = []
            for row, vector in zip(stale_rows, refreshed_vectors):
                row_id = int(row["id"])
                row_vectors[row_id] = vector
                updates.append((json.dumps(vector), self.embedding.model_id, row_id))
            with self._lock, self.connection:
                self.connection.executemany(
                    """
                    UPDATE rag_chunks
                    SET embedding = ?, embedding_model = ?
                    WHERE id = ?
                    """,
                    updates,
                )
##计算余弦相似度并排序 因为用了 normalize_embeddings=True，这里的向量都是单位向量，点积直接等于余弦相似度。然后降序排序
        scored = [
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "source": row["source"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "score": cosine_similarity(
                    query_vector,
                    row_vectors[int(row["id"])],
                ),
            }
            for row in rows
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
##把 content（正文）和 metadata（页码、章节、表格）一起返回。把上下文和出处死死绑在一起
        return scored[: max(1, int(limit))]
##如果用户传进来的 query 是一串空白字符（比如不小心敲了几个空格），它直接返回空列表 []。
# 因为如果你把一个空字符串丢给 _search，embed_query 可能会报错，或者算出一个毫无意义的向量，导致检索出完全无关的内容
    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        clean_query = query.strip()
        return self._search(clean_query, limit) if clean_query else []
##在前面的 execute 方法里，如果你传了 "search" 这个指令，它就会去调这个方法
    def _search_action(self, query: str, limit: int = 5, **kwargs: Any) -> str:
        results = self.search(query, limit)
        if not results:
            return "知识库中没有找到相关内容。"
        return self._format_context(results)

#####search_action 先调用 search，search 再调用 _search

    @staticmethod
##（给大模型的参考资料）
    def _format_context(results: list[dict[str, Any]]) -> str:
        sections = []
        for index, item in enumerate(results, start=1):
            location = format_location(item.get("metadata", {}))
            sections.append(
                f"[{index}] 来源：{item['source']}，片段：{item['chunk_index'] + 1}，"
                f"{location + '，' if location else ''}"
                f"相似度：{item['score']:.3f}\n{item['content']}"
            )
        return "\n\n".join(sections)
#专门用来共MQE和HyDE使用
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
##要解决的痛点：用户提问常常极其口语化，或者字数太少。比如只问“杭州下雨去哪”，
# 文档里却写的是“杭州室内避雨景点指南”。字面差太远，向量检索可能搜不到 通过这个MQE扩写后再发给大模型
    def _generate_multi_queries(self, question: str) -> list[str]:#负责“把问题问宽”。
        """MQE：改写问题以扩大召回，失败时保留原问题。"""
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
##用户提问往往很短（Query 短），而文档往往很长、很正式（Document 长）。这导致“长短不匹配”，在向量空间里的距离可能很远
##原理：假答案的长度和用词习惯，跟真实文档非常像。用假答案去匹配真文档，精准度往往高得惊人。
##安全兜底：这段假答案绝对不能被当成参考资料喂给最终的大模型。它仅仅用来“探路”，探完路直接扔掉，真正吃进去的，是搜出来的真实文档
    def _generate_hyde(self, question: str) -> str:# 负责“把问题问深”
        """HyDE：假设答案仅用于检索，不作为真实证据。"""
        if self.llm is None:
            return ""
        try:
            return self._llm_text(
                "写一段简短的假设性答案，用于检索相关文档。\n问题：" + question,
                "你是文档检索助手。",
            )
        except Exception:
            return ""
##前面所有的准备工作（解析、切块、向量化、入库），都是为了这个函数服务。
# 当用户提出一个复杂问题时，_ask 负责把所有能用的高级检索手段全用上，
# 最后让大模型根据搜出来的资料，给出一个带精确出处的回答
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
##如果开启了高级检索，用户原本的一句话，会被扩充成：
#原问题
#大模型改写的 2~3 个变体（MQE）
#大模型“编”的一段假答案（HyDE）
        queries = [clean_question]
        if enable_advanced_search and enable_mqe:
            queries = self._generate_multi_queries(clean_question)
        if enable_advanced_search and enable_hyde:
            hyde = self._generate_hyde(clean_question)
            if hyde:
                queries.append(hyde)
        # 多查询按块 ID 去重，同一片段保留最高分。多路并发搜索与合并去重
        #因为不同的查询词（比如“酒店”和“住宿”）可能搜出同一段攻略。
        # 去重后，避免了把重复的参考文档喂给大模型，节约 Token
        merged: dict[int, dict[str, Any]] = {}
        for query in queries:## 每次检索多捞几条（至少 5 条）
            for item in self._search(query, limit=max(int(limit), 5)):
                ## 核心逻辑：如果同一个文本块被多次命中，只保留最高分的那次
                if item["id"] not in merged or item["score"] > merged[item["id"]]["score"]:
                    merged[item["id"]] = item
        results = sorted(
            merged.values(),
            key=lambda item: item["score"],
            reverse=True,
        )[: max(1, int(limit))]
        if not results:
            return "知识库中没有找到相关内容。"
##1. 把查出来的结构化结果，转成带页码和来源的 Markdown 文本
        context = self._format_context(results)
##2. 拼装 Prompt，请大模型回答
        return self._llm_text(
            f"用户问题：\n{clean_question}\n\n参考资料：\n{context}\n\n"
            "只依据参考资料回答。资料不足时明确说明。"
            "回答末尾列出使用的来源、片段编号，以及资料提供的页码和章节；"
            "没有提供页码时不得编造页码。",
            "你是严谨的本地文档问答助手，不得编造资料中不存在的事实。",
        )
#它负责去数据库里数一数 chunks document_id，然后汇报结果
    def _stats(self, **kwargs: Any) -> str:
        with self._lock:
            counts = self.connection.execute(
                "SELECT COUNT(*) AS chunks, COUNT(DISTINCT document_id) AS documents "
                "FROM rag_chunks"
            ).fetchone()
        return json.dumps(
            {
                "namespace": self.rag_namespace,
                "embedding_model": self.embedding.model_id,
                "documents": int(counts["documents"]),
                "chunks": int(counts["chunks"]),
            },
            ensure_ascii=False,
        )
##汇报的是微观明细（具体有哪几个文件？每个文件分别被切成了几块？什么时候传上来的？）
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
        return [dict(row) for row in rows]

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


# ------------------------- 独立文档问答 ---------------------------

##前面讲的 RAGTool（文档检索）和 MemoryTool（记忆笔记）都是非常底层的零件，它们各自负责一块。
# 而 DocumentAssistant 就是把这些零件拼装在一起，对外提供一个“文档问答系统”的完整服务
class DocumentAssistant:
    """组合本地 LLM、RAG 和持久化记忆的文档问答服务。"""

    def __init__(
        self,
        user_id: str = "default_user",
        llm: HelloAgentsLLM | None = None,
        knowledge_base_path: str | Path = KNOWLEDGE_BASE_DIR,
        memory_database_path: str | Path | None = None,
        embedding: EmbeddingBackend | None = None,
    ):
        self.user_id = user_id.strip() or "default_user"
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.llm = llm or create_llm()
        self.memory_tool = MemoryTool(
            user_id=self.user_id,
            database_path=memory_database_path,
        )
#它把 RAGTool 的命名空间强行绑定成了 pdf_{user_id}。这意味着，用户 A 和用户 B 虽然用同一个类，
# 但底层连的是两个完全不同的 SQLite 数据库文件，实现了物理级别的数据隔离。
        self.rag_tool = RAGTool(
            knowledge_base_path=knowledge_base_path,
            rag_namespace=f"pdf_{self.user_id}",
            llm=self.llm,
            embedding=embedding,
        )
        self.session_started_at = datetime.now().astimezone()
        self.documents_loaded = 0  #导入了多少文档
        self.questions_asked = 0   #问了多少问题
        self.notes_saved = 0       #存了多少笔记
##这个函数体现了“组合”的魅力：不仅把文件丢给 RAGTool 处理，还把这动作作为一条“情景记忆”写入了 MemoryTool。
##上传 PDF/Markdown 文件
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
##处理复制粘贴的文本
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

    def ask(self, question: str, use_advanced_search: bool = False) -> str:
        clean_question = question.strip()
        if not clean_question:
            return "问题不能为空。"
        #调 RAGTool 执行检索+大模型生成
        answer = self.rag_tool.execute(
            "ask",
            question=clean_question,
            limit=5,
            enable_advanced_search=use_advanced_search,
        )
        ## 记录提问次数，并写入记忆库
        self.questions_asked += 1
        self.memory_tool.execute(
            "add",
            content=f"提问：{clean_question}",
            memory_type="working",
            importance=0.6,
            session_id=self.session_id,
        )
        return str(answer)
#纯粹的笔记记录，不经过大模型，直接存进 MemoryTool
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
##它是在帮你回顾：“我之前问过哪些跟杭州有关的问题？我记过哪些关于杭州的笔记？”
#它不是去庞大的 RAG 知识库（PDF 攻略）里搜，而是去你个人的“日记本”（MemoryTool）里搜。
    def recall(self, query: str, limit: int = 5) -> str:
        return self.memory_tool.execute("search", query=query, limit=limit)

    def list_documents(self) -> list[dict[str, Any]]:
        return self.rag_tool.execute("list_documents")

    def delete_document(self, document_id: str) -> dict[str, Any]:
        return self.rag_tool.execute("delete_document", document_id=document_id)
#实时统计把本会话的计数器（本次导入了多少、问了多少）和底层数据库的真实数字（库里有几个文档）汇总在一起，方便前端展示
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
##生成 JSON 报告
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


def create_gradio_ui(assistant: DocumentAssistant):
    """创建文档导入、问答和记忆管理页面。"""
    import gradio as gr

    def document_rows() -> list[list[Any]]:
        columns = ("document_id", "source", "chunks", "created_at")
        return [[item[column] for column in columns] for item in assistant.list_documents()]

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
        gr.Markdown(f"当前模型：`{assistant.llm.model}` · 本地服务：`{assistant.llm.base_url}`")
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
        for event in (send.click, message.submit):
            event(
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
