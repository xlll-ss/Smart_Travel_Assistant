"""结构化 PDF 的确定性回归测试；默认不下载版面/向量模型。"""

import json
import os
import sqlite3
from types import SimpleNamespace

import pytest
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    TableCell,
    TableData,
)

from travel_assistant import KnowledgeSnippet, TravelKnowledgeBase, TravelKnowledgeReport
from wise_douctment import (
    ContentBlock,
    PDFBlockParser,
    RAGTool,
    chunk_blocks,
    markdown_blocks,
    render_table,
)


class TestEmbedding:
    model_id = "test:pdf"

    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def provenance(page):
    return ProvenanceItem(
        page_no=page,
        bbox=BoundingBox(l=10, t=10, r=200, b=40),
        charspan=(0, 20),
    )


def structured_document():
    document = DoclingDocument(name="travel")
    document.add_text(DocItemLabel.TITLE, "杭州攻略", prov=provenance(1))
    document.add_heading("室内地点", level=1, prov=provenance(1))
    document.add_text(DocItemLabel.TEXT, "博物馆适合下雨天参观。", prov=provenance(1))
    document.add_text(DocItemLabel.PAGE_FOOTER, "不要导入的页脚", prov=provenance(1))
    document.add_text(DocItemLabel.TEXT, "书店适合安静阅读。", prov=provenance(2))
    rows = [["地点", "门票", "室内"], ["博物馆", "免费", "是"], ["书店", "按消费", "是"]]
    cells = [
        TableCell(
            text=value,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
    ]
    document.add_table(
        data=TableData(num_rows=3, num_cols=3, table_cells=cells),
        prov=provenance(2),
    )
    return document


def test_docling_structure_keeps_pages_sections_and_cells():
    blocks = PDFBlockParser.from_document(structured_document())
    assert all("不要导入" not in block.content for block in blocks)
    bookstore = next(block for block in blocks if "书店适合" in block.content)
    assert bookstore.page_numbers == (2,)
    assert bookstore.section == ("杭州攻略", "室内地点")
    table = next(block for block in blocks if block.block_type == "table")
    assert table.page_numbers == (2,)
    assert table.table["header_rows"] == 1
    assert table.table["rows"][1] == ["博物馆", "免费", "是"]
    assert table.table["cells"][0]["column_header"] is True


def test_table_chunks_repeat_headers_and_keep_complete_rows():
    table = next(
        block
        for block in PDFBlockParser.from_document(structured_document())
        if block.block_type == "table"
    )
    chunks = chunk_blocks([table], chunk_size=80, chunk_overlap=10)
    assert len(chunks) == 2
    assert all("| 地点 | 门票 | 室内 |" in chunk.content for chunk in chunks)
    assert all(chunk.section == table.section and chunk.page_numbers == (2,) for chunk in chunks)
    assert "| 博物馆 | 免费 | 是 |" in chunks[0].content
    assert "| 书店 | 按消费 | 是 |" in chunks[1].content
    assert [chunk.table["row_start"] for chunk in chunks] == [1, 2]


def test_headerless_table_does_not_discard_first_row():
    text = render_table([["博物馆", "免费"], ["游乐园", "200 元"]], 0)
    assert "| 列 1 | 列 2 |" in text
    assert "| 博物馆 | 免费 |" in text
    assert "| 游乐园 | 200 元 |" in text


def test_long_cell_keeps_column_and_page_without_truncation():
    table = ContentBlock(
        content="",
        page_numbers=(3,),
        section=("攻略",),
        block_type="table",
        table={
            "rows": [["地点", "说明"], ["书店", "适合安静阅读" * 40]],
            "header_rows": 1,
            "cells": [],
            "caption": "",
        },
    )
    chunks = chunk_blocks([table], 80, 0, fits_text=lambda text: len(text) <= 55)
    assert len(chunks) > 2
    assert all(len(chunk.content) <= 55 and chunk.page_numbers == (3,) for chunk in chunks)
    descriptions = [chunk for chunk in chunks if chunk.table.get("column_index") == 1]
    assert (
        "".join(chunk.content.split("说明：", 1)[1] for chunk in descriptions)
        == "适合安静阅读" * 40
    )


def test_paragraph_chunks_keep_section_on_every_piece():
    blocks = [ContentBlock("甲" * 100, (2, 3), ("章节",))]
    chunks = chunk_blocks(blocks, 50, 0, fits_text=lambda text: len(text) <= 25)
    assert all(chunk.content.startswith("章节\n") for chunk in chunks)
    assert all(chunk.page_numbers == (2, 3) for chunk in chunks)
    assert "".join(chunk.content.split("\n", 1)[1] for chunk in chunks) == "甲" * 100


def test_metadata_survives_import_search_reopen_and_citations(tmp_path):
    blocks = PDFBlockParser.from_document(structured_document())
    parser = SimpleNamespace(parse=lambda path: blocks)
    pdf = tmp_path / "travel.pdf"
    pdf.touch()
    rag = RAGTool(tmp_path, "structured", embedding=TestEmbedding(), pdf_parser=parser)
    try:
        assert rag.execute("add_document", file_path=str(pdf))["success"]
    finally:
        rag.close()
    rag = RAGTool(tmp_path, "structured", embedding=TestEmbedding())
    try:
        results = rag.search("门票", limit=20)
        item = next(item for item in results if item["metadata"]["block_type"] == "table")
        assert item["metadata"]["page_numbers"] == [2]
        assert item["metadata"]["table"]["rows"][1][1] == "免费"
        assert "第 2 页" in rag._format_context([item])
        snippet = KnowledgeSnippet(
            **{
                key: item[key]
                for key in (
                    "document_id",
                    "source",
                    "chunk_index",
                    "content",
                    "score",
                    "metadata",
                )
            }
        )
        report = TravelKnowledgeReport((snippet,))
        assert "第 2 页" in report.to_context()
        assert "章节：杭州攻略 > 室内地点" in report.to_markdown()
        assert "表格" in report.to_markdown()
    finally:
        rag.close()


def test_old_database_migrates_without_losing_documents(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE rag_chunks (
            id INTEGER PRIMARY KEY, document_id TEXT, source TEXT, chunk_index INTEGER,
            content TEXT, embedding TEXT, created_at TEXT)"""
        )
        connection.execute(
            "INSERT INTO rag_chunks VALUES (1, 'old', 'old.pdf', 0, ?, ?, '')",
            ("原来的攻略", json.dumps([1.0])),
        )
    rag = RAGTool(tmp_path, "old", embedding=TestEmbedding())
    try:
        result = rag.search("攻略")[0]
        assert result["content"] == "原来的攻略"
        assert result["metadata"] == {}
        assert "第 " not in rag._format_context([result])
    finally:
        rag.close()


def test_travel_upload_and_search_use_structured_pdf_pipeline(tmp_path, monkeypatch):
    """覆盖网页使用的旅行知识库入口，防止解析模块存在却没有接入 RAG。"""
    calls = []

    def parse(self, path):
        calls.append(path)
        return PDFBlockParser.from_document(structured_document())

    monkeypatch.setattr(PDFBlockParser, "parse", parse)
    pdf = tmp_path / "trip.pdf"
    pdf.touch()
    knowledge_base = TravelKnowledgeBase(tmp_path, embedding=TestEmbedding())
    try:
        assert knowledge_base.import_file(pdf)["success"]
        report = knowledge_base.search("根据攻略推荐室内地点", limit=20)
        assert calls == [pdf]
        tables = [
            snippet for snippet in report.snippets if snippet.metadata.get("block_type") == "table"
        ]
        assert tables and tables[0].metadata["page_numbers"] == [2]
        assert "第 2 页" in report.to_context()
        assert "章节：杭州攻略 > 室内地点" in report.to_markdown()
        assert "| 博物馆 | 免费 | 是 |" in report.to_context()
    finally:
        knowledge_base.close()


def test_failed_pdf_import_preserves_existing_document(tmp_path):
    def fail(path):
        raise RuntimeError("PDF 解析失败")

    rag = RAGTool(
        tmp_path, "failure", embedding=TestEmbedding(), pdf_parser=SimpleNamespace(parse=fail)
    )
    pdf = tmp_path / "guide.pdf"
    pdf.touch()
    try:
        rag.execute("add_text", text="原来的攻略", document_id="guide.pdf")
        result = rag.execute("add_document", file_path=str(pdf))
        assert result["success"] is False
        assert rag.search("攻略")[0]["content"] == "原来的攻略"
    finally:
        rag.close()


@pytest.mark.parametrize("status", ["partial_success", "failure"])
def test_incomplete_pdf_is_rejected(status, tmp_path):
    parser = PDFBlockParser()
    parser._converter = SimpleNamespace(convert=lambda path: SimpleNamespace(status=status))
    with pytest.raises(RuntimeError, match="未完整解析"):
        parser.parse(tmp_path / "input.pdf")


def test_empty_pdf_points_to_ocr_and_markdown_has_no_fake_pages(tmp_path):
    parser = PDFBlockParser()
    parser._converter = SimpleNamespace(
        convert=lambda path: SimpleNamespace(
            status="success",
            document=DoclingDocument(name="scanned"),
        )
    )
    with pytest.raises(ValueError, match="PDF_ENABLE_OCR=true"):
        parser.parse(tmp_path / "scan.pdf")
    blocks = markdown_blocks("# 杭州\n## 室内\n书店攻略。")
    assert blocks[0].section == ("杭州", "室内")
    assert blocks[0].page_numbers == ()


def create_layout_pdf(path):
    """生成两页有真实文字和表格线的回归输入，不依赖在线 PDF。"""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for number in (1, 2):
        page = writer.add_blank_page(width=595, height=842)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            }
        )
        commands = [
            "BT /F1 24 Tf 50 780 Td (Travel Guide) Tj ET",
            "BT /F1 18 Tf 50 735 Td (Indoor Activities) Tj ET",
            f"BT /F1 12 Tf 50 700 Td (Page {number}: quiet places for reading and learning.) Tj ET",
        ]
        rows = [
            ["Place", "Ticket", "Indoor"],
            ["Museum", "Free", "Yes"],
            ["Bookstore", "10", "Yes"],
            ["Gallery", "20", "Yes"],
            ["Library", "Free", "Yes"],
        ]
        for index in range(6):
            y = 640 - 32 * index
            commands.append(f"50 {y} m 530 {y} l S")
        for x in (50, 240, 400, 530):
            commands.append(f"{x} 640 m {x} 480 l S")
        for r, row in enumerate(rows):
            for x, value in zip((60, 250, 410), row):
                commands.append(f"BT /F1 12 Tf {x} {620 - r * 32} Td ({value}) Tj ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


@pytest.mark.skipif(
    os.getenv("RUN_PDF_INTEGRATION") != "1", reason="真实 Docling 测试需先下载版面与表格模型"
)
def test_real_pdf_layout_and_table_extraction(tmp_path):
    pdf = tmp_path / "layout.pdf"
    create_layout_pdf(pdf)
    blocks = PDFBlockParser().parse(pdf)
    assert {page for block in blocks for page in block.page_numbers} == {1, 2}
    tables = [block for block in blocks if block.block_type == "table"]
    assert len(tables) == 2
    assert all("Museum" in block.content and "Free" in block.content for block in tables)
    assert any(block.section for block in blocks)
