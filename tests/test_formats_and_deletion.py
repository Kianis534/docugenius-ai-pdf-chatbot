"""Tests for Word, text and Markdown support and for deleting documents."""

from __future__ import annotations

import io
import zipfile

from conftest import REPORT_TABLE, make_docx

from docugenius.services.documents import _group_into_sections, _markdown_tables

NOTES = (
    "Project kickoff notes.\n\n"
    "The team agreed to ship the beta in March and to focus on accessibility.\n\n"
    "Budget: the hardware budget was approved at 40,000 dollars.\n"
)
MARKDOWN = """# Release plan

Milestones for the next quarter.

| Milestone | Owner | Due |
| --- | :---: | ---: |
| Beta | Sara | March |
| GA | Omar | June |

Everything else is on track.
"""


def upload(client, name: str, payload: bytes, content_type: str = "application/octet-stream"):
    return client.post("/api/documents", files={"file": (name, payload, content_type)})


# --- Plain text and Markdown ---------------------------------------------- #
def test_text_upload_cites_sections(client, fake_llm):
    response = upload(client, "notes.txt", NOTES.encode(), "text/plain")
    assert response.status_code == 201, response.text
    doc = response.json()
    assert doc["kind"] == "text"
    assert doc["pages"] == 1  # one section
    answer = client.post(f"/api/documents/{doc['doc_id']}/query", json={"question": "What was the budget?"}).json()
    assert answer["sources"] and all(s["page"] is None and s["section"] == 1 for s in answer["sources"])
    assert "[Section 1]" in fake_llm.calls[-1]["prompt"]


def test_text_encodings_are_handled(client):
    utf16 = "Café menu: crème brûlée and pain au chocolat.".encode("utf-16")
    cp1252 = "Price list in euros: 5 € per item.".encode("cp1252")
    assert upload(client, "menu.txt", utf16).status_code == 201
    assert upload(client, "prices.txt", cp1252).status_code == 201


def test_binary_file_with_text_extension_is_rejected(client):
    response = upload(client, "notes.txt", b"\x00\x01\x02binary\x00data")
    assert response.status_code == 422
    assert "plain text" in response.json()["detail"]


def test_empty_text_file_is_rejected(client):
    response = upload(client, "empty.md", b"   \n\n  ")
    assert response.status_code == 422
    assert "does not contain any text" in response.json()["detail"]


def test_markdown_pipe_tables_are_extracted(client):
    doc = upload(client, "plan.md", MARKDOWN.encode(), "text/markdown").json()
    tables = client.get(f"/api/documents/{doc['doc_id']}/tables").json()["tables"]
    assert len(tables) == 1
    assert tables[0]["page"] is None
    assert tables[0]["rows"] == [["Milestone", "Owner", "Due"], ["Beta", "Sara", "March"], ["GA", "Omar", "June"]]


def test_markdown_table_parser_requires_a_separator_row():
    assert _markdown_tables("| a | b |\n| c | d |\n") == []
    assert _markdown_tables("| a | b |\n|---|---|\n| 1 | 2 |\n") == [[["a", "b"], ["1", "2"]]]


def test_sections_group_paragraphs_by_size():
    blocks = ["a" * 1200, "b" * 1200, "c" * 1200, "d" * 100]
    sections = _group_into_sections(blocks, size=2500)
    assert [number for number, _ in sections] == [1, 2]
    assert sections[0][1].count("\n\n") == 1  # first two paragraphs share a section


# --- Word ------------------------------------------------------------------ #
def test_docx_upload_query_and_tables(client, fake_llm):
    payload = make_docx(["Revenue grew 18 percent in Asia.", "Hiring doubled in Lahore."], table=REPORT_TABLE)
    response = upload(client, "update.docx", payload)
    assert response.status_code == 201, response.text
    doc = response.json()
    assert doc["kind"] == "docx"

    client.post(f"/api/documents/{doc['doc_id']}/query", json={"question": "Where did revenue grow?"})
    prompt = fake_llm.calls[-1]["prompt"]
    assert "[Section 1]" in prompt
    assert "Revenue grew 18 percent in Asia." in prompt

    tables = client.get(f"/api/documents/{doc['doc_id']}/tables").json()["tables"]
    assert len(tables) == 1
    assert tables[0]["rows"][0] == ["Region", "Revenue", "Growth"]
    assert "| Asia | $4.2M | 18% |" in tables[0]["markdown"]


def test_zip_that_is_not_a_word_file_is_rejected(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "not a word document")
    response = upload(client, "fake.docx", buffer.getvalue())
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]


def test_corrupted_docx_is_rejected(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", "<<< not xml")
    response = upload(client, "broken.docx", buffer.getvalue())
    assert response.status_code == 422
    assert "Word file could not be read" in response.json()["detail"]


# --- Deletion -------------------------------------------------------------- #
def test_delete_removes_document_and_files(client, uploaded, settings, report_pdf):
    doc_id = uploaded["doc_id"]
    client.get(f"/api/documents/{doc_id}/tables")  # create a cached artefact too

    response = client.delete(f"/api/documents/{doc_id}")
    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.post(f"/api/documents/{doc_id}/query", json={"question": "hi"}).status_code == 404
    assert not (settings.documents_dir / doc_id / "meta.json").exists()
    assert client.get("/health").json()["services"]["documents"]["indexed"] == 0

    # Deleting twice is a clean 404, and the same file can be uploaded and queried again.
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    again = client.post("/api/documents", files={"file": ("report.pdf", report_pdf, "application/pdf")})
    assert again.status_code == 201
    assert again.json()["doc_id"] == doc_id
    query = client.post(f"/api/documents/{doc_id}/query", json={"question": "How much did revenue grow?"})
    assert query.status_code == 200, query.text


def test_delete_validates_document_id(client):
    assert client.delete("/api/documents/not-a-valid-id").status_code == 422
    assert client.delete(f"/api/documents/{'f' * 32}").status_code == 404


def test_cors_preflight_allows_delete(client):
    response = client.options(
        "/api/documents/" + "a" * 32,
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "DELETE"},
    )
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]
