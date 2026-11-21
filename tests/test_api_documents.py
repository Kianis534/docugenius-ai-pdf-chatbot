"""End-to-end tests for the document (RAG) endpoints."""

from __future__ import annotations

from conftest import REPORT_PAGES, make_blank_pdf, make_pdf

from docugenius.errors import LLMQuotaError


def test_upload_indexes_pdf(uploaded):
    assert len(uploaded["doc_id"]) == 32
    assert uploaded["filename"] == "report.pdf"
    assert uploaded["pages"] == 3
    assert uploaded["chunks"] >= 3


def test_reupload_is_idempotent(client, report_pdf, uploaded):
    again = client.post("/api/documents", files={"file": ("copy.pdf", report_pdf, "application/pdf")})
    assert again.status_code == 201
    assert again.json()["doc_id"] == uploaded["doc_id"]
    assert again.json()["created_at"] == uploaded["created_at"]
    assert again.json()["filename"] == "copy.pdf"


def test_get_document_metadata(client, uploaded):
    response = client.get(f"/api/documents/{uploaded['doc_id']}")
    assert response.status_code == 200
    assert response.json() == uploaded


def test_rejects_non_pdf(client):
    response = client.post("/api/documents", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert response.status_code == 422
    assert "PDF" in response.json()["detail"]


def test_rejects_pdf_without_text(client):
    response = client.post("/api/documents", files={"file": ("scan.pdf", make_blank_pdf(), "application/pdf")})
    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]


def test_rejects_corrupted_pdf(client):
    response = client.post("/api/documents", files={"file": ("bad.pdf", b"%PDF-1.7 garbage", "application/pdf")})
    assert response.status_code == 422


def test_rejects_oversized_upload(client):
    big = make_pdf(["x"]) + b"0" * (2 * 1024 * 1024 + 10)
    response = client.post("/api/documents", files={"file": ("big.pdf", big, "application/pdf")})
    assert response.status_code == 413


def test_query_returns_answer_with_page_sources(client, uploaded, fake_llm):
    response = client.post(
        f"/api/documents/{uploaded['doc_id']}/query", json={"question": "How much did revenue grow?"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == fake_llm.reply
    assert body["model"] == "fake-model"
    assert body["sources"] and all(s["page"] in (1, 2, 3) for s in body["sources"])
    prompt = fake_llm.calls[-1]["prompt"]
    assert "Question: How much did revenue grow?" in prompt
    assert "[Page" in prompt


def test_query_validates_input(client, uploaded):
    assert client.post(f"/api/documents/{uploaded['doc_id']}/query", json={"question": "   "}).status_code == 422
    assert client.post(f"/api/documents/{uploaded['doc_id']}/query", json={}).status_code == 422


def test_unknown_and_malformed_document_ids(client):
    unknown = "0" * 32
    assert client.post(f"/api/documents/{unknown}/query", json={"question": "hi"}).status_code == 404
    assert client.post("/api/documents/not-an-id/query", json={"question": "hi"}).status_code == 422
    assert client.get("/api/documents/..%2F..%2Fetc/tables").status_code in (404, 422)


def test_summary_uses_whole_document(client, uploaded, fake_llm):
    response = client.post(f"/api/documents/{uploaded['doc_id']}/summary")
    assert response.status_code == 200
    assert response.json()["summary"] == fake_llm.reply
    prompt = fake_llm.calls[-1]["prompt"]
    assert all(page.split(".")[0] in prompt for page in REPORT_PAGES)


def test_tables_are_extracted_with_pdfplumber(client, uploaded):
    response = client.get(f"/api/documents/{uploaded['doc_id']}/tables")
    assert response.status_code == 200
    tables = response.json()["tables"]
    assert len(tables) == 1
    assert tables[0]["page"] == 3
    assert tables[0]["rows"][0] == ["Region", "Revenue", "Growth"]
    assert "| Asia | $4.2M | 18% |" in tables[0]["markdown"]
    # Second call is served from the cache.
    assert client.get(f"/api/documents/{uploaded['doc_id']}/tables").json() == response.json()


def test_llm_errors_are_mapped_to_http_status(client, uploaded, fake_llm):
    fake_llm.error = LLMQuotaError()
    response = client.post(f"/api/documents/{uploaded['doc_id']}/query", json={"question": "Revenue?"})
    assert response.status_code == 429
    assert "quota" in response.json()["detail"].lower()


def test_llm_not_configured_returns_503(client, uploaded, fake_llm):
    fake_llm.configured = False
    response = client.post(f"/api/documents/{uploaded['doc_id']}/summary")
    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]
