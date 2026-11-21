# Contributing to DocuGenius

Thanks for taking the time to contribute! Bug reports, feature ideas, docs fixes and pull requests are all welcome.

## Development setup

```bash
git clone https://github.com/Kianis534/docugenius-ai-pdf-chatbot.git
cd docugenius-ai-pdf-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, smaller CPU-only build
pip install -r requirements-dev.txt
cp .env.example .env             # then add your GOOGLE_API_KEY
python -m docugenius             # http://127.0.0.1:7860
```

Set `RELOAD=true` to auto-restart the server when you edit code.

## Before you open a pull request

```bash
ruff check .            # lint
ruff format .           # format
pytest                  # tests run offline and never call real APIs
```

* Keep changes focused. One feature or fix per pull request is easiest to review.
* Add or update tests for any behaviour you change. External services (Gemini, Reddit, Google Speech) are
  faked in `tests/conftest.py`, so tests stay fast and deterministic.
* Never commit API keys, `.env` files or personal documents.

## Project layout

| Path | What lives there |
| --- | --- |
| `docugenius/api.py` | HTTP routes (thin: validation and mapping only) |
| `docugenius/services/` | Business logic: RAG, Gemini client, speech, Reddit, ghostwriter, sketch |
| `docugenius/schemas.py` | Pydantic request and response models |
| `docugenius/errors.py` | Domain errors carrying HTTP status codes and user-safe messages |
| `frontend/` | Single-page UI (vanilla JS + Bootstrap, no build step) |
| `tests/` | Pytest suite |

## Commit messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) style, for example
`feat: add DOCX upload support` or `fix: handle encrypted PDFs`.
