# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report them privately through
[GitHub Security Advisories](https://github.com/Kianis534/docugenius-ai-pdf-chatbot/security/advisories/new).
You can expect an initial response within a few days.

## Handling secrets

* API keys are read from environment variables or a local `.env` file, which is git-ignored. They are never
  hard-coded or sent to the browser.
* If you think a key has leaked (for example, it was committed by mistake), **revoke it immediately** in
  [Google AI Studio](https://aistudio.google.com/app/apikey) or the [Reddit app settings](https://www.reddit.com/prefs/apps)
  and create a new one. Removing the commit alone is not enough.

## Deployment hardening checklist

* Set `ALLOWED_ORIGINS` to your own domain instead of `*` when the API is public.
* Set `GEMINI_DAILY_REQUEST_LIMIT` to cap spending on a shared key.
* Put the service behind HTTPS (a reverse proxy, Hugging Face Spaces, Cloud Run, etc.).
* Uploaded PDFs are stored under `DATA_DIR`. Use a private volume and clean it up regularly.
