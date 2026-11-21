"""Run the development server with ``python -m docugenius``."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "docugenius.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "7860")),
        reload=os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
