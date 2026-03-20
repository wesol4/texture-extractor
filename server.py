#!/usr/bin/env python3
"""
Texture Extractor — static file server.
Całe przetwarzanie odbywa się w przeglądarce (JavaScript).

Uruchomienie:
  python server.py
  uvicorn server:app --reload
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Texture Extractor")


@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path(__file__).parent / "static" / "index.html").read_text()


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=True)
