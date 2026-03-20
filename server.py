#!/usr/bin/env python3
"""
FastAPI backend dla texture_extractor.

Uruchomienie:
  python server.py
  # lub
  uvicorn server:app --reload
"""

import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from extract_textures import (
    TextureEntry,
    collect_fbm,
    collect_fbx_embedded,
    collect_glb,
    collect_usdz,
    detect_ext,
    normalize_name,
    patch_fbx,
    patch_glb,
    write_entries,
)

app = FastAPI(title="Texture Extractor")

# Sesje: {session_id: {entries, src_path, tmp_dir, ext}}
SESSIONS: dict[str, dict[str, Any]] = {}

SUPPORTED = {".fbx", ".glb", ".usdz"}
SUPPORTED_IN_ZIP = {".fbx", ".glb", ".usdz"}


# ── ZIP extraction ───────────────────────────────────────────────────────────

def extract_zip_to_tmp(zip_path: Path, tmp_dir: Path) -> list[Path]:
    """
    Wypakowuje ZIP do tmp_dir, zwraca listę znalezionych plików 3D.
    Obsługuje też towarzyszące foldery .fbm wewnątrz ZIPa.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    found = []
    for p in sorted(tmp_dir.rglob("*")):
        if p.suffix.lower() in SUPPORTED_IN_ZIP and p.is_file():
            found.append(p)
    return found


def build_session_entries(src: Path, ext: str):
    if ext == ".fbx":
        entries = collect_fbx_embedded(src)
        if not entries:
            fbm_dir = src.parent / (src.stem + ".fbm")
            entries = collect_fbm(fbm_dir) if fbm_dir.is_dir() else []
    elif ext == ".glb":
        entries = collect_glb(src)
    elif ext == ".usdz":
        entries = collect_usdz(src)
    else:
        entries = []
    return entries


def make_session_response(session_id: str, entries, filename: str, src_dir: str | None = None):
    suffixes = detect_suffixes(entries)
    resp = {
        "session_id": session_id,
        "filename": filename,
        "texture_count": len(entries),
        "textures": [
            {
                "id": i,
                "orig_name": e.orig_name,
                "suggested": e.suggested,
                "format": detect_ext(e.data)[1:].upper(),
                "size_kb": len(e.data) // 1024,
                "suffix": suffixes[i],
            }
            for i, e in enumerate(entries)
        ],
    }
    if src_dir:
        resp["src_dir"] = src_dir
    return resp


# ── suffix detection ──────────────────────────────────────────────────────────

# Znane sufiksy PBR — kolejność ważna (dłuższe pierwsze)
PBR_KEYWORDS = [
    "basecolor", "base_color", "albedo", "diffuse", "color",
    "normal", "nrm", "nml",
    "roughness", "rough",
    "metallic", "metalness", "metal",
    "specular", "spec",
    "ambient_occlusion", "ao", "occlusion",
    "emissive", "emission",
    "height", "displacement", "bump",
    "opacity", "alpha", "mask",
    "subsurface", "sss",
    "transmission",
]


def extract_pbr_suffix(stem: str) -> str | None:
    """Wyciąga znany sufiks PBR z nazwy pliku, np. 'char_roughness' → '_roughness'."""
    lower = stem.lower()
    for kw in PBR_KEYWORDS:
        # Sprawdź czy stem kończy się na _keyword lub -keyword
        if lower.endswith("_" + kw) or lower.endswith("-" + kw):
            return "_" + kw
        # Lub czy keyword jest ostatnim 'słowem' po separatorze
        if lower == kw:
            return ""
    return None


def detect_suffixes(entries: list) -> list[str]:
    """
    Wykrywa sufiksy textur.
    Najpierw próbuje rozpoznać znane słowa PBR (_roughness, _metallic itd.),
    jeśli nie — używa wspólnego prefiksu nazw.
    """
    if not entries:
        return []

    stems = [Path(e.suggested).stem for e in entries]

    # Próba 1: rozpoznanie znanych PBR keywords
    pbr = [extract_pbr_suffix(s) for s in stems]
    if any(p is not None for p in pbr):
        # Użyj PBR jeśli przynajmniej część rozpoznana;
        # nierozpoznane dostaną indeks (_0, _1, ...)
        result = []
        unrecognized_idx = 0
        for s, p in zip(stems, pbr):
            if p is not None:
                result.append(p)
            else:
                result.append(f"_{unrecognized_idx}")
                unrecognized_idx += 1
        return result

    # Próba 2: wspólny prefiks
    common = os.path.commonprefix(stems)
    remaining = [s[len(common):] for s in stems]
    if not all(r == "" or r.startswith("_") for r in remaining):
        if "_" in common:
            common = common[:common.rfind("_") + 1]
        else:
            common = ""

    suffixes = [s[len(common):] for s in stems]
    return [("_" + s if s and not s.startswith("_") else s) for s in suffixes]


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    accepted = SUPPORTED | {".zip"}
    if ext not in accepted:
        raise HTTPException(400, f"Nieobsługiwany format: {ext}. Obsługiwane: {', '.join(sorted(accepted))}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="texext_"))
    uploaded = tmp_dir / file.filename
    uploaded.write_bytes(await file.read())

    try:
        if ext == ".zip":
            found = extract_zip_to_tmp(uploaded, tmp_dir / "zip_contents")
            if not found:
                raise HTTPException(400, "ZIP nie zawiera pliku FBX / GLB / USDZ")
            if len(found) > 1:
                # Zwróć listę do wyboru
                return {
                    "zip_choice": True,
                    "tmp_dir": str(tmp_dir),
                    "files": [{"name": f.name, "path": str(f), "ext": f.suffix.lower()} for f in found],
                }
            src = found[0]
            ext = src.suffix.lower()
        else:
            src = uploaded

        entries = build_session_entries(src, ext)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, str(e))

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "entries": entries, "src": src, "src_dir": None,
        "tmp_dir": tmp_dir, "ext": ext, "filename": src.name,
    }
    return make_session_response(session_id, entries, src.name)


class RenameRequest(BaseModel):
    textures: list[dict]  # [{id, new_name}, ...]
    output_dir: str | None = None
    base_name: str | None = None  # nazwa podfolderu i pliku (np. "glowa")


@app.post("/export/{session_id}")
async def export(session_id: str, req: RenameRequest):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Sesja nie istnieje lub wygasła")

    session = SESSIONS[session_id]
    entries: list[TextureEntry] = session["entries"]
    src: Path = session["src"]
    tmp_dir: Path | None = session["tmp_dir"]
    ext: str = session["ext"]
    model_stem = Path(session["filename"]).stem
    folder_name = req.base_name.strip() if req.base_name and req.base_name.strip() else model_stem

    # Jeśli plik był wgrany przez upload, tmp_dir istnieje; przez pick-source — nie
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="texext_"))
        session["tmp_dir"] = tmp_dir

    # Zastosuj nowe nazwy
    name_map = {item["id"]: item["new_name"] for item in req.textures}
    renamed = []
    for i, e in enumerate(entries):
        new_name = name_map.get(i, e.suggested).strip() or e.suggested
        if not Path(new_name).suffix:
            new_name += detect_ext(e.data)
        renamed.append(TextureEntry(e.orig_name, new_name, e.data))

    # Ustal folder wyjściowy — zawsze podfolder o nazwie bazowej
    use_custom_dir = bool(req.output_dir and req.output_dir.strip())
    if use_custom_dir:
        base_out = Path(req.output_dir.strip()) / folder_name
        try:
            base_out.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise HTTPException(400, f"Nie można utworzyć folderu: {e}")
        tex_out = base_out
    else:
        tex_out = tmp_dir / folder_name

    rename_map = write_entries(renamed, tex_out) if renamed else {}

    # Zaktualizuj referencje w pliku źródłowym (lub po prostu skopiuj jeśli brak textur)
    patched_model = None
    if ext == ".fbx":
        patched_model = patch_fbx(src, rename_map, tex_out, out_stem=folder_name)
    elif ext == ".glb":
        patched_model = patch_glb(src, rename_map, tex_out, out_stem=folder_name)
    elif not patched_model and src.exists():
        # Pozostałe formaty lub brak textur — skopiuj oryginał z nową nazwą
        dest = tex_out / (folder_name + src.suffix)
        tex_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        patched_model = dest

    if use_custom_dir:
        # Zapis na dysk — zwróć ścieżkę
        saved_files = [f.name for f in tex_out.iterdir() if f.is_file()]
        return {
            "saved_to": str(tex_out),
            "files": sorted(saved_files),
        }
    else:
        # Spakuj do ZIP
        zip_path = tmp_dir / "output.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for tex_file in tex_out.iterdir():
                if tex_file.is_file():
                    zf.write(tex_file, tex_file.name)
            if patched_model and patched_model.exists():
                zf.write(patched_model, patched_model.name)

        session["zip_path"] = zip_path
        return {"download_url": f"/download/{session_id}"}


@app.get("/download/{session_id}")
async def download(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Sesja nie istnieje")
    session = SESSIONS[session_id]
    zip_path = session.get("zip_path")
    if not zip_path or not zip_path.exists():
        raise HTTPException(404, "Plik nie gotowy — najpierw wywołaj /export")

    orig_stem = Path(session["filename"]).stem
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{orig_stem}_textures.zip",
    )


@app.post("/select-from-zip")
async def select_from_zip(body: dict):
    """Gdy ZIP zawiera wiele plików 3D, user wybiera który."""
    path_str = body.get("path")
    tmp_dir_str = body.get("tmp_dir")
    if not path_str:
        raise HTTPException(400, "Brak ścieżki")

    src = Path(path_str)
    if not src.exists():
        raise HTTPException(400, "Plik nie istnieje")

    ext = src.suffix.lower()
    try:
        entries = build_session_entries(src, ext)
    except Exception as e:
        raise HTTPException(500, str(e))

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "entries": entries, "src": src, "src_dir": None,
        "tmp_dir": Path(tmp_dir_str) if tmp_dir_str else src.parent,
        "ext": ext, "filename": src.name,
    }
    return make_session_response(session_id, entries, src.name)


@app.get("/browse-folder")
async def browse_folder():
    """Otwiera natywny dialog wyboru folderu output."""
    try:
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory", "--title=Wybierz folder output"],
            capture_output=True, text=True, timeout=60
        )
        path = result.stdout.strip()
        return {"path": path if result.returncode == 0 and path else None}
    except Exception as e:
        raise HTTPException(500, f"Błąd dialogu: {e}")


@app.post("/pick-source")
async def pick_source():
    """
    Otwiera dialog wyboru pliku źródłowego (FBX/GLB/USDZ),
    wczytuje go i zwraca sesję — tak jak /upload, ale ze znajomością oryginalnej ścieżki.
    """
    try:
        result = subprocess.run(
            ["zenity", "--file-selection",
             "--title=Wybierz plik 3D",
             "--file-filter=Pliki 3D | *.fbx *.FBX *.glb *.GLB *.usdz *.USDZ *.zip *.ZIP"],
            capture_output=True, text=True, timeout=60
        )
        src_path_str = result.stdout.strip()
        if result.returncode != 0 or not src_path_str:
            return {"cancelled": True}
    except Exception as e:
        raise HTTPException(500, f"Błąd dialogu: {e}")

    src = Path(src_path_str)
    if not src.exists():
        raise HTTPException(400, f"Plik nie istnieje: {src}")

    ext = src.suffix.lower()
    src_dir = str(src.parent)

    tmp_dir = None
    zip_filename = None
    try:
        if ext == ".zip":
            zip_filename = src.name
            tmp_dir = Path(tempfile.mkdtemp(prefix="texext_"))
            found = extract_zip_to_tmp(src, tmp_dir / "zip_contents")
            if not found:
                raise HTTPException(400, "ZIP nie zawiera pliku FBX / GLB / USDZ")
            if len(found) > 1:
                return {
                    "zip_choice": True,
                    "zip_filename": zip_filename,
                    "tmp_dir": str(tmp_dir),
                    "src_dir": src_dir,
                    "files": [{"name": f.name, "path": str(f), "ext": f.suffix.lower()} for f in found],
                }
            src = found[0]
            ext = src.suffix.lower()
        elif ext not in SUPPORTED:
            raise HTTPException(400, f"Nieobsługiwany format: {ext}")

        entries = build_session_entries(src, ext)
    except HTTPException:
        raise
    except Exception as e:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, str(e))

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "entries": entries, "src": src, "src_dir": src_dir,
        "tmp_dir": tmp_dir, "ext": ext, "filename": src.name,
    }
    resp = make_session_response(session_id, entries, src.name, src_dir)
    if zip_filename:
        resp["zip_filename"] = zip_filename
    return resp


@app.post("/set-src-dir/{session_id}")
async def set_src_dir(session_id: str):
    """Otwiera dialog wyboru folderu źródłowego i zapisuje go w sesji."""
    if session_id not in SESSIONS:
        raise HTTPException(404, "Sesja nie istnieje")
    try:
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory", "--title=Wskaż folder z plikiem źródłowym"],
            capture_output=True, text=True, timeout=60
        )
        path = result.stdout.strip()
        if result.returncode == 0 and path:
            SESSIONS[session_id]["src_dir"] = path
            return {"src_dir": path}
        return {"src_dir": None}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/resolve-path")
async def resolve_path(path: str, session_id: str | None = None):
    """Rozwiązuje ścieżkę względem katalogu pliku źródłowego (jeśli znany)."""
    stripped = path.strip()

    # Ścieżka absolutna — Linux lub Windows
    if stripped.startswith("/") or (len(stripped) > 2 and stripped[1] == ":"):
        return {"resolved": stripped, "relative": False}

    # Relatywna — rozwiąż względem src_dir jeśli znane, inaczej home
    base_dir = Path.home()
    src_dir_known = False
    if session_id and session_id in SESSIONS:
        src_dir = SESSIONS[session_id].get("src_dir")
        if src_dir:
            base_dir = Path(src_dir)
            src_dir_known = True

    resolved = str((base_dir / stripped).resolve())
    return {"resolved": resolved, "relative": True, "src_dir_known": src_dir_known}


@app.get("/preview/{session_id}/{texture_id}")
async def preview(session_id: str, texture_id: int):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Sesja nie istnieje")
    entries = SESSIONS[session_id]["entries"]
    if texture_id >= len(entries):
        raise HTTPException(404, "Brak textury")
    e = entries[texture_id]
    ext = detect_ext(e.data)
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".bmp": "image/bmp",
            ".tif": "image/tiff", ".dds": "image/vnd.ms-dds"}.get(ext, "application/octet-stream")
    from fastapi.responses import Response
    return Response(content=e.data, media_type=mime)


@app.get("/", response_class=HTMLResponse)
async def root():
    index = Path(__file__).parent / "static" / "index.html"
    return index.read_text()


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=True)
