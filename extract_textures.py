#!/usr/bin/env python3
"""
texture_extractor — wyciąga embedded textury z FBX / GLB / USDZ,
pokazuje interaktywne okno do edycji nazw przed zapisem.

Użycie:
  python extract_textures.py plik.fbx
  python extract_textures.py plik.glb --out textures/
  python extract_textures.py plik.usdz --no-interactive
"""

import argparse
import os
import re
import readline
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dds", ".exr", ".tga"}
FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class TextureEntry:
    orig_name: str       # oryginalna nazwa w pliku
    suggested: str       # zaproponowana znormalizowana nazwa
    data: bytes          # surowe bajty


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_name(name: str, ext: str) -> str:
    name = name.replace("\x00", "").strip()
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^\w]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_") or "texture"
    orig_ext = Path(name).suffix.lower() or ext
    return stem + orig_ext


def detect_ext(data: bytes) -> str:
    if data[:4] == b"\x89PNG":  return ".png"
    if data[:3] == b"\xff\xd8\xff": return ".jpg"
    if data[:4] == b"DDS ":    return ".dds"
    if data[:2] == b"BM":      return ".bmp"
    if data[:4] in (b"MM\x00*", b"II*\x00"): return ".tif"
    return ".bin"


def resolve_collision(out_dir: Path, name: str) -> str:
    dest = out_dir / name
    if not dest.exists():
        return name
    stem, ext = Path(name).stem, Path(name).suffix
    counter = 1
    while (out_dir / f"{stem}_{counter}{ext}").exists():
        counter += 1
    return f"{stem}_{counter}{ext}"


def write_entries(entries: list[TextureEntry], out_dir: Path) -> dict[str, str]:
    """Zapisuje textury do out_dir. Zwraca mapę {orig_name: final_filename}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rename_map = {}
    for e in entries:
        final = resolve_collision(out_dir, e.suggested)
        (out_dir / final).write_bytes(e.data)
        rename_map[e.orig_name] = final
    return rename_map


# ── interactive rename ────────────────────────────────────────────────────────

def interactive_rename(entries: list[TextureEntry]) -> list[TextureEntry]:
    """
    Pokazuje listę textur z proponowanymi nazwami.
    Użytkownik może edytować każdą nazwę (Enter = zachowaj).
    """
    if not entries:
        return entries

    print("\n" + "─" * 60)
    print(f"  Znaleziono {len(entries)} textur — edytuj nazwy (Enter = zachowaj):")
    print("─" * 60)

    updated = []
    for i, e in enumerate(entries, 1):
        ext = detect_ext(e.data)
        size_kb = len(e.data) // 1024

        print(f"\n  [{i}/{len(entries)}]  oryginalna: {e.orig_name}")
        print(f"             format:    {ext[1:].upper()}  {size_kb} KB")

        # readline pre-fill — użytkownik widzi sugestię i może ją edytować
        def prefill(text):
            def hook():
                readline.insert_text(text)
                readline.redisplay()
            return hook

        readline.set_pre_input_hook(prefill(e.suggested))
        try:
            new_name = input("             nowa nazwa: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Przerwano — używam proponowanych nazw]")
            readline.set_pre_input_hook(None)
            updated.extend(entries[i - 1:])
            return updated
        finally:
            readline.set_pre_input_hook(None)

        if not new_name:
            new_name = e.suggested

        # Upewnij się że ma rozszerzenie
        if not Path(new_name).suffix:
            new_name += ext

        updated.append(TextureEntry(e.orig_name, new_name, e.data))

    print("\n" + "─" * 60)
    return updated


# ── FBX binary parser ─────────────────────────────────────────────────────────

def _read_uint(data, offset, size):
    fmt = {4: "<I", 8: "<Q"}[size]
    return struct.unpack_from(fmt, data, offset)[0], offset + size


def _fbx_parse_node(data, offset, end, version):
    ptr_size = 8 if version >= 7500 else 4
    if offset >= end:
        return None
    end_offset, offset = _read_uint(data, offset, ptr_size)
    num_props, offset = _read_uint(data, offset, ptr_size)
    _prop_list_len, offset = _read_uint(data, offset, ptr_size)
    name_len = data[offset]; offset += 1
    name = data[offset: offset + name_len].decode("utf-8", errors="replace")
    offset += name_len
    if end_offset == 0:
        return None

    props = []
    for _ in range(num_props):
        type_code = chr(data[offset]); offset += 1
        if type_code == "Y":
            val = struct.unpack_from("<h", data, offset)[0]; offset += 2
        elif type_code == "C":
            val = bool(data[offset]); offset += 1
        elif type_code == "I":
            val = struct.unpack_from("<i", data, offset)[0]; offset += 4
        elif type_code == "F":
            val = struct.unpack_from("<f", data, offset)[0]; offset += 4
        elif type_code == "D":
            val = struct.unpack_from("<d", data, offset)[0]; offset += 8
        elif type_code == "L":
            val = struct.unpack_from("<q", data, offset)[0]; offset += 8
        elif type_code in "SR":
            length = struct.unpack_from("<I", data, offset)[0]; offset += 4
            raw = data[offset: offset + length]
            val = raw.decode("utf-8", errors="replace") if type_code == "S" else raw
            offset += length
        elif type_code == "B":
            length = struct.unpack_from("<I", data, offset)[0]; offset += 4
            val = data[offset: offset + length]; offset += length
        else:
            # array types
            arr_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
            encoding = struct.unpack_from("<I", data, offset)[0]; offset += 4
            comp_len = struct.unpack_from("<I", data, offset)[0]; offset += 4
            offset += comp_len
            val = None
        props.append((type_code, val))

    children = []
    while offset < end_offset - (ptr_size * 3 + 1 + 1):
        child = _fbx_parse_node(data, offset, end_offset, version)
        if child is None:
            break
        n, p, c, offset = child
        children.append((n, p, c))

    return name, props, children, end_offset


def _fbx_find_nodes(nodes, target):
    results = []
    for name, props, children in nodes:
        if name == target:
            results.append((name, props, children))
        results.extend(_fbx_find_nodes(children, target))
    return results


def collect_fbx_embedded(fbx_path: Path) -> list[TextureEntry]:
    data = fbx_path.read_bytes()
    if not data.startswith(FBX_MAGIC):
        raise ValueError("Plik nie jest binarnym FBX")

    version = struct.unpack_from("<I", data, 23)[0]
    offset = 27
    end = len(data) - 160
    nodes = []
    while offset < end:
        result = _fbx_parse_node(data, offset, end, version)
        if result is None:
            break
        name, props, children, offset = result
        nodes.append((name, props, children))

    entries = []
    for _, video_props, video_children in _fbx_find_nodes(nodes, "Video"):
        tex_name = None
        for tc, val in video_props:
            if tc == "S" and val:
                tex_name = val.split("\x00")[0].strip()
                break

        for child_name, child_props, _ in video_children:
            if child_name in ("RelativeFilename", "Filename"):
                for tc, val in child_props:
                    if tc == "S" and val:
                        fname = Path(val.replace("\\", "/")).name.replace("\x00", "").strip()
                        if fname:
                            tex_name = fname
                        break

        content_data = None
        for child_name, child_props, _ in video_children:
            if child_name == "Content":
                for tc, val in child_props:
                    if tc in ("R", "B") and val:
                        content_data = val
                        break

        if tex_name and content_data and len(content_data) > 4:
            raw = content_data
            if raw[:4] not in (b"\x89PNG", b"\xff\xd8\xff", b"DDS ", b"BM"):
                raw = raw[4:]
            ext = detect_ext(raw)
            entries.append(TextureEntry(
                orig_name=tex_name,
                suggested=normalize_name(tex_name, ext),
                data=raw,
            ))

    return entries


def collect_fbm(fbm_dir: Path) -> list[TextureEntry]:
    entries = []
    for f in sorted(fbm_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXTS:
            data = f.read_bytes()
            ext = detect_ext(data)
            entries.append(TextureEntry(
                orig_name=f.name,
                suggested=normalize_name(f.name, ext),
                data=data,
            ))
    return entries


# ── GLB ───────────────────────────────────────────────────────────────────────

def collect_glb(glb_path: Path) -> list[TextureEntry]:
    try:
        from pygltflib import GLTF2
    except ImportError:
        raise ImportError("Zainstaluj pygltflib: pip install pygltflib")

    gltf = GLTF2().load(str(glb_path))
    entries = []

    for i, image in enumerate(gltf.images):
        name = image.name or image.uri or f"texture_{i}"
        data = None

        if image.uri and not image.uri.startswith("data:"):
            uri_path = glb_path.parent / image.uri
            if uri_path.exists():
                data = uri_path.read_bytes()
        elif image.bufferView is not None:
            bv = gltf.bufferViews[image.bufferView]
            blob = gltf.binary_blob()
            data = blob[bv.byteOffset: bv.byteOffset + bv.byteLength]

        if data:
            ext = detect_ext(data)
            entries.append(TextureEntry(
                orig_name=name,
                suggested=normalize_name(name, ext),
                data=data,
            ))

    return entries


def patch_glb(glb_path: Path, rename_map: dict, out_dir: Path, out_stem: str | None = None) -> Path:
    try:
        from pygltflib import GLTF2
    except ImportError:
        raise ImportError("Zainstaluj pygltflib: pip install pygltflib")

    gltf = GLTF2().load(str(glb_path))
    for i, image in enumerate(gltf.images):
        name = image.name or image.uri or f"texture_{i}"
        if name in rename_map:
            image.uri = rename_map[name]
            image.bufferView = None
            image.mimeType = None

    stem = out_stem or glb_path.stem
    out_gltf = out_dir / (stem + ".gltf")
    gltf.save(str(out_gltf))
    return out_gltf


# ── USDZ ─────────────────────────────────────────────────────────────────────

def collect_usdz(usdz_path: Path) -> list[TextureEntry]:
    entries = []
    with zipfile.ZipFile(usdz_path, "r") as z:
        for info in z.infolist():
            p = Path(info.filename)
            if p.suffix.lower() in IMAGE_EXTS:
                data = z.read(info.filename)
                ext = detect_ext(data)
                entries.append(TextureEntry(
                    orig_name=p.name,
                    suggested=normalize_name(p.name, ext),
                    data=data,
                ))
    return entries


# ── FBX patch ─────────────────────────────────────────────────────────────────

def patch_fbx(fbx_path: Path, rename_map: dict, out_dir: Path, out_stem: str | None = None) -> Path:
    data = bytearray(fbx_path.read_bytes())
    for orig, new_name in rename_map.items():
        orig_bytes = orig.encode("utf-8")
        new_bytes = new_name.encode("utf-8")
        start = 0
        while True:
            idx = data.find(orig_bytes, start)
            if idx == -1:
                break
            if len(new_bytes) <= len(orig_bytes):
                data[idx: idx + len(orig_bytes)] = new_bytes + b"\x00" * (len(orig_bytes) - len(new_bytes))
            start = idx + len(orig_bytes)

    stem = out_stem or fbx_path.stem
    out_fbx = out_dir / (stem + fbx_path.suffix)
    out_fbx.write_bytes(bytes(data))
    return out_fbx


# ── dispatcher ────────────────────────────────────────────────────────────────

def process(input_path: str, out_dir_override: str | None = None, interactive: bool = True):
    src = Path(input_path).resolve()
    if not src.exists():
        print(f"[ERROR] Plik nie istnieje: {src}")
        return

    out_dir = Path(out_dir_override).resolve() if out_dir_override else src.parent / "textures"
    ext = src.suffix.lower()

    print(f"\nPlik:   {src.name}  ({src.stat().st_size // 1024} KB)")
    print(f"Output: {out_dir}")

    # --- zbierz textury ---
    entries: list[TextureEntry] = []

    if ext == ".fbx":
        entries = collect_fbx_embedded(src)
        if not entries:
            fbm_dir = src.parent / (src.stem + ".fbm")
            if fbm_dir.is_dir():
                print(f"  [FBM] Wykryto zewnętrzne textury: {fbm_dir.name}/")
                entries = collect_fbm(fbm_dir)
            else:
                print("[INFO] Brak embedded textur i brak folderu .fbm/")
                return
    elif ext == ".glb":
        entries = collect_glb(src)
    elif ext == ".usdz":
        entries = collect_usdz(src)
    else:
        print(f"[ERROR] Nieobsługiwany format: {ext}")
        return

    if not entries:
        print("[INFO] Nie znaleziono textur.")
        return

    # --- opcjonalne interaktywne rename ---
    if interactive:
        entries = interactive_rename(entries)

    # --- zapisz ---
    rename_map = write_entries(entries, out_dir)

    for orig, final in rename_map.items():
        print(f"  {orig}  →  {final}")

    # --- zaktualizuj referencje w pliku ---
    if ext == ".fbx":
        patched = patch_fbx(src, rename_map, out_dir)
        print(f"\n[OK] Zapisano: {patched.name}")
    elif ext == ".glb":
        patched = patch_glb(src, rename_map, out_dir)
        print(f"\n[OK] Zapisano: {patched.name}")

    print(f"\nWyciągnięto {len(rename_map)} textur → {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Wyciąga embedded textury z FBX / GLB / USDZ"
    )
    parser.add_argument("input", help="Ścieżka do pliku 3D")
    parser.add_argument("--out", help="Folder wyjściowy (domyślnie: textures/ obok pliku)")
    parser.add_argument("--no-interactive", action="store_true", help="Pomiń okno edycji nazw")
    args = parser.parse_args()
    process(args.input, args.out, interactive=not args.no_interactive)


if __name__ == "__main__":
    main()
