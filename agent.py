# agent.py
print("[BOOT] agent.py v2.1 with debug_post-guard")

import os, sys, io, json, time, argparse, base64
from pathlib import Path
from typing import Optional, Tuple
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


# --- put this right after imports & dotenv load ---
try:
    debug_post  # noqa: F821  # exists?
except NameError:
    import requests
    def debug_post(url: str, *, data=None, json_data=None, files=None, headers=None, timeout=120):
        # minimal fallback (no pretty logs, just a safe POST)
        return requests.post(url, headers=headers, data=data, json=json_data, files=files, timeout=timeout)
# --- end fallback ---



# =========================
# Config
# =========================
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
MODEL           = os.getenv("IMAGE_MODEL", "gpt-image-1").strip()
IMAGE_SIZE      = os.getenv("IMAGE_SIZE", "1024x1024").strip()     # edits denemesi için (kare önerilir)
FINAL_SIZE      = os.getenv("FINAL_SIZE", "2048x1024").strip()     # nihai çıktı boyutu
HTTP_TIMEOUT    = int(os.getenv("HTTP_TIMEOUT", "120"))
RETRY_MAX       = int(os.getenv("RETRY_MAX", "3"))
RETRY_BACKOFF   = float(os.getenv("RETRY_BACKOFF", "1.6"))
FORCE_SQUARE_FOR_EDITS = os.getenv("FORCE_SQUARE_FOR_EDITS", "1") in ("1","true","True")

if not OPENAI_API_KEY:
    print("[FATAL] OPENAI_API_KEY missing", file=sys.stderr)
    sys.exit(1)

BASE_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

# Output kökü: varsa volume'a yaz
OUT_BASE = Path("/mnt/data/output") if Path("/mnt/data").exists() else Path("output")


# =========================
# Utils
# =========================
def parse_size(s: str) -> Tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)

def image_to_png_bytes_rgb(path: Path) -> bytes:
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def pad_to_square(png_bytes: bytes, fill=(255,255,255)) -> bytes:
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    s = max(im.width, im.height)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(im, ((s - im.width)//2, (s - im.height)//2))
    out = io.BytesIO()
    canvas.save(out, format="PNG", optimize=True)
    return out.getvalue()

def resize_to(png_bytes: bytes, size: str) -> bytes:
    w, h = parse_size(size)
    im = Image.open(io.BytesIO(png_bytes))
    im = im.resize((w, h), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()

def b64_to_png(b64: str) -> bytes:
    return base64.b64decode(b64)


# =========================
# Debug POST wrapper  ✅
# =========================
def _mask(token: str, visible=6) -> str:
    try:
        return token[:visible] + "…" if len(token) > visible else token
    except Exception:
        return "…"

def debug_post(url: str, *, data: dict=None, json_data: dict=None,
               files: dict=None, headers: dict=None, timeout:int=HTTP_TIMEOUT):
    """requests.post için güvenli loglayıcı (Authorization maskeli)."""
    log_headers = dict(headers or {})
    if "Authorization" in log_headers:
        try:
            log_headers["Authorization"] = "Bearer " + _mask(log_headers["Authorization"].split()[-1])
        except Exception:
            log_headers["Authorization"] = "Bearer …"
    print("\n[HTTP] POST", url)
    print("[HTTP] headers:", json.dumps(log_headers, indent=2))
    if json_data is not None:
        pr = dict(json_data)
        if "prompt" in pr and isinstance(pr["prompt"], str):
            pr["prompt"] = pr["prompt"][:140] + "…(trim)"
        print("[HTTP] json:", json.dumps(pr, indent=2, ensure_ascii=False))
    if data is not None:
        pr = {}
        for k, v in data.items():
            if k == "prompt" and isinstance(v, str):
                pr[k] = v[:140] + "…(trim)"
            else:
                pr[k] = v
        print("[HTTP] form-data:", json.dumps(pr, indent=2, ensure_ascii=False))
    if files:
        meta = {k: {"name": v[0], "mime": v[2], "size": (len(v[1]) if isinstance(v[1], (bytes, bytearray)) else "stream")}
                for k, v in files.items()}
        print("[HTTP] files:", json.dumps(meta, indent=2))
    resp = requests.post(url, headers=headers, data=data, json=json_data, files=files, timeout=timeout)
    print("[HTTP] status:", resp.status_code)
    try:
        body = resp.json()
        print("[HTTP] body:", json.dumps(body, indent=2)[:2000], "…")
    except Exception:
        print("[HTTP] body(raw) len:", len(resp.content))
    return resp


# =========================
# OpenAI Images
# =========================
def images_generate(prompt: str, size: str, n:int=1) -> dict:
    """Generations endpoint JSON bekler. Kare dışı boyut 400 verebildiği için 1024x1024 sabitliyoruz."""
    url = f"{OPENAI_API_BASE}/images/generations"
    gen_size = "1024x1024"  # sabit
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "size": gen_size,
        "n": n,
        "response_format": "b64_json",
    }
    headers = {**BASE_HEADERS, "Content-Type": "application/json"}
    resp = debug_post(url, json_data=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Generate failed: {resp.status_code} {resp.text[:400]}")
    return resp.json()

def images_edit(image_bytes: bytes, prompt: str, size: str, n:int=1, mask_bytes: Optional[bytes]=None) -> dict:
    """Edits endpoint multipart/form-data ister. Content-Type'ı requests ayarlasın (elle eklemiyoruz)."""
    url = f"{OPENAI_API_BASE}/images/edits"
    files = {"image": ("input.png", image_bytes, "image/png")}
    if mask_bytes is not None:
        files["mask"] = ("mask.png", mask_bytes, "image/png")
    form = {
        "model": MODEL,
        "prompt": prompt,
        "size": size,
        "n": n,
        "response_format": "b64_json",
    }
    resp = debug_post(url, data=form, files=files, headers=BASE_HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Edits failed: {resp.status_code} {resp.text[:400]}")
    return resp.json()


# =========================
# Pipeline
# =========================
def run_one(src: Path, out_dir: Path, prompt: str):
    print(f"\n[RUN] {src.name}")
    png_bytes = image_to_png_bytes_rgb(src)
    print(f"[INFO] re-encoded PNG size: {len(png_bytes)} bytes")

    # Edits için kare / boyut hazırlığı
    edit_size = IMAGE_SIZE
    if FORCE_SQUARE_FOR_EDITS:
        w, h = parse_size(IMAGE_SIZE)
        if w != h:
            print(f"[WARN] IMAGE_SIZE={IMAGE_SIZE} kare değil; edits'i 1024x1024 ile deneyeceğim.")
            edit_size = "1024x1024"
            png_sq = pad_to_square(png_bytes)
        else:
            png_sq = png_bytes
    else:
        png_sq = png_bytes

    # 1) Edits dene (retry)
    result = None
    last_err = None
    for attempt in range(1, RETRY_MAX+1):
        try:
            print(f"[TRY] edits attempt {attempt}/{RETRY_MAX} size={edit_size}")
            r = images_edit(png_sq, prompt=prompt, size=edit_size, n=1)
            b64 = r["data"][0]["b64_json"]
            out_png = b64_to_png(b64)
            print("[OK] edits succeeded.")
            result = out_png
            break
        except Exception as e:
            last_err = e
            print(f"[ERR] edits failed: {e}")
            if attempt < RETRY_MAX:
                wait = RETRY_BACKOFF ** attempt
                print(f"[WAIT] backoff {wait:.2f}s …")
                time.sleep(wait)

    # 2) Fallback: generations (1024x1024)
    if result is None:
        print("[FALLBACK] switching to generations …")
        r = images_generate(prompt=prompt, size="1024x1024", n=1)
        b64 = r["data"][0]["b64_json"]
        result = b64_to_png(b64)
        print("[OK] generate succeeded.")

    # 3) Nihai ölçüye ölçekle
    final_png = resize_to(result, FINAL_SIZE)

    # 4) Kaydet
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}__redesign.png"
    with open(out_path, "wb") as f:
        f.write(final_png)
    print(f"[SAVE] {out_path} ({len(final_png)} bytes)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=str, default="input")
    ap.add_argument("--output", type=str, default=str(OUT_BASE))
    ap.add_argument("--prompt", type=str, required=False, default=(
        "Freepik style, clean modern mobile UI redesign, keep original layout, spacing and hierarchy exactly; "
        "change color palette, icons, images and all texts; final 2:1 canvas; "
        "realistic device mockups with glossy highlights, soft shadows, crisp typography, "
        "cohesive Freepik-inspired color grading, vector icons"
    ))
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    if not in_dir.exists():
        print(f"[FATAL] Input folder not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    images = [p for p in in_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    print(f"[INFO] found {len(images)} image(s) in input")
    if not images:
        return

    for p in images:
        try:
            run_one(p, out_dir, args.prompt)
        except Exception as e:
            print(f"[ERROR] {p.name} failed: {e}")

if __name__ == "__main__":
    main()
