import os, sys, io, json, time, base64, argparse, hashlib
from pathlib import Path
from typing import Tuple, Optional
import requests
from PIL import Image, ImageOps

from dotenv import load_dotenv
load_dotenv()

# -------- Config --------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE  = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
MODEL            = os.getenv("IMAGE_MODEL", "gpt-image-1").strip()
IMAGE_SIZE       = os.getenv("IMAGE_SIZE", "1024x1024").strip()        # edits için önerilen kare
FINAL_SIZE       = os.getenv("FINAL_SIZE", "2048x1024").strip()        # kaydedilecek nihai boyut (örn. 2:1)
TIMEOUT          = int(os.getenv("HTTP_TIMEOUT", "120"))
RETRY_MAX        = int(os.getenv("RETRY_MAX", "3"))
RETRY_BACKOFF    = float(os.getenv("RETRY_BACKOFF", "1.6"))
FORCE_SQUARE_FOR_EDITS = os.getenv("FORCE_SQUARE_FOR_EDITS", "1") in ("1","true","True")

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

if not OPENAI_API_KEY:
    print("[FATAL] OPENAI_API_KEY missing in .env", file=sys.stderr)
    sys.exit(1)

# -------- Utilities --------
def parse_size(size_str: str) -> Tuple[int,int]:
    w, h = size_str.lower().split("x")
    return int(w), int(h)

def image_to_png_bytes_rgb(path: Path) -> bytes:
    """Alpha dahil sorunları önlemek için görüntüyü RGB'ye çevirip temiz PNG olarak encode et."""
    im = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def pad_to_square(png_bytes: bytes, fill=(255,255,255)) -> bytes:
    """Gerektiğinde edits endpoint’i için kare pad’leme (beyaz zemin)."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    s = max(im.width, im.height)
    canvas = Image.new("RGB", (s, s), fill)
    x = (s - im.width)//2
    y = (s - im.height)//2
    canvas.paste(im, (x,y))
    out = io.BytesIO()
    canvas.save(out, format="PNG", optimize=True)
    return out.getvalue()

def resize_to(png_bytes: bytes, size_str: str) -> bytes:
    w, h = parse_size(size_str)
    im = Image.open(io.BytesIO(png_bytes))
    im = im.resize((w,h), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()

def b64_to_png_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)

def safe_mask(s: str, visible=6) -> str:
    return s[:visible] + "…" if len(s) > visible else s

# -------- Debug wrapper --------
def images_generate(prompt: str, size: str, n:int=1) -> dict:
    url = f"{OPENAI_API_BASE}/images/generations"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "size": size,
        "n": n,
        "response_format": "b64_json",
    }
    headers = {**HEADERS, "Content-Type": "application/json"}
    resp = debug_post(url, json_data=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Generate failed: {resp.status_code} {resp.text[:300]}")
    return resp.json()


# -------- OpenAI Images (raw HTTP) --------
def images_generate(prompt: str, size: str, n:int=1) -> dict:
    url = f"{OPENAI_API_BASE}/images/generations"
    # Bu endpointte kare dışı boyutlar 400 veriyor; kare sabitliyoruz.
    gen_size = "1024x1024"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "size": gen_size,
        "n": n,
        "response_format": "b64_json",
    }
    headers = {**HEADERS, "Content-Type": "application/json"}
    resp = debug_post(url, json_data=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Generate failed: {resp.status_code} {resp.text[:300]}")
    return resp.json()


def images_edit(image_bytes: bytes, prompt: str, size: str, n:int=1, mask_bytes: Optional[bytes]=None) -> dict:
    url = f"{OPENAI_API_BASE}/images/edits"
    files = {
        "image": ("input.png", image_bytes, "image/png"),
    }
    if mask_bytes is not None:
        files["mask"] = ("mask.png", mask_bytes, "image/png")
    data = {
        "model": MODEL,
        "prompt": prompt,
        "size": size,
        "n": n,
        "response_format": "b64_json",
    }
    resp = debug_post(url, data=data, files=files, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Edits failed: {resp.status_code} {resp.text[:300]}")
    return resp.json()

# -------- Pipeline --------
def run_one_image(src_path: Path, out_dir: Path, prompt: str):
    print(f"\n[RUN] {src_path.name}")

    # 1) PNG’yi güvenli biçimde yeniden encode et
    png_bytes = image_to_png_bytes_rgb(src_path)
    print(f"[INFO] re-encoded PNG size: {len(png_bytes)} bytes")

    # 2) Edits için gerekirse kare pad
    edit_size = IMAGE_SIZE
    if FORCE_SQUARE_FOR_EDITS:
        w,h = parse_size(IMAGE_SIZE)
        if w != h:
            print(f"[WARN] IMAGE_SIZE={IMAGE_SIZE} kare değil → edits için square pad etkin.")
            edit_size = "1024x1024"  # edits kare boyut (genelde en uyumlusu)
            png_bytes_sq = pad_to_square(png_bytes)
        else:
            png_bytes_sq = png_bytes
    else:
        png_bytes_sq = png_bytes

    # 3) edits→retry with backoff
    last_err = None
    for attempt in range(1, RETRY_MAX+1):
        try:
            print(f"[TRY] edits attempt {attempt}/{RETRY_MAX} size={edit_size}")
            result = images_edit(png_bytes_sq, prompt=prompt, size=edit_size, n=1)
            img_b64 = result["data"][0]["b64_json"]
            out_png = b64_to_png_bytes(img_b64)
            print("[OK] edits succeeded.")
            break
        except Exception as e:
            last_err = e
            print(f"[ERR] edits failed: {e}")
            if attempt < RETRY_MAX:
                sleep_s = RETRY_BACKOFF ** attempt
                print(f"[WAIT] backoff {sleep_s:.2f}s …")
                time.sleep(sleep_s)
            else:
                print("[FALLBACK] switching to generations…")
                result = None

    # 4) Fallback generate (aynı prompt, FINAL_SIZE’ı deneyelim; olmazsa 1024x1024)
    if result is None:
        try:
            print(f"[TRY] generate size={FINAL_SIZE}")
            result = images_generate(prompt=prompt, size=FINAL_SIZE, n=1)
        except Exception as ge:
            print(f"[ERR] generate failed at FINAL_SIZE: {ge}")
            print("[TRY] generate 1024x1024 connectivity test")
            result = images_generate(prompt=prompt, size="1024x1024", n=1)

        img_b64 = result["data"][0]["b64_json"]
        out_png = b64_to_png_bytes(img_b64)
        print("[OK] generate succeeded.")

    # 5) Nihai boyuta ölçekle (layout kırpmadan)
    final_png = resize_to(out_png, FINAL_SIZE)

    # 6) Kaydet
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src_path.stem
    out_path = out_dir / f"{stem}__redesign.png"
    with open(out_path, "wb") as f:
        f.write(final_png)
    print(f"[SAVE] {out_path} ({len(final_png)} bytes)")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=str, default="input", help="Girdi klasörü (PNG)")
    ap.add_argument("--output", type=str, default="output", help="Çıktı klasörü")
    ap.add_argument("--prompt", type=str, required=False, default=(
        "Freepik style, clean modern mobile UI redesign, keep original layout, spacing and hierarchy, "
        "change color palette, icons, images and texts, professional app screens, crisp typography, "
        "realistic device bezels, soft shadows, cohesive visual system"
    ))
    args = ap.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.exists():
        print(f"[FATAL] Input folder not found: {in_dir}")
        sys.exit(1)

    images = [p for p in in_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    if not images:
        print("[WARN] No images found in input/.")
        return

    print(f"[INFO] Found {len(images)} images in {in_dir}")
    for p in images:
        try:
            run_one_image(p, out_dir, prompt=args.prompt)
        except Exception as e:
            print(f"[ERROR] {p.name} failed: {e}")

if __name__ == "__main__":
    main()
