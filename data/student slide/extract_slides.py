"""
Extract slide images and text from student PPTX files.
For full-slide images (Google Slides flattened export), an AI vision model
is used via OpenRouter to extract the chart description and explanation.

Output:
  extracted/<pptx_name>/slide001.png, slide002.png, ...
  extracted_slides.csv  (columns: pptx_file, slide_num, img_path, img_type,
                                   explanation, ai_chart, ai_explanation)

  img_type values:
    chart      — image smaller than full slide (an actual chart/plot)
    full_slide — image covers the whole slide (Google Slides flattened export)
    no_image   — slide has no embedded image

Usage:
  pip install python-pptx Pillow openai python-dotenv
  python extract_slides.py
  python extract_slides.py --model google/gemini-2.0-flash-001
"""

import argparse
import base64
import csv
import json
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from PIL import Image
import os

load_dotenv(Path(__file__).parent.parent.parent / "chart_verifier" / ".env")

# ── Paths & config ─────────────────────────────────────────────────────────────

SLIDE_DIR  = Path(__file__).parent
OUTPUT_DIR = SLIDE_DIR / "extracted"
CSV_PATH   = SLIDE_DIR / "extracted_slides.csv"

FULL_SLIDE_THRESHOLD = 0.90
DEFAULT_MODEL        = "anthropic/claude-sonnet-4.6"

AI_PROMPT = """You are analyzing a presentation slide image.

Extract two things:

1. plot_bbox: The bounding box of the chart or plot in the image, as percentages (0.0–1.0) of image width/height.
   Format: {"x1": left, "y1": top, "x2": right, "y2": bottom}
   If no chart is visible, set to null.

2. explanation: All explanatory or descriptive text written on the slide (titles, labels, bullet points, captions).
   If no text is present, set to null.

Return ONLY a JSON object, no markdown:
{
  "plot_bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
  "explanation": "<all text from the slide>"
}"""


# ── OpenRouter client ──────────────────────────────────────────────────────────

def make_client() -> OpenAI:
    key = os.getenv("OPENROUTER_API_KEY", "")
    assert key, "Set OPENROUTER_API_KEY in chart_verifier/.env"
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


# ── Utilities ──────────────────────────────────────────────────────────────────

def relative_path(path: Path, anchor: str = "Github") -> str:
    for i, part in enumerate(path.parts):
        if part == anchor:
            return "/".join(path.parts[i:])
    return str(path)


def safe_stem(name: str, max_len: int = 40) -> str:
    return re.sub(r"[^\w]", "_", name)[:max_len]


def save_as_png(blob: bytes, src_ext: str, dest: Path) -> Path:
    dest.write_bytes(blob)
    if src_ext.lower() != "png":
        png = dest.with_suffix(".png")
        with Image.open(dest) as im:
            im.save(png, "PNG")
        dest.unlink()
        return png
    return dest


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def extract_text(slide) -> str:
    shapes = sorted(slide.shapes, key=lambda s: (s.top or 0, s.left or 0))
    lines = [
        shape.text_frame.text.strip()
        for shape in shapes
        if shape.has_text_frame and shape.text_frame.text.strip()
    ]
    return " | ".join(lines)


def image_type(shape, slide_w, slide_h) -> str:
    w_ratio = shape.width  / slide_w
    h_ratio = shape.height / slide_h
    return "full_slide" if (w_ratio >= FULL_SLIDE_THRESHOLD and
                            h_ratio >= FULL_SLIDE_THRESHOLD) else "chart"


# ── AI extraction for full-slide images ───────────────────────────────────────

def crop_chart(img_path: Path, bbox: dict) -> Path:
    """Crop the chart region from the slide image and save as a new PNG."""
    with Image.open(img_path) as im:
        w, h   = im.size
        left   = int(bbox["x1"] * w)
        top    = int(bbox["y1"] * h)
        right  = int(bbox["x2"] * w)
        bottom = int(bbox["y2"] * h)
        cropped = im.crop((left, top, right, bottom))
        chart_path = img_path.with_name(img_path.stem + "_chart.png")
        cropped.save(chart_path, "PNG")
    return chart_path


def ai_extract(img_path: Path, client: OpenAI, model: str) -> dict:
    img_b64 = encode_image(img_path)
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": AI_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
        )
        raw = resp.choices[0].message.content or ""
        obj = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])

        bbox        = obj.get("plot_bbox")
        chart_path  = crop_chart(img_path, bbox) if bbox else None

        return {
            "ai_chart_path": relative_path(chart_path) if chart_path else None,
            "ai_explanation": obj.get("explanation"),
        }
    except Exception as e:
        return {"ai_chart_path": None, "ai_explanation": f"[error: {e}]"}


# ── Core extraction ────────────────────────────────────────────────────────────

def extract_images(slide, slide_w, slide_h, out_dir: Path, prefix: str) -> list[dict]:
    results = []

    def process(shape, label):
        ext   = (shape.image.ext or "png").lower().lstrip(".")
        dest  = out_dir / f"{prefix}_{label}.{ext}"
        saved = save_as_png(shape.image.blob, ext, dest)
        results.append({
            "img_path": relative_path(saved),
            "img_type": image_type(shape, slide_w, slide_h),
            "_abs_path": saved,
        })

    for idx, shape in enumerate(slide.shapes):
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            process(shape, f"img{idx:02d}")
        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for cidx, child in enumerate(shape.shapes):
                if child.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    process(child, f"img{idx:02d}_{cidx:02d}")

    return results


def process_pptx(pptx_path: Path, out_dir: Path,
                 client: OpenAI, model: str) -> list[dict]:
    prs       = Presentation(str(pptx_path))
    slide_out = out_dir / safe_stem(pptx_path.stem)
    slide_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for num, slide in enumerate(prs.slides, 1):
        prefix = f"slide{num:03d}"
        images = extract_images(slide, prs.slide_width, prs.slide_height, slide_out, prefix)
        text   = extract_text(slide)

        if images:
            for img in images:
                ai = {}
                if img["img_type"] == "full_slide":
                    print(f"    AI extracting slide {num} ...")
                    ai = ai_extract(img["_abs_path"], client, model)

                img_path    = ai.get("ai_chart_path") or img["img_path"]
                explanation = ai.get("ai_explanation") or text
                rows.append({
                    "pptx_file":   pptx_path.name,
                    "slide_num":   num,
                    "img_path":    img_path,
                    "img_type":    img["img_type"],
                    "explanation": explanation,
                })
        # slides with no image are skipped

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main(out_dir: Path, csv_path: Path, model: str) -> None:
    client = make_client()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    pptx_files = sorted(SLIDE_DIR.glob("*.pptx"))
    if not pptx_files:
        print("No .pptx files found in", SLIDE_DIR)
        return

    all_rows = []
    for pptx_path in pptx_files:
        print(f"Processing: {pptx_path.name}")
        rows = process_pptx(pptx_path, out_dir, client, model)
        all_rows.extend(rows)
        n_chart = sum(1 for r in rows if r["img_type"] == "chart")
        n_full  = sum(1 for r in rows if r["img_type"] == "full_slide")
        print(f"  {len(rows)} slides — chart: {n_chart}, full_slide: {n_full}")

    fieldnames = ["pptx_file", "slide_num", "img_path", "img_type", "explanation"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone — {len(all_rows)} rows saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path,  default=OUTPUT_DIR)
    parser.add_argument("--csv",     type=Path,  default=CSV_PATH)
    parser.add_argument("--model",   type=str,   default=DEFAULT_MODEL)
    args = parser.parse_args()
    main(args.out_dir, args.csv, args.model)
