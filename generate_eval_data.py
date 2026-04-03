"""Generate eval Q/A pairs from PDFs.

Usage:
  python generate_eval_data.py --dir uploads --total 50 --out eval_data.generated.json --delay-s 3

Notes:
- Requires GOOGLE_API_KEY and llama_index installed.
- Uses Gemini to generate Q/A pairs grounded in each PDF.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Dict

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from data_loader import load_and_chunk_pdf

GEN_MODEL = "gemini-2.5-flash"


def load_pdf_text(path: str, max_chars: int = 12000) -> str:
    chunks = load_and_chunk_pdf(path)
    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def generate_qa(text: str, count: int, source_id: str) -> List[Dict[str, object]]:
    if genai is None:
        raise RuntimeError("google-generativeai is not installed")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required")

    genai.configure(api_key=api_key)

    prompt = (
        "You are generating evaluation questions and answers for a RAG system.\n"
        "Create EXACTLY the requested number of Q/A pairs.\n"
        "All answers MUST be grounded in the provided document text.\n"
        "Return JSON ONLY: a list of objects with keys question, answer, sources.\n"
        f"Each sources value must be a list with the single source_id: {source_id}.\n\n"
        f"Requested count: {count}\n\n"
        f"Document text:\n{text}\n"
    )

    model = genai.GenerativeModel(GEN_MODEL)
    response = model.generate_content(prompt)
    raw = response.text.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return valid JSON: {exc}\nRaw: {raw}")

    # Normalize/validate
    out = []
    for item in data:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        out.append({"question": q, "answer": a, "sources": [source_id]})
    return out


def gather_pdfs(dir_path: str) -> List[str]:
    base = Path(dir_path)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return [str(p) for p in base.rglob("*.pdf")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval Q/A pairs from PDFs")
    parser.add_argument("--dir", default="uploads", help="Directory containing PDFs")
    parser.add_argument("--total", type=int, default=50, help="Total number of questions to generate")
    parser.add_argument("--out", default="eval_data.generated.json", help="Output JSON path")
    parser.add_argument("--delay-s", type=float, default=3.0, help="Delay between Gemini calls")
    args = parser.parse_args()

    pdfs = gather_pdfs(args.dir)
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {args.dir}")

    per_pdf = max(1, math.ceil(args.total / len(pdfs)))
    records: List[Dict[str, object]] = []

    for idx, pdf in enumerate(pdfs):
        source_id = Path(pdf).name
        text = load_pdf_text(pdf)
        qa = generate_qa(text, per_pdf, source_id)
        records.extend(qa)

        if idx < len(pdfs) - 1:
            time.sleep(args.delay_s)

    # Trim to requested total
    records = records[: args.total]

    out_path = Path(args.out)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
