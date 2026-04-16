"""Minimal RAG evaluation harness with checkpoint/resume support.

Usage:
  python eval_rag.py --data eval_data.sample.json --top-k 5
  python eval_rag.py --data eval_data.sample.json --top-k 5 --judge --output eval_results.json
  python eval_rag.py --data eval_data.sample.json --top-k 5 --judge --output eval_results.json --resume

Expectations:
- Qdrant is running and contains vectors for the referenced sources.
- GOOGLE_API_KEY is set if you enable --judge or want generated answers.

Checkpoint feature:
- Use --output to save incremental results to a file.
- Use --resume to skip already-processed questions and continue from where you left off.
- This allows pause/resume if you hit rate limits or model availability changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
  import google.generativeai as genai
except ImportError:
  genai = None

from data_loader import embed_texts
from vector_db import QdrantStorage

DEFAULT_TOP_K = 5
GEN_MODEL = "gemini-3-flash-preview"
JUDGE_MODEL = "gemini-3-flash-preview"


def normalize_text(text: str) -> str:
  return " ".join(text.lower().strip().split())


def token_f1(pred: str, ref: str) -> float:
  pred_tokens = normalize_text(pred).split()
  ref_tokens = normalize_text(ref).split()
  if not pred_tokens and not ref_tokens:
    return 1.0
  if not pred_tokens or not ref_tokens:
    return 0.0
  pred_set = {}
  for t in pred_tokens:
    pred_set[t] = pred_set.get(t, 0) + 1
  ref_set = {}
  for t in ref_tokens:
    ref_set[t] = ref_set.get(t, 0) + 1
  overlap = 0
  for t, c in pred_set.items():
    if t in ref_set:
      overlap += min(c, ref_set[t])
  precision = overlap / len(pred_tokens)
  recall = overlap / len(ref_tokens)
  if precision + recall == 0:
    return 0.0
  return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_sources: List[str], gold_sources: List[str]) -> float:
  if not gold_sources:
    return 0.0
  gold_set = set(gold_sources)
  return 1.0 if any(src in gold_set for src in retrieved_sources) else 0.0


def ndcg_at_k(retrieved_sources: List[str], gold_sources: List[str], k: int) -> float:
  if not gold_sources:
    return 0.0
  gold_set = set(gold_sources)
  rels = [1 if src in gold_set else 0 for src in retrieved_sources[:k]]
  dcg = 0.0
  for i, rel in enumerate(rels):
    dcg += (2 ** rel - 1) / math.log2(i + 2)
  ideal_rels = [1] * min(len(gold_set), k)
  idcg = 0.0
  for i, rel in enumerate(ideal_rels):
    idcg += (2 ** rel - 1) / math.log2(i + 2)
  return dcg / idcg if idcg > 0 else 0.0


def generate_answer(question: str, contexts: List[str]) -> str:
  if genai is None:
    raise RuntimeError("google-generativeai is not installed")
  api_key = os.getenv("GOOGLE_API_KEY")
  if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is required to generate answers")
  genai.configure(api_key=api_key)
  context_block = "\n\n".join(f"- {c}" for c in contexts)
  prompt = (
    "Use the following context to answer the question. If the answer is not in the context, say so.\n\n"
    f"Context:\n{context_block}\n\n"
    f"Question: {question}\n"
    "Answer clearly and only using the context above."
  )

  model = genai.GenerativeModel(GEN_MODEL)
  response = model.generate_content(prompt)
  return response.text.strip()


def judge_answer(question: str, answer: str, reference: str, contexts: List[str]) -> Dict[str, float]:
  if genai is None:
    raise RuntimeError("google-generativeai is not installed")
  api_key = os.getenv("GOOGLE_API_KEY")
  if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is required to judge answers")
  genai.configure(api_key=api_key)

  context_block = "\n\n".join(f"- {c}" for c in contexts)
  prompt = (
    "You are a strict evaluator. Score correctness and faithfulness from 0 to 1.\n"
    "Correctness: Does the answer match the reference?\n"
    "Faithfulness: Is the answer fully supported by the context?\n\n"
    f"Question: {question}\n"
    f"Reference Answer: {reference}\n"
    f"Answer: {answer}\n\n"
    f"Context:\n{context_block}\n\n"
    "Respond ONLY as JSON: {\"correctness\": 0.0, \"faithfulness\": 0.0}"
  )

  model = genai.GenerativeModel(JUDGE_MODEL)
  response = model.generate_content(prompt)
  text = response.text.strip()
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return {"correctness": 0.0, "faithfulness": 0.0}


def retrieve_contexts(question: str, top_k: int) -> Tuple[List[str], List[str]]:
  query_vec = embed_texts([question])[0]
  store = QdrantStorage()
  results = store.search(query_vec, top_k)
  return results.get("contexts", []), results.get("sources", [])


def load_checkpoint(output_file: str) -> List[Dict]:
  """Load existing results from checkpoint file."""
  if not os.path.exists(output_file):
    return []
  with open(output_file) as f:
    return json.load(f)


def save_checkpoint(output_file: str, results: List[Dict]) -> None:
  """Save results to checkpoint file."""
  with open(output_file, "w") as f:
    json.dump(results, f, indent=2)


def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
  """Compute metrics from all results."""
  totals = {
    "recall_at_k": 0.0,
    "ndcg_at_k": 0.0,
    "exact_match": 0.0,
    "token_f1": 0.0,
    "judge_correctness": 0.0,
    "judge_faithfulness": 0.0,
  }
  if not results:
    return totals
  for res in results:
    totals["recall_at_k"] += res.get("recall_at_k", 0.0)
    totals["ndcg_at_k"] += res.get("ndcg_at_k", 0.0)
    totals["exact_match"] += res.get("exact_match", 0.0)
    totals["token_f1"] += res.get("token_f1", 0.0)
    totals["judge_correctness"] += res.get("judge_correctness", 0.0)
    totals["judge_faithfulness"] += res.get("judge_faithfulness", 0.0)
  for k in list(totals.keys()):
    totals[k] = totals[k] / len(results)
  return totals


def run_eval(
  records: List[Dict],
  top_k: int,
  run_judge: bool,
  delay_s: float,
  judge_delay_s: float,
  embed_delay_s: float,
  output_file: str = None,
  resume: bool = False,
) -> Dict[str, float]:
  results = []
  processed_questions = set()
  
  # Load existing results if resuming
  if resume and output_file:
    results = load_checkpoint(output_file)
    processed_questions = {r["question"] for r in results}
    print(f"Loaded {len(results)} existing results from {output_file}")

  for rec in records:
    question = rec["question"]
    if question in processed_questions:
      continue
    
    reference = rec.get("answer", "")
    gold_sources = rec.get("sources", [])

    try:
      contexts, retrieved_sources = retrieve_contexts(question, top_k)
      if embed_delay_s > 0:
        time.sleep(embed_delay_s)
      answer = generate_answer(question, contexts)

      result = {
        "question": question,
        "reference": reference,
        "answer": answer,
        "recall_at_k": recall_at_k(retrieved_sources, gold_sources),
        "ndcg_at_k": ndcg_at_k(retrieved_sources, gold_sources, top_k),
        "exact_match": 1.0 if normalize_text(answer) == normalize_text(reference) else 0.0,
        "token_f1": token_f1(answer, reference),
      }

      if run_judge:
        judge = judge_answer(question, answer, reference, contexts)
        result["judge_correctness"] = float(judge.get("correctness", 0.0))
        result["judge_faithfulness"] = float(judge.get("faithfulness", 0.0))
        if judge_delay_s > 0:
          time.sleep(judge_delay_s)
      else:
        result["judge_correctness"] = 0.0
        result["judge_faithfulness"] = 0.0

      results.append(result)
      
      if output_file:
        save_checkpoint(output_file, results)
        print(f"✓ Processed and saved: {question[:50]}... ({len(results)} total)")

      if delay_s > 0:
        time.sleep(delay_s)
    
    except Exception as e:
      print(f"✗ Error processing: {question}: {e}")
      continue

  return aggregate_metrics(results)


def main() -> None:
  parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer quality")
  parser.add_argument("--data", required=True, help="Path to eval JSON file")
  parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
  parser.add_argument("--judge", action="store_true", help="Use LLM judge for correctness/faithfulness")
  parser.add_argument("--delay-s", type=float, default=0.0, help="Delay between questions (seconds)")
  parser.add_argument("--judge-delay-s", type=float, default=0.0, help="Delay after each judge call (seconds)")
  parser.add_argument("--embed-delay-s", type=float, default=0.0, help="Delay after each embed call (seconds)")
  parser.add_argument("--output", default=None, help="Output file to save incremental results")
  parser.add_argument("--resume",action="store_true", help="Resume from checkpoint (requires --output)")
  args = parser.parse_args()

  with open(args.data, "r", encoding="utf-8") as f:
    records = json.load(f)

  metrics = run_eval(
    records,
    args.top_k,
    args.judge,
    args.delay_s,
    args.judge_delay_s,
    args.embed_delay_s,
    output_file=args.output,
    resume=args.resume,
  )
  print("\n=== Final Metrics ===")
  print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
  main()