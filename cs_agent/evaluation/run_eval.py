"""
评估脚本：计算意图准确率、实体 F1、RAG 忠实度（简化版）。
运行：python -m cs_agent.evaluation.run_eval
"""
from __future__ import annotations
import sys
import io
# Force UTF-8 output on Windows (avoids GBK encoding errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
from collections import defaultdict
from cs_agent.graph import chat


def _intent_accuracy(preds: list[str], golds: list[str]) -> float:
    correct = sum(p == g for p, g in zip(preds, golds))
    return correct / len(golds) if golds else 0.0


def _entity_f1(pred_entities: list[list[dict]], gold_entities: list[list[dict]]) -> dict:
    total_tp = total_fp = total_fn = 0
    for preds, golds in zip(pred_entities, gold_entities):
        pred_set = {(e["text"], e["label"]) for e in preds}
        gold_set = {(e["text"], e["label"]) for e in golds}
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        total_tp += tp; total_fp += fp; total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _simple_faithfulness(answer: str, sources_content: list[str]) -> float:
    """简化忠实度：检查答案关键词是否在检索内容中出现。"""
    if not sources_content or not answer:
        return 0.0
    words = [w for w in answer if '\u4e00' <= w <= '\u9fff']
    if not words:
        return 0.0
    combined = " ".join(sources_content)
    hits = sum(1 for w in words if w in combined)
    return hits / len(words)


def run_eval(eval_path: str | None = None, max_samples: int = 30) -> dict:
    eval_path = eval_path or str(Path(__file__).parent / "eval_set.json")
    samples = json.loads(Path(eval_path).read_text(encoding="utf-8"))[:max_samples]

    pred_intents, gold_intents = [], []
    pred_entities_all, gold_entities_all = [], []
    faithfulness_scores = []
    intent_by_class: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    print(f"\n{'='*60}")
    print(f"  评估开始 | 样本数: {len(samples)}")
    print(f"{'='*60}")

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['query'][:40]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = chat(sample["query"], session_id=f"eval_{i}")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        elapsed = time.time() - t0
        pred_intent = result["intent"]
        gold_intent = sample["expected_intent"]

        pred_intents.append(pred_intent)
        gold_intents.append(gold_intent)
        pred_entities_all.append(result["entities"])
        gold_entities_all.append(sample.get("expected_entities", []))

        intent_by_class[gold_intent]["total"] += 1
        if pred_intent == gold_intent:
            intent_by_class[gold_intent]["correct"] += 1

        if result.get("retrieved_chunks"):
            fth = _simple_faithfulness(
                result["answer"],
                [c["content"] for c in result["retrieved_chunks"]],
            )
            faithfulness_scores.append(fth)

        match = "✓" if pred_intent == gold_intent else f"✗({pred_intent})"
        print(f"{match} [{elapsed:.1f}s]")

    # 汇总
    intent_acc = _intent_accuracy(pred_intents, gold_intents)
    entity_metrics = _entity_f1(pred_entities_all, gold_entities_all)
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0

    print(f"\n{'='*60}")
    print(f"  评估结果")
    print(f"{'='*60}")
    print(f"意图准确率:    {intent_acc:.3f}  ({sum(p==g for p,g in zip(pred_intents,gold_intents))}/{len(gold_intents)})")
    print(f"实体 Precision: {entity_metrics['precision']:.3f}")
    print(f"实体 Recall:    {entity_metrics['recall']:.3f}")
    print(f"实体 F1:        {entity_metrics['f1']:.3f}")
    print(f"RAG 忠实度:    {avg_faithfulness:.3f}  (基于 {len(faithfulness_scores)} 条检索结果)")
    print(f"\n各意图类别准确率:")
    for intent, stats in sorted(intent_by_class.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {intent:<15}: {acc:.3f}  ({stats['correct']}/{stats['total']})")

    metrics = {
        "intent_accuracy": intent_acc,
        "entity_precision": entity_metrics["precision"],
        "entity_recall": entity_metrics["recall"],
        "entity_f1": entity_metrics["f1"],
        "rag_faithfulness": avg_faithfulness,
        "intent_by_class": dict(intent_by_class),
        "n_samples": len(samples),
    }

    out_path = Path(__file__).parent / "eval_results.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至: {out_path}")
    return metrics


if __name__ == "__main__":
    run_eval()
