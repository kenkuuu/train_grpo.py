"""
Sparsity analysis for GRPO training checkpoints.

2つのチェックポイント間のパラメータ差分からスパース性を分析する。
チェックポイント間で変化したパラメータの割合・レイヤー分布・閾値感度を出力。

Usage:
    # ベースモデル vs チェックポイント
    python scripts/analyze_sparsity.py \\
        --checkpoint outputs/grpo-gsm8k-1.5b-full_s42/checkpoint-234

    # チェックポイント同士の比較（step 1 → step 50）
    python scripts/analyze_sparsity.py \\
        --base outputs/grpo-gsm8k-1.5b-full_s42/checkpoint-1 \\
        --checkpoint outputs/grpo-gsm8k-1.5b-full_s42/checkpoint-50

    # 結果を JSON に保存
    python scripts/analyze_sparsity.py \\
        --checkpoint outputs/grpo-gsm8k-1.5b-full_s42/checkpoint-234 \\
        --output outputs/sparsity_analysis.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from grpo_trainer.mask import compute_delta_mask, get_sparsity_stats

logger = logging.getLogger(__name__)

DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TOLERANCES = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def parse_args():
    p = argparse.ArgumentParser(description="Analyze sparsity of model checkpoint deltas")
    p.add_argument(
        "--checkpoint", required=True,
        help="Fine-tuned checkpoint to analyze",
    )
    p.add_argument(
        "--base", default=DEFAULT_BASE,
        help=f"Base model or checkpoint to compare against (default: {DEFAULT_BASE})",
    )
    p.add_argument(
        "--threshold", type=float, default=1e-5,
        help="Primary threshold for active/frozen classification (default: 1e-5)",
    )
    p.add_argument(
        "--tolerances", type=float, nargs="+", default=DEFAULT_TOLERANCES,
        help="Threshold sweep for sensitivity analysis",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Save full results to JSON file",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Show top-K most/least active layers (default: 5)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    print(f"\n{'=' * 64}")
    print(f"  Sparsity Analysis")
    print(f"  base       : {args.base}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"{'=' * 64}")

    # ── threshold sweep ──────────────────────────────────────────────
    print(f"\n{'threshold':>12}  {'sparsity':>10}  {'active_frac':>12}  {'active_params':>14}")
    print("-" * 56)

    sweep_results = {}
    for tol in args.tolerances:
        mask = compute_delta_mask(args.base, args.checkpoint, threshold=tol)
        stats = get_sparsity_stats(mask)
        print(
            f"{tol:>12.1e}  {stats.global_sparsity:>10.4f}  "
            f"{stats.active_fraction:>12.4f}  {stats.active_params:>14,}"
        )
        sweep_results[str(tol)] = {
            "global_sparsity": stats.global_sparsity,
            "active_fraction": stats.active_fraction,
            "active_params": stats.active_params,
            "total_params": stats.total_params,
        }

    # ── detailed stats at primary threshold ─────────────────────────
    print(f"\n{'=' * 64}")
    print(f"  Detailed stats  (threshold = {args.threshold:.1e})")
    print(f"{'=' * 64}")

    mask = compute_delta_mask(args.base, args.checkpoint, threshold=args.threshold)
    stats = get_sparsity_stats(mask)
    print(stats)

    # ── layer ranking ────────────────────────────────────────────────
    if stats.layerwise:
        sorted_layers = sorted(stats.layerwise.items(), key=lambda x: x[1])

        k = min(args.top_k, len(sorted_layers))
        print(f"\nTop-{k} most sparse layers (fewest active params):")
        for layer, frac in sorted_layers[:k]:
            print(f"  Layer {int(layer):>3d}: active={frac:.4f}  sparse={1 - frac:.4f}")

        print(f"\nTop-{k} most active layers (most params updated):")
        for layer, frac in sorted_layers[-k:][::-1]:
            print(f"  Layer {int(layer):>3d}: active={frac:.4f}  sparse={1 - frac:.4f}")

    # ── per-parameter top-K ──────────────────────────────────────────
    sorted_params = sorted(stats.paramwise.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop-{args.top_k} most active parameters:")
    for name, frac in sorted_params[:args.top_k]:
        print(f"  {frac:.4f}  {name}")

    print(f"\nTop-{args.top_k} most frozen parameters:")
    for name, frac in sorted_params[-args.top_k:][::-1]:
        print(f"  {frac:.4f}  {name}")

    # ── JSON output ──────────────────────────────────────────────────
    if args.output:
        out = {
            "base": str(args.base),
            "checkpoint": str(args.checkpoint),
            "threshold_sweep": sweep_results,
            "detailed": {
                "threshold": args.threshold,
                "global_sparsity": stats.global_sparsity,
                "active_fraction": stats.active_fraction,
                "active_params": stats.active_params,
                "total_params": stats.total_params,
                "layerwise": {k: v for k, v in stats.layerwise.items()},
                "paramwise": stats.paramwise,
            },
        }
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
