"""
ZO-RLVR: Zeroth-Order Reinforcement Learning from Verifiable Rewards
backprop なしの有限差分 ZO 推定で RLVR を実行する toy 実験

RLHF の ZO 手法（ZPG, MeZO）が RLVR に単純に transfer できるかを検証する。
RLHF では pairwise 比較 → σ^{-1}(p) で価値差を推定するが、
RLVR では検証可能な報酬が直接得られるため (r+ - r-) を直接使える。

Usage:
    # epsilon 測定（適切な ε を探す）
    python scripts/train_zo_rlvr.py --mode measure --gpu 0

    # 学習（metrics.jsonl に r+/r-/grad_scale/eval_reward の推移を記録）
    python scripts/train_zo_rlvr.py --mode train --epsilon 1e-2 --seed 42 --gpu 0
"""

import sys
import json
import random
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from grpo_trainer.rewards import extract_xml_answer, extract_hash_answer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# ZO 摂動ユーティリティ（MeZO スタイル seed-based）
# メモリ効率のため u ベクトルを保持せず seed で再生成
# ──────────────────────────────────────────────


def _zo_iter(model, seed):
    """seed から u = N(0,I) を生成しパラメータとともに yield する。"""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            z = torch.randn(p.shape, generator=rng, dtype=p.dtype, device="cpu").to(p.device)
            yield p, z


def zo_perturb(model, epsilon: float, seed: int, sign: float = 1.0):
    """θ ← θ + sign * ε * u(seed)  in-place"""
    for p, z in _zo_iter(model, seed):
        p.data.add_(sign * epsilon * z)


def zo_restore(model, epsilon: float, seed: int, sign: float = 1.0):
    """zo_perturb の逆操作: θ ← θ - sign * ε * u(seed)"""
    zo_perturb(model, epsilon, seed, sign=-sign)


def zo_sgd_update(model, grad_scale: float, lr: float, seed: int):
    """backprop なし SGD: θ ← θ - lr * grad_scale * u(seed)"""
    if abs(grad_scale) < 1e-12:
        return
    for p, z in _zo_iter(model, seed):
        p.data.add_(-lr * grad_scale * z)


# ──────────────────────────────────────────────
# ロールアウト生成・報酬計算
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful math tutor. "
    "Solve the problem step by step, then provide the final answer in XML format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n42\n</answer>"
)


def make_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def compute_reward(
    model,
    tokenizer,
    questions: list[str],
    answers: list[str],
    num_generations: int,
    max_new_tokens: int,
    device: torch.device,
) -> float:
    """
    各 prompt から num_generations 個の回答を生成し、
    正解率（correctness）の平均を返す。

    ZO は勾配を使わないので全て no_grad。
    """
    total, count = 0.0, 0

    for question, answer in zip(questions, answers):
        prompt = make_prompt(question, tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_generations,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        for seq in outputs:
            generated = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            extracted = extract_xml_answer(generated)
            total += 1.0 if (extracted and extracted == answer) else 0.0
            count += 1

    return total / count if count > 0 else 0.0


# ──────────────────────────────────────────────
# データセット
# ──────────────────────────────────────────────


def load_gsm8k(split: str = "train"):
    ds = load_dataset("gsm8k", "main", split=split)
    data = [
        (row["question"], extract_hash_answer(row["answer"]))
        for row in ds
        if extract_hash_answer(row["answer"]) is not None
    ]
    return data


def sample_batch(data, batch_size: int):
    batch = random.sample(data, batch_size)
    questions = [b[0] for b in batch]
    answers = [b[1] for b in batch]
    return questions, answers


# ──────────────────────────────────────────────
# epsilon 測定
# ──────────────────────────────────────────────


def measure_epsilon(model, tokenizer, data, args, device):
    """
    各 epsilon で r+ と r- を測定し、ZO 推定の質を複数の指標で報告する。

    【指標の意味】
    nonzero_rate:
        r+ ≠ r- になる確率。εが小さすぎると摂動の影響が報酬に現れず
        ほぼ 0 になる。εが大きすぎるとモデルが壊れ両者ともほぼ 0 になる。
        ただし nonzero_rate 単体では「学習できるか」の判断には不十分。

    sign_consistency:
        grad_scale = (r+ - r-) / 2ε の符号が一致する割合。
        0.5 に近い → 更新方向がランダム（学習しない）
        1.0 に近い → 一貫した更新方向がある（学習できる可能性がある）
        nonzero_rate より本質的な指標。

    【εの決め方】
        この測定はスクリーニングであり、最終的な ε の選択は
        --mode train で実際に eval_reward の上昇を確認して決定する。
        nonzero_rate や sign_consistency が極端に低い ε は
        train モードで試す候補から除外する目安として使う。
    """
    print("\n=== Epsilon Measurement ===")
    print(
        f"{'epsilon':>10}  {'nonzero_rate':>12}  {'sign_consist':>13}  "
        f"{'r+_mean':>8}  {'r-_mean':>8}  {'|Δr|_mean':>10}"
    )
    print("-" * 75)

    results = {}
    for eps in args.epsilons:
        r_plus_list, r_minus_list, grad_scale_list = [], [], []

        for trial in range(args.num_batches):
            questions, answers = sample_batch(data, args.batch_size)
            seed = random.randint(0, 2**31)

            zo_perturb(model, eps, seed, sign=+1.0)
            r_plus = compute_reward(
                model,
                tokenizer,
                questions,
                answers,
                args.num_generations,
                args.max_new_tokens,
                device,
            )
            zo_restore(model, eps, seed, sign=+1.0)

            zo_perturb(model, eps, seed, sign=-1.0)
            r_minus = compute_reward(
                model,
                tokenizer,
                questions,
                answers,
                args.num_generations,
                args.max_new_tokens,
                device,
            )
            zo_restore(model, eps, seed, sign=-1.0)

            r_plus_list.append(r_plus)
            r_minus_list.append(r_minus)
            grad_scale_list.append((r_plus - r_minus) / (2 * eps))

        nonzero_rate = np.mean([abs(rp - rm) > 1e-8 for rp, rm in zip(r_plus_list, r_minus_list)])
        delta_mean = np.mean([abs(rp - rm) for rp, rm in zip(r_plus_list, r_minus_list)])

        # sign_consistency: 符号が多数派と一致する割合
        # 0.5 = ランダム（学習不可）、1.0 = 完全一致（学習可能）
        signs = [np.sign(g) for g in grad_scale_list if abs(g) > 1e-12]
        if len(signs) > 0:
            sign_consistency = max(signs.count(1.0), signs.count(-1.0)) / len(signs)
        else:
            sign_consistency = 0.5  # 全てゼロ → ランダムと同等

        results[eps] = {
            "nonzero_rate": float(nonzero_rate),
            "sign_consistency": float(sign_consistency),
            "r_plus_mean": float(np.mean(r_plus_list)),
            "r_minus_mean": float(np.mean(r_minus_list)),
            "delta_mean": float(delta_mean),
        }
        print(
            f"{eps:>10.1e}  {nonzero_rate:>12.3f}  {sign_consistency:>13.3f}  "
            f"{np.mean(r_plus_list):>8.3f}  {np.mean(r_minus_list):>8.3f}  "
            f"{delta_mean:>10.4f}"
        )

    print("\n[Note] 最終的な ε は --mode train で eval_reward の上昇を確認して決定する。")
    return results


# ──────────────────────────────────────────────
# ZO-RLVR 学習ループ
# ──────────────────────────────────────────────


def train_zo_rlvr(model, tokenizer, train_data, eval_data, args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 固定 eval バッチ（訓練中のθそのままで評価する）
    eval_questions, eval_answers = sample_batch(eval_data, args.eval_batch_size)

    metrics_log = []

    print(f"\n=== ZO-RLVR Training ===")
    print(
        f"ε={args.epsilon}  lr={args.lr}  batch={args.batch_size}  "
        f"N={args.num_generations}  steps={args.max_steps}"
    )
    print(
        f"{'step':>5}  {'r+':>7}  {'r-':>7}  {'grad_scale':>12}  " f"{'train_r':>8}  {'eval_r':>8}"
    )
    print("-" * 60)

    for step in range(1, args.max_steps + 1):
        questions, answers = sample_batch(train_data, args.batch_size)
        zo_seed = random.randint(0, 2**31)

        # θ+εu でロールアウト
        zo_perturb(model, args.epsilon, zo_seed, sign=+1.0)
        r_plus = compute_reward(
            model, tokenizer, questions, answers, args.num_generations, args.max_new_tokens, device
        )
        zo_restore(model, args.epsilon, zo_seed, sign=+1.0)

        # θ-εu でロールアウト
        zo_perturb(model, args.epsilon, zo_seed, sign=-1.0)
        r_minus = compute_reward(
            model, tokenizer, questions, answers, args.num_generations, args.max_new_tokens, device
        )
        zo_restore(model, args.epsilon, zo_seed, sign=-1.0)

        # ZO 勾配推定 & SGD 更新（backprop なし）
        # RLVR では検証可能な報酬が直接得られるので link function 逆変換は不要
        #
        # 【grad_scale の爆発問題】
        # バイナリ報酬では r+ - r- が最大 1.0 になるため、
        # grad_scale = (r+ - r-) / 2ε は ε が小さいほど爆発する。
        # 例: ε=1e-4, r+-r-=0.188 → grad_scale=937.5
        # これをそのまま lr=1e-3 で使うと更新量 ≈ 0.94 * u となり
        # ε=1e-4 の摂動幅の 9375 倍の更新が起きてモデルが崩壊する。
        #
        # 【対処】
        # (1) grad_scale を [-grad_clip, grad_clip] にクリップ（デフォルト 1.0）
        # (2) lr を ε と同スケールに下げる（目安: lr ≈ epsilon）
        #     例: ε=1e-4 → lr=1e-4 で更新量 = lr * clip * u = 1e-4 * u
        grad_scale = (r_plus - r_minus) / (2 * args.epsilon)
        grad_scale = float(np.clip(grad_scale, -args.grad_clip, args.grad_clip))
        zo_sgd_update(model, grad_scale, args.lr, zo_seed)

        train_reward = (r_plus + r_minus) / 2

        # ── 定期 eval：摂動なしの θ そのままで評価 ──────────────────────
        # train_reward は摂動モデルの平均であり訓練ノイズを含む。
        # eval_reward は現在の θ での真の性能を反映する。
        # 学習が起きているかどうかはこちらで判断する。
        eval_reward = None
        if step % args.eval_interval == 0 or step == 1 or step == args.max_steps:
            eval_reward = compute_reward(
                model,
                tokenizer,
                eval_questions,
                eval_answers,
                args.num_generations,
                args.max_new_tokens,
                device,
            )
            print(
                f"{step:>5}  {r_plus:>7.3f}  {r_minus:>7.3f}  {grad_scale:>+12.4f}  "
                f"{train_reward:>8.3f}  {eval_reward:>8.3f}  ← eval"
            )
        else:
            print(
                f"{step:>5}  {r_plus:>7.3f}  {r_minus:>7.3f}  {grad_scale:>+12.4f}  "
                f"{train_reward:>8.3f}  {'':>8}"
            )
        # ────────────────────────────────────────────────────────────────

        metrics_log.append(
            {
                "step": step,
                "r_plus": r_plus,
                "r_minus": r_minus,
                "grad_scale": grad_scale,
                "train_reward": train_reward,
                "eval_reward": eval_reward,  # eval 非実施ステップは None
            }
        )

    # メトリクス保存
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.jsonl", "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")
    logger.info(f"metrics saved to {out / 'metrics.jsonl'}")

    return metrics_log


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="ZO-RLVR toy experiment")
    p.add_argument(
        "--mode",
        choices=["train", "measure"],
        default="train",
        help="train: 学習実行  measure: epsilon 測定",
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # ZO ハイパーパラメータ
    p.add_argument("--epsilon", type=float, default=1e-2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="grad_scale のクリップ幅。バイナリ報酬では grad_scale が "
        "爆発するため lr と合わせて調整が必要。"
        "更新量の目安: lr * grad_clip * |u| ≈ lr * grad_clip",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)

    # 学習モード
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--output-dir", type=str, default="outputs/zo-rlvr-toy")
    # 定期 eval の設定（追加）
    p.add_argument(
        "--eval-interval", type=int, default=10, help="何ステップごとに eval を実施するか"
    )
    p.add_argument(
        "--eval-batch-size", type=int, default=16, help="eval 時に使う問題数（固定バッチ）"
    )

    # 測定モード
    p.add_argument("--epsilons", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1])
    p.add_argument("--num-batches", type=int, default=10, help="epsilon 測定で使うバッチ数")

    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # モデルロード（ZO は eval モード固定、勾配不要）
    logger.info(f"loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()  # ZO は forward のみ、backward 不要

    # データロード
    logger.info("loading GSM8K...")
    train_data = load_gsm8k(split="train")
    eval_data = load_gsm8k(split="test")  # 追加：eval 用に test split を分離
    logger.info(f"train: {len(train_data)} samples  eval: {len(eval_data)} samples")

    if args.mode == "measure":
        results = measure_epsilon(model, tokenizer, train_data, args, device)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "epsilon_results.json", "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        logger.info(f"saved to {out / 'epsilon_results.json'}")

    else:
        # train_data と eval_data を分けて渡す（追加）
        train_zo_rlvr(model, tokenizer, train_data, eval_data, args, device)


if __name__ == "__main__":
    main()
