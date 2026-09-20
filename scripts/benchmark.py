"""Benchmark Hana VITS on the current machine.

Examples:
  python scripts/benchmark.py --device auto --dtype auto --iterations 8
  HANA_DTYPE=fp16 python scripts/benchmark.py --device cuda:0 --iterations 8
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.engine import HanaVITSEngine
TEST_TEXTS = [
    "こんにちは、今日は調子はいかがですか？",
    "おはようございます、マスター。今日も一緒に頑張りましょう。",
    "えっ、本当に？ それなら、とってもうれしいです！",
    "少しだけ待ってくださいね。すぐに準備します。",
    "マスター、今日の予定を教えてもらえますか？",
    "なるほど、それなら別の方法を試してみましょう。",
    "えへへ、そんなに褒められると照れちゃいます。",
    "大丈夫ですよ。私はここにいますから、ゆっくり話しましょう。",
    "それでは、次の話題に行きましょう。何について話しますか？",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp32", choices=["auto", "fp16", "fp32"])
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--text-cache-items", type=int, default=0,
                        help="CPU text/phoneme cache entries; 0 disables it for baseline runs")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-remove-weight-norm", action="store_true")
    args = parser.parse_args()

    engine = HanaVITSEngine(
        ROOT / "models" / "G_latest.pth",
        ROOT / "config" / "fine.json",
        args.device,
        args.dtype,
        remove_weight_norm=not args.no_remove_weight_norm,
        compile_model=args.compile,
        cache_items=0,
        cache_max_mb=0,
        text_cache_items=args.text_cache_items,
    )
    engine.load()

    # Warm-up with the same short test phrase; the cache is disabled.
    try:
        for i in range(max(0, args.warmup)):
            engine.synthesize(TEST_TEXTS[i % len(TEST_TEXTS)], "Hana", "Japanese", 1.0)

        results = []
        for i in range(max(1, args.iterations)):
            result = engine.synthesize(TEST_TEXTS[i % len(TEST_TEXTS)], "Hana", "Japanese", 1.0)
            results.append(result)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing runtime dependency: {exc.name}. Install requirements.txt before benchmarking."
        ) from exc

    times = [r.generation_ms for r in results]
    rtfs = [r.rtf for r in results]
    durations = [r.audio_seconds for r in results]
    print("Hana VITS benchmark")
    print("-------------------")
    print(f"device: {engine.device}")
    print(f"dtype: {engine.dtype}")
    print(f"weight_norm_removed: {engine.weight_norm_removed}")
    print(f"torch_compile_requested: {args.compile}")
    print(f"text_cache_items: {args.text_cache_items}")
    print(f"iterations: {len(results)} (warmup {args.warmup})")
    preprocess = [r.preprocess_ms for r in results]
    inference = [r.inference_ms for r in results]
    text_hits = sum(1 for r in results if r.text_cache_hit)
    print(f"generation mean: {statistics.mean(times):.2f} ms")
    print(f"preprocess mean: {statistics.mean(preprocess):.2f} ms")
    print(f"inference mean: {statistics.mean(inference):.2f} ms")
    print(f"text cache hits: {text_hits}/{len(results)}")
    print(f"generation median: {statistics.median(times):.2f} ms")
    if len(times) >= 2:
        print(f"generation stdev: {statistics.stdev(times):.2f} ms")
    print(f"audio mean: {statistics.mean(durations):.3f} s")
    print(f"RTF mean: {statistics.mean(rtfs):.4f}")
    if engine.device.type == "cuda":
        import torch
        idx = engine.device.index or torch.cuda.current_device()
        print(f"GPU allocated: {torch.cuda.memory_allocated(idx)/1024**2:.0f} MB")
        print(f"GPU reserved: {torch.cuda.memory_reserved(idx)/1024**2:.0f} MB")


if __name__ == "__main__":
    main()
