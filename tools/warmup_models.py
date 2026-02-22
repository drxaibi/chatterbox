import argparse
import time

import torch

from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.vc import ChatterboxVC

MODEL_LOADERS = {
    "standard": ChatterboxTTS,
    "turbo": ChatterboxTurboTTS,
    "multilingual": ChatterboxMultilingualTTS,
    "vc": ChatterboxVC,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-download and cache Chatterbox model checkpoints for local/offline-ready usage."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_LOADERS.keys()) + ["all"],
        default=["all"],
        help="Which models to warm up. Default: all",
    )
    args = parser.parse_args()

    selected = list(MODEL_LOADERS.keys()) if "all" in args.models else args.models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    start = time.time()

    for name in selected:
        print(f"\n=== Warming up: {name} ===", flush=True)
        model_class = MODEL_LOADERS[name]
        model_class.from_pretrained(device)
        print(f"OK: {name}", flush=True)

    elapsed = time.time() - start
    print(f"\nDone. Model artifacts are cached locally. Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
