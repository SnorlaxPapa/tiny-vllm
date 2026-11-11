"""Download an immutable HF revision and write resolved run configurations."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="HF commit SHA preferred; main is resolved once")
    parser.add_argument("--model-dir", default="checkpoints/Qwen3-4B")
    args = parser.parse_args()
    repo = "Qwen/Qwen3-4B"
    sha = HfApi().model_info(repo, revision=args.revision).sha
    snapshot_download(repo, revision=sha, local_dir=args.model_dir,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    folder = Path(__file__).parent
    config = json.loads((folder / "a100.json").read_text())
    config.update(model=str(Path(args.model_dir).resolve()), model_revision=sha)
    variants = {"a100": config,
                "smoke": config | dict(prompt_lengths=[64, 256], output_tokens=8, repeats=1, warmups=1),
                "eager": config | dict(cuda_graphs=False)}
    for name, variant in variants.items():
        path = folder / f"{name}.local.json"
        path.write_text(json.dumps(variant, indent=2)+"\n")
        print(f"Wrote {path}: {repo}@{sha}")


if __name__ == "__main__":
    main()
