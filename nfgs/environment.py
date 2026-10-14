"""Record the numerical environment used for a reconstruction campaign."""

import argparse, json, platform, importlib.metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    import numpy, scipy, torch, matplotlib

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but unavailable")
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ["numpy", "scipy", "torch", "matplotlib"]
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
