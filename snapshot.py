import os
import shutil
import argparse
from utils import load_env_file


def main():
    env = load_env_file()
    parser = argparse.ArgumentParser(description="Snapshot current outputs into outputs/snapshots/<tag>")
    parser.add_argument("--tag", required=True, help="Snapshot tag name")
    parser.add_argument("--src-cache", default=env.get("CACHE_DIR", "outputs/preds"))
    parser.add_argument("--src-jsonl", default=env.get("COMBINED_JSONL", "outputs/ball_detections.jsonl"))
    args = parser.parse_args()

    dst_root = os.path.join("outputs", "snapshots", args.tag)
    dst_cache = os.path.join(dst_root, "preds")
    os.makedirs(dst_root, exist_ok=True)

    if os.path.isdir(args.src_cache):
        if os.path.exists(dst_cache):
            shutil.rmtree(dst_cache)
        shutil.copytree(args.src_cache, dst_cache)
        print(f"Copied cache -> {dst_cache}")
    else:
        print(f"Skip cache copy; not found: {args.src_cache}")

    if os.path.exists(args.src_jsonl):
        shutil.copy2(args.src_jsonl, os.path.join(dst_root, os.path.basename(args.src_jsonl)))
        print(f"Copied jsonl -> {dst_root}")
    else:
        print(f"Skip jsonl copy; not found: {args.src_jsonl}")

    print(f"Snapshot saved at: {dst_root}")


if __name__ == "__main__":
    main()

