#!/usr/bin/env python
"""
upload.py

Upload the QMCSoftware/LDData repository (low discrepancy generating vectors
and matrices) to the Hugging Face Datasets Hub as a dataset repo.

This script:

1. Creates (or reuses) a dataset repo on the Hub.
2. Uploads all files from a local LDData checkout using `upload_folder`.
3. Leaves README.md in place so it becomes the dataset card.
   After upload you can edit the card in the web UI to:
     - Link to the paper (arXiv:2502.14256).
     - Add "Citation" and "Uses" sections, similar to
       - facebook/omnilingual-asr-corpus
       - nvidia/PhysicalAI-Autonomous-Vehicles
       - moondream/refcoco-m

Requirements:
    pip install "huggingface_hub>=0.32.0"

Authentication:
    - Either set HF_TOKEN in your environment:
        export HF_TOKEN=hf_xxx...
      OR pass --token on the command line.

Example usage:
    python upload.py \
        --repo-id QMCSoftware/LDData \
        --local-path /path/to/local/LDData

After upload you’ll be able to do, e.g.:

    from datasets import load_dataset
    ds = load_dataset("QMCSoftware/LDData")

and link the dataset to your paper page on Hugging Face.
"""

import argparse
import os
import sys
from pathlib import Path
import httpx
import fnmatch

from huggingface_hub import HfApi, create_repo  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload LDData to the Hugging Face Datasets Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="QMCSoftware/LDData",
        help="Target dataset repo id on the Hub (e.g. 'QMCSoftware/LDData').",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=".",
        help="Path to local LDData checkout (default: current directory).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token. If omitted, HF_TOKEN env var is used.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private (default: public).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not upload, just print what would be done.",
    )
    parser.add_argument(
        "--reset-remote",
        action="store_true",
        help="Delete the remote dataset repo on the Hub before uploading (destructive).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Answer yes to any interactive confirmation prompts (use with care).",
    )
    return parser.parse_args()


def get_token(cmd_token: str | None) -> str:
    token = cmd_token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "No token provided. Please either:\n"
            "  - set HF_TOKEN in your environment, or\n"
            "  - pass --token hf_xxx... on the command line."
        )
    return token


def main() -> None:
    args = parse_args()

    local_path = Path(args.local_path).expanduser().resolve()
    if not local_path.exists():
        raise SystemExit(f"Local path does not exist: {local_path}")

    # Sanity check: are we in LDData?
    readme = local_path / "README.md"
    if not readme.exists():
        print(
            f"WARNING: {readme} does not exist. "
            "Are you sure this is the LDData repo root?",
            file=sys.stderr,
        )

    token = get_token(args.token)
    repo_id = args.repo_id

    print(f"Using repo_id: {repo_id}")
    print(f"Local path : {local_path}")
    print(f"Private    : {args.private}")
    if args.dry_run:
        print("Dry run enabled: NOT creating or uploading, just showing intent.")
        return

    # Initialize API client
    api = HfApi(token=token)

    # 1. Create (or reuse) the dataset repo on the Hub
    print(f"Creating (or reusing) dataset repo '{repo_id}' on the Hub...")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    # 2. Upload folder contents
    #    We ignore some typical non-data files to keep the repo clean.
    #    Adjust this list if you want to exclude more or fewer things.
    ignore_patterns = [
        "sc/*",
        ".git/*",
        ".gitignore",
        ".DS_Store",
        "__pycache__/*",
        "*.pyc",
        "*.pyo",
        "*~",
        "*.ipynb_checkpoints*",
        "raw.githubusercontent.com"
        # If you do NOT want to upload the demo notebook or env file, keep these:
        # "LDData Demo.ipynb",
        # "env.yml",
    ]

    print("Uploading local folder to the Hub (this may take a while)...")

    # NOTE:
    # Older/newer versions of huggingface_hub's HfApi.upload_folder do not accept
    # a `timeout` keyword argument. Passing it raises:
    #   TypeError: HfApi.upload_folder() got an unexpected keyword argument 'timeout'
    # To remain compatible, call upload_folder without a `timeout` kwarg and
    # handle httpx.ReadTimeout explicitly.

    # Build list of candidate files and compute total size (exclude ignored)
    def is_ignored(rel_path: str) -> bool:
        # fnmatch against each ignore pattern; use posix style paths
        for patt in ignore_patterns:
            if fnmatch.fnmatch(rel_path, patt):
                return True
            if fnmatch.fnmatch("/" + rel_path, patt):
                return True
        return False

    files_to_upload = []
    total_size = 0
    for root, _, files in os.walk(local_path):
        for fname in files:
            full = Path(root) / fname
            # Robust relative path computation: use os.path.relpath to avoid Path.relative_to errors
            rel = os.path.relpath(str(full), start=str(local_path))
            # Normalize to posix separators and strip any leading './' or '/'
            rel = rel.replace(os.sep, "/")
            if rel.startswith("./"):
                rel = rel[2:]
            if rel.startswith("/"):
                rel = rel.lstrip("/")
            if is_ignored(rel):
                continue
            try:
                sz = full.stat().st_size
            except OSError:
                sz = 0
            files_to_upload.append((full, rel))
            total_size += sz

    # Threshold to prefer upload_large_folder (50 MiB)
    LARGE_THRESHOLD = 50 * 1024 * 1024

    try:
        if total_size > LARGE_THRESHOLD and hasattr(api, "upload_large_folder"):
            # Prefer API method designed for large folders when available.
            try:
                api.upload_large_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    repo_type="dataset",
                    ignore_patterns=ignore_patterns,
                    token=token,
                )
            except TypeError:
                try:
                    api.upload_large_folder(
                        folder_path=str(local_path),
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=token,
                    )
                except Exception as exc:
                    print()
                    print("ERROR: upload_large_folder failed:")
                    print(f"  {exc!r}")
                    print("Falling back to per-file upload...")
                    raise
        else:
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                repo_type="dataset",
                ignore_patterns=ignore_patterns,
            )
    except httpx.ReadTimeout:
        print()
        print("ERROR: Upload read timed out.")
        print("Possible actions:")
        print("  - Check your network connection and try again.")
        print("  - Try uploading in smaller batches (split large files or directories).")
        print("  - Use `HfApi().upload_large_folder(...)` or the CLI `hf upload-large-folder` if available.")
        print("  - Upgrade huggingface_hub to the latest version in case it adds improved timeout handling.")
        print("  - If you have very large files, consider using git-lfs or the web UI.")
        sys.exit(1)
    except TypeError as exc:
        print()
        print("ERROR: Upload function raised a TypeError (likely a signature mismatch):")
        print(f"  {exc!r}")
        print("Falling back to per-file upload...")
    except Exception:
        # If upload_large_folder/upload_folder raised but we want to fallback to per-file, continue below.
        pass

    # Per-file upload fallback: ensure we use the normalized relative paths so subfolders are preserved.
    if files_to_upload:
        print("Falling back to per-file upload (this is slower but more robust for flaky networks)...")
        for file_path, rel in files_to_upload:
            # Ensure rel is relative and posix-normalized (it already is from above)
            path_in_repo = rel.lstrip("/")  # safety
            success = False
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
                success = True
            except httpx.ReadTimeout:
                print()
                print("ERROR: Per-file upload read timed out on:", path_in_repo)
                print("You can retry this script or use `hf upload-large-folder` / `HfApi.upload_large_folder`.")
                sys.exit(1)
            except Exception as exc:
                print(f"WARNING: Failed to upload {path_in_repo!s}: {exc!r}")
            if not success:
                print(f"Failed to upload: {path_in_repo}")

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    print()
    print("✅ Upload complete.")
    print(f"Dataset is now available at: {dataset_url}")
    print()
    print("Next steps (recommended):")
    print("  1. Open the dataset page above in your browser.")
    print("  2. Edit the Dataset Card (README.md) to:")
    print("     - Add paper links (e.g., your QMCSoftware/LDData arXiv paper).")
    print("     - Add a 'Citations' section.")
    print("     - Add 'Uses' and 'Limitations' sections, similar to:")
    print("       - facebook/omnilingual-asr-corpus")
    print("       - nvidia/PhysicalAI-Autonomous-Vehicles")
    print("       - moondream/refcoco-m")
    print("  3. Use 'Paper' / 'Dataset' linking in the Hugging Face UI to")
    print("     attach the dataset to your paper so it shows up on the")
    print("     paper page and in discovery views.")


if __name__ == "__main__":
    main()