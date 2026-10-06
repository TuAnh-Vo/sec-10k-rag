"""Upload the demo to a Hugging Face Space running on ZeroGPU (free, see docs/DEPLOY.md).

    pip install huggingface_hub
    hf auth login                      # paste a token with WRITE access
    python scripts/deploy_space.py YOUR_HF_USERNAME/sec-10k-rag

Only what the demo needs is uploaded: app.py, rag/, artifacts/, data/ plus the
Space-specific README.md (its YAML header configures the Space) and requirements.txt.
Run it again after any change; it uploads a new version of the same Space.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python scripts/deploy_space.py <hf_username>/<space_name>")
    repo_id = sys.argv[1]
    for needed in ["artifacts/chunks.jsonl", "artifacts/embeddings.npy"]:
        if not (ROOT / needed).exists():
            sys.exit(f"missing {needed}: run the export cell in the notebook first (README step 1)")

    api = HfApi()
    # A free account can only host a Gradio Space on ZeroGPU, so the hardware is
    # chosen here, at creation. exist_ok=True: re-running just updates the Space.
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                    space_hardware="zero-a10g", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy(ROOT / "app.py", stage / "app.py")
        shutil.copy(ROOT / "space" / "README.md", stage / "README.md")
        shutil.copy(ROOT / "space" / "requirements.txt", stage / "requirements.txt")
        shutil.copytree(ROOT / "rag", stage / "rag",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "artifacts", stage / "artifacts")
        shutil.copytree(ROOT / "data", stage / "data")
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=stage,
                          commit_message="Deploy demo")

    print(f"done -> https://huggingface.co/spaces/{repo_id}")
    print("The first start downloads ~16 GB of model weights; expect several minutes.")
