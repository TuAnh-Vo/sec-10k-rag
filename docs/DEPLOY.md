# Publishing guide: GitHub repo + live demo

Three stages, in this order. Each one depends on the one before it.

```
A. Export the index (Kaggle)  ->  B. Push to GitHub  ->  C. Deploy the demo (Hugging Face)
```

---

## A. Export the index from Kaggle (~10 minutes)

**Why:** the demo must not re-download and re-embed 15 filings every time it starts.
The notebook already computed the 5,089 chunk vectors, so you save them once as two
plain files and ship those.

1. In Kaggle, open your notebook and replace it with `notebooks/RAG-system-project.ipynb`
   from this repo (*File → Import notebook*). Three things changed compared to your copy:
   - the SEC e-mail is read from a **Kaggle secret** instead of being written in the code,
   - the 15 filings are **pinned** by accession number (so Apple's next 10-K, filed each
     autumn, cannot silently change the corpus and break your gold IDs),
   - a new cell **8b** writes `artifacts/`.
2. Add the secret: *Add-ons → Secrets → Add a new secret*, label `SEC_USER_AGENT`,
   value like `Tu Anh your.email@example.com`. Tick it so it is attached to this notebook.
3. Turn on the GPU and run **all** cells top to bottom. Check the numbers still match:
   `FINAL CORPUS: 5,089 chunks` and `PASS: index, corpus and document list are aligned`.
4. Download from the right-hand **Output** panel (`/kaggle/working/artifacts/`):
   `chunks.jsonl`, `embeddings.npy`, `info.json`.
5. Put the three files into this repo's `artifacts/` folder.

> If any number differs from the README (5,089 chunks, the results table), stop and find
> out why before publishing. A published number that doesn't reproduce is worse than none.

---

## B. Push to GitHub (~15 minutes)

### B1. Create the empty repository
On github.com: **New repository** → name `sec-10k-rag` → **Public** → do **not** add a
README, .gitignore or license (the repo already has them) → *Create repository*.

### B2. Replace the placeholders
Search the repo for `YOUR_GITHUB_USERNAME` and `YOUR_HF_USERNAME` (in `README.md` and
`space/README.md`) and put in your real usernames.

### B3. Upload — pick ONE way

**Way 1: command line (recommended — you'll use git in every job)**
```bash
cd sec-10k-rag
git init
git add .
git status            # READ THIS LIST: no edgar_cache/, no tokens, no .env
git commit -m "SEC 10-K RAG: hybrid retrieval, cited generation, live demo"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/sec-10k-rag.git
git push -u origin main
```
GitHub does not accept your account password on the command line. The easiest login is
the GitHub CLI: install it, run `gh auth login`, choose HTTPS, and follow the browser steps.

**Way 2: no command line** — on the empty repo page, click *uploading an existing file*
and drag the **contents** of the folder in. Browser uploads allow files up to 25 MB, and
`embeddings.npy` is about 16 MB, so it fits. Hidden files like `.gitignore` can be skipped
by your file browser; if so, create it on GitHub with *Add file → Create new file*.

### B4. Check the result as a stranger would
Open the repo in a private browser window: the README renders, the diagram shows, the
results table is readable, and the notebook opens with its outputs.

---

## C. Deploy the live demo on Hugging Face (~20 minutes + first start)

### C1. What you need
- A Hugging Face account with a **verified e-mail** that is **older than 30 days**.
  Free accounts in that state can host up to **2 ZeroGPU Spaces**. Without it, creating a
  Gradio Space requires a paid plan.
- A **write** token: *Settings → Access Tokens → Create new token → type: Write*.
  Treat it like a password: never paste it into code or a notebook.

### C2. Deploy
```bash
pip install huggingface_hub
hf auth login                                            # paste the write token
python scripts/deploy_space.py YOUR_HF_USERNAME/sec-10k-rag
```
The script creates the Space **on ZeroGPU hardware** and uploads only what the demo needs.
Run it again any time you change something; it updates the same Space.

Prefer clicking? *New Space* → SDK **Gradio** → hardware **ZeroGPU** → create, then upload
`app.py`, `rag/`, `artifacts/`, `data/`, and the two files in `space/` (as `README.md` and
`requirements.txt` at the top level).

### C3. Watch the first start
Open the Space and click **Logs**. The first start installs packages and downloads about
16 GB of Qwen3-8B weights, so it takes several minutes. You are done when the log prints:
```
ready: 5,089 chunks | generator Qwen/Qwen3-8B | device cuda
```
Then try the three example questions. The share-price one must be refused.

### C4. How ZeroGPU behaves (so you can explain it in an interview)
- The Space runs on a **CPU** most of the time. When someone asks a question, the function
  marked `@spaces.GPU` borrows a GPU for that call and returns it afterwards.
- Each **visitor** spends their own daily GPU quota (about 2 minutes/day without an account,
  5 minutes with a free account). One question takes seconds of GPU time, so visitors can
  ask several. Your own hosting is free.
- A Space nobody uses goes to **sleep**. The next visitor wakes it, which takes minutes.
  Before an interview, open the demo yourself 10 minutes early.

---

## Troubleshooting

| What you see | Likely cause | Fix |
|---|---|---|
| Creating the Space says a paid plan is needed | Account < 30 days old, e-mail not verified, or hardware not set to ZeroGPU | Verify the e-mail; wait until day 30; make sure ZeroGPU is selected |
| Build fails mentioning the Python version | ZeroGPU supports Python 3.10.13 and 3.12.12 | In `space/README.md` set `python_version: "3.10"`, redeploy |
| `missing artifacts/...` or `DESYNC` on start | Stage A not finished, or files from two different runs | Re-export all three files from one notebook run |
| GPU task aborted / took too long | An answer needed more than 60 s of GPU | In `app.py` change `@spaces.GPU(duration=60)` to `120` |
| `KeyError: 'SEC_USER_AGENT'` in the notebook | Kaggle secret missing or not attached | Stage A, step 2 |
| `build_index.py` warns about the chunk count | Different library versions extract text differently | Use the versions in `requirements.txt`, or use the shipped `artifacts/` |

## Before you share the links — checklist
- [ ] No e-mail address, token or password anywhere in the repo (`git grep -i "@gmail"`)
- [ ] Placeholders replaced in `README.md` and `space/README.md`
- [ ] Demo answers the examples and refuses the share-price question
- [ ] README numbers match `results/results_validation.csv`
- [ ] GitHub repo *About* box: description, demo link as website, topics like
      `rag`, `retrieval`, `bm25`, `faiss`, `llm`, `sec-filings`
