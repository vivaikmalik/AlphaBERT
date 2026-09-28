# Running FinBERT scoring on the DGX Spark

Scores all 373,139 8-K filings with ProsusAI/finbert on GPU (minutes) instead of CPU (~20h). See
`src/text.py` module docstring and `docs/SPEC.md` section 4 for the design. This file covers only
the mechanics of running it on the DGX Spark (GB10 Grace Blackwell, ARM64, CUDA GPU, 128GB unified
memory); it does not decide when to run it.

**Data sensitivity**: `data/8k_20150101_20260831_identified.parquet` is competition-provider data.
Keep it on the team's own machines only; do not publish it or push it to a public remote.

## 1. Copy code + data to the DGX

From the laptop, replace `<user>@<dgx-host>` with your DGX login:

```bash
# code: everything needed to run src/text.py (src/, requirements.txt, factor_char_list.csv)
scp -r src requirements.txt data/factor_char_list.csv <user>@<dgx-host>:~/alphabert/

# data: ONLY the 8-K filings file (342 MB) -- do not copy chars_final_with_names.parquet or
# anything else in data/ unless another module also needs to run there.
scp data/8k_20150101_20260831_identified.parquet <user>@<dgx-host>:~/alphabert/data/
```

Alternative for the code (keeps history/permissions, still leaves data/ out via .gitignore):

```bash
git archive --format=tar HEAD src requirements.txt | ssh <user>@<dgx-host> \
  'mkdir -p ~/alphabert && tar -x -C ~/alphabert'
```

Directory layout expected on the DGX: `~/alphabert/{src/, requirements.txt, data/8k_...parquet}`.
`src/config.py` derives all other paths (`outputs/cache/...`) relative to the repo root, so nothing
else needs to exist up front -- `config.py` creates `outputs/cache/` etc. on import.

## 2. Environment on the DGX (aarch64)

Use an NGC PyTorch container (has CUDA + a working aarch64 PyTorch build already):

```bash
ssh <user>@<dgx-host>
cd ~/alphabert
docker run --gpus all -it --rm -v $PWD:/work -w /work nvcr.io/nvidia/pytorch:<recent-tag>-py3 bash
# inside the container:
pip install transformers pyarrow pandas   # versions per requirements.txt if pinned there
```

Pick `<recent-tag>` from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags (e.g.
`25.08`); any recent tag with CUDA + aarch64 support works, exact version isn't load-bearing here.

`_load_finbert()` downloads ProsusAI/finbert at the pinned `FINBERT_REVISION` (see `src/text.py`)
from the HF Hub the first time it runs, so the DGX needs outbound internet, OR pre-fetch the
snapshot on the laptop and copy it over instead:

```bash
# on the laptop (once; downloads into outputs/hf_cache)
HF_HOME=outputs/hf_cache python -c "
from src.text import FINBERT_MODEL, FINBERT_REVISION
from transformers import AutoModelForSequenceClassification, AutoTokenizer
AutoTokenizer.from_pretrained(FINBERT_MODEL, revision=FINBERT_REVISION, use_fast=True)
AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL, revision=FINBERT_REVISION, use_safetensors=False)
"
scp -r outputs/hf_cache <user>@<dgx-host>:~/alphabert/outputs/hf_cache
# on the DGX, before running anything:
export HF_HOME=~/alphabert/outputs/hf_cache
```

## 3. Precision check (run this first)

```bash
python -m src.text --check 500 --device cuda
```

Expect `corr(pos-neg) fp32 vs fp16 > 0.999` and a small max abs diff (fp16 is the target dtype on
GPU; see `_setup`). If it doesn't clear 0.999, stop and report back rather than scoring the
full corpus in fp16 -- the score wraps to bf16 autocast only as documented in `src/text.py`, so if
fp16 fails on this hardware that's the point where to switch.

## 4. Full scoring run

```bash
python -m src.text --score --device cuda --max-length 512
```

`--batch-size` defaults to 256 on GPU (override with `--batch-size N` if you hit an OOM or want to
push it higher given 128GB unified memory). Expected runtime: minutes, not hours -- the CPU
fallback (this laptop, ml=128, fp32) ran at ~5-6 docs/s single-threaded-equivalent; even a
conservative GPU estimate (order-of-magnitude faster per doc, plus scoring ~4x more tokens/doc at
ml=512) should finish comfortably inside an hour, most likely in well under 15 minutes.

Resumable: if interrupted, re-running the same command skips chunks already written under
`outputs/cache/finbert_chunks_L512/` and picks up where it left off.

Output: `outputs/cache/finbert_scores_L512.parquet` (373,139 rows), written once every chunk for
this max_length exists. Do not confuse this with any `outputs/cache/finbert_scores.parquet` file
(no `_L512` suffix) that might exist from the old, pre-flag CPU run -- that legacy name is a
different, no-longer-current codepath and is never read by the current `build_text_features()`.

## 5. Copy the result back to the laptop

```bash
scp <user>@<dgx-host>:~/alphabert/outputs/cache/finbert_scores_L512.parquet \
  "D:/Work/Hackathon/AlphaBERT/outputs/cache/"
```

## 6. Verify on the laptop

```bash
python -c "
import pandas as pd
df = pd.read_parquet('outputs/cache/finbert_scores_L512.parquet')
assert len(df) == 373139, len(df)
assert df['document_id'].is_unique
probs = df[['fb_pos','fb_neg','fb_neu']]
finite = probs.notna().all(axis=1)
assert (probs[finite].sum(axis=1).sub(1.0).abs() < 1e-4).all()
assert (df['max_length'] == 512).all()
print('OK:', len(df), 'rows,', (~finite).sum(), 'empty-text (NaN) rows excluded from probability check')
"
```

A handful of NaN rows is expected and fine (filings that `clean_text()` stripped to nothing --
pure boilerplate/exhibit filings; see `score_texts` in `src/text.py`); they're excluded from
the sum-to-1 check above and from downstream tone aggregates.

Once verified, `src/text.build_text_features()` will pick up `finbert_scores_L512.parquet`
automatically (it always reads `scores_path_for(MAX_LENGTH)`, and `MAX_LENGTH` defaults to 512).
