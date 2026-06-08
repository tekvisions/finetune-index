# The Fine-Tuning Index

A living index of **LLM fine-tuning & post-training tooling** — frameworks, PEFT/LoRA, RLHF/DPO,
and training data — ranked by **momentum** (stars, push-recency, rising-newness) from live GitHub
signals.

Live: https://finetune-index.vercel.app · part of [The Living Indexes](https://living-indexes.vercel.app)

## How it works (self-updating)

A daily GitHub Action runs `build_data.py` (searches GitHub across fine-tuning queries, dedupes,
filters to real training tools — excluding inference/RAG/sibling-index repos — categorizes, scores),
`gen_details.py` (one SEO'd page per tool), `gen_og.py`, then `deploy.py` (Vercel REST).

Static HTML/CSS/JS, no framework. Industrial "forge" aesthetic (Big Shoulders Display + JetBrains
Mono, charcoal + ember).

## Run locally

```bash
GITHUB_TOKEN=... python3 build_data.py
python3 gen_details.py && python3 gen_og.py
python3 -m http.server 8080
```
