# Use this repository with Codex Cloud

## One-time setup

Codex Cloud can work on a GitHub repository. Its coding sandbox should not be considered a dedicated long-running multi-GPU training cluster.

1. Create an **empty** GitHub repository named `NoBannedSex` under your account (public/private is your choice).
2. Unzip the source package. In that directory run:

```bash
git init
git add .
git commit -m "feat: initialize NoBannedSex independent LM v0.1"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/NoBannedSex.git
git push -u origin main
```

3. Open https://chatgpt.com/codex and select that GitHub repository. The steps can differ if your Codex workspace is not connected yet.
4. Paste the following task into Codex:

> Read AGENTS.md, docs/ARCHITECTURE.md, docs/CLOUD_GPU.md and docs/EVALUATION.md. Confirm the independent from-scratch NoBannedSex v0.1 smoke training passes in your environment; make fixes via a tested PR. Then implement the next small, testable milestone: memory-mapped token shard ingestion with deterministic document-level splits, a dataset manifest and an evaluation report. Do not substitute downloaded pretrained weights or external model APIs. Do not launch paid GPU jobs, change infrastructure or expose secrets. Maintain the no-hidden-content-filter design. Show commands, reproducible test results, and remaining limitations.

5. After that PR is reviewed, prioritize a **single-GPU paid experiment with an explicit cap**; Codex can create scripts and configurations but the user must provide/pay for cloud compute.

## What Codex should not do yet

- Claim a low-hallucination, adult-creative, 128K context, Q4 or GGUF release already exists.
- Upload sensitive datasets or checkpoints into public GitHub.
- Start expensive cloud jobs without clear maximum cost and persistent checkpoints.
