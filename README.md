# NoBannedSex

**Independent, open-source language model research. 从随机权重训练的独立通用语言模型研究项目。**

[![CPU CI](https://github.com/Aelurillus/NoBannedSex/actions/workflows/ci.yml/badge.svg)](https://github.com/Aelurillus/NoBannedSex/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> **当前状态：v0.1 工程研究基线。** 仓库包含可运行的随机初始化模型、单进程训练器、CPU 测试、数据分片与评测工具。尚无具备实用对话能力的公开模型权重。测试通过证明软件运行与数值行为，不代表通用能力或低幻觉已经实现。

## Vision / 项目目标

NoBannedSex aims to develop a user-controlled, independently trained, quantizable general-purpose language model. Training and inference use project-owned weights, initialized randomly; they do not call hosted foundation-model APIs or download pretrained weights.

- 自然对话与多任务：写作、编程、推理、翻译、文档理解等。
- 事实可靠性：区分事实、推测、不确定性与虚构；同时衡量回答质量和回答覆盖率。
- 自由创作与角色扮演：角色一致性、自然表达，以及自愿成年设定下的成人向虚构创作。
- 长上下文与长文本：逐步训练和评估更长输入与输出，不把修改窗口参数当成能力证明。
- 本地部署：后续验证低精度推理、量化工具链和可分发模型格式。

项目不计划内置强制远程审核、关键词输出过滤器或内容审核分类器。**事实真实性训练与内容审核分开设计。** 这描述工程设计，不保证未来模型在任何提示词下的输出行为。数据使用权和第三方许可证需要单独记录。

## Implemented / 已实现的工程能力

| 模块 | 当前实现与边界 |
| --- | --- |
| 模型 | Decoder-only Transformer、因果 SDPA、GQA、RoPE、RMSNorm、SwiGLU、共享嵌入、梯度检查点 |
| 训练 | 单 CPU / 单 CUDA 进程、AdamW、梯度累积/裁剪、预热与余弦学习率、FP32 或 CUDA BF16 自动混合精度 |
| 恢复 | 原子保存模型/优化器/随机状态；校验完整配置、语料、分词器、运行环境；CPU 连续训练与恢复训练逐位一致测试 |
| 分词 | UTF-8 字节基线；可自行训练 ByteLevel BPE；分词器随检查点保存并校验哈希 |
| 数据 | 本地来源/许可清单、流式文档读取、NFC 精确去重、SQLite 文档索引、稳定文档哈希划分、mmap token 分片及校验 |
| 推理与导出 | 贪心、temperature、top-k、top-p、EOS；safetensors 导出与重载一致性测试 |
| 评测 | 验证损失、困惑度、next-token accuracy；受控短答案准确率、回答覆盖率和弃答率；JSON 报告 |
| 自动化 | GitHub Actions CPU 测试、静态检查、训练→恢复→生成→导出→分片训练→评测验收 |
| GPU 准备 | 单卡候选配置、无权重分配的形状预检、合成吞吐基准和已有 GPU 上的手动启动脚本；尚未实测 CUDA |

GQA 当前显式复制 KV 头。推理逐 token 重算上下文，超过窗口后使用滑动截断；没有 KV cache、分页注意力、分布式训练或高效长上下文运行时。数据仅做标准化后的精确去重，没有近重复/语义去重或自动许可证判定。

## CPU quickstart / 无付费 GPU 验证

Python 3.10+。CI 覆盖 Python 3.11/3.12；本次本地验证使用 Python 3.12 与 PyTorch 2.6 CPU。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
bash scripts/cpu_acceptance.sh runs/my-first-check
```

最后一个命令在 CPU 上执行完整工程验收，输出路径必须为空。也可以分别运行：

```bash
python -m nobannedsex.train --config configs/smoke.json --data data/sample.txt --out runs/smoke --device cpu --stop-after 4
python -m nobannedsex.train --config configs/smoke.json --data data/sample.txt --out runs/smoke --device cpu --resume runs/smoke/latest.pt
python -m nobannedsex.generate --checkpoint runs/smoke/latest.pt --prompt '你好' --max-new-tokens 20 --temperature 0 --device cpu
python -m nobannedsex.export --checkpoint runs/smoke/latest.pt --out exports/smoke
python -m nobannedsex.evaluate --checkpoint runs/smoke/latest.pt --suite data/eval-smoke.jsonl --out runs/smoke/evaluation.json --lm-data data/eval-heldout.txt
```

`steps` 是固定的学习率日程终点；用 `--stop-after` 提前停止，再用相同配置 `--resume`。改变日程、种子或采样配置会被拒绝，避免悄悄改变实验。旧 v0.1 格式检查点可推理，但不支持新的严格恢复流程。

**Smoke 输出没有实用语义能力。** 小样本训练损失下降、自动短答案得分、允许生成很多 token，都不能证明对话、事实性、角色扮演或长文本质量。

## Data and experiments / 数据与实验

```bash
python -m nobannedsex.data --sources data/sources.sample.json --out prepared/sample --val-fraction 0.15
python -m nobannedsex.train --config configs/smoke.json --data prepared/sample --out runs/sharded --device cpu
python -m nobannedsex.preflight --config configs/research-280m.json
python -m nobannedsex.benchmark --config configs/smoke.json --device cpu --steps 3 --out runs/benchmark.json
```

自有语料必须附上来源和使用权记录。准备正式数据时使用自己的来源清单与单独的测试集，勿将工程 fixture 当训练配方。[数据管线](docs/DATA_PIPELINE.md)、[实验评测](docs/EVALUATION.md) 和 [验证记录](docs/VALIDATION.md) 说明数据格式、指标定义及限制。

## Compute plan / 算力方案

工程与 CI 可以先在 CPU 上验证。`research-280m.json` 只是约 280M 参数的形状候选，未验证显存、收敛、成本或模型质量。实际参数量由预检工具计算；32K BPE 必须由自有语料训练，实际词表大小须与配置相等。

**没有启动付费云 GPU。** 仓库不会自动租赁实例。只在用户明确批准预算、已有 GPU 和持久存储的情况下手动运行 [GPU 交接流程](docs/CLOUD_GPU.md)。启动脚本的预算字段只是记录，不能代替提供商的账单上限。

## Roadmap / 尚未交付

1. 实测单 GPU 的显存、吞吐、混合精度与故障恢复。
2. 扩大有使用权的语料，补充近重复去重、独立测试集和质量分析。
3. 实现分布式预训练与大规模检查点管理。
4. 开展多任务、事实真实性、对话、虚构角色扮演的训练与独立评估。
5. 实测长上下文与长文本一致性，再实现高效 KV cache 和量化部署。
6. 发布有复现实验、模型卡、数据来源与明确许可证的权重。

目前未实现指令微调/偏好优化、已验证的低幻觉能力、有效 128K 上下文、4-bit/GGUF/GPTQ/AWQ 发布、多模态或可用的通用聊天模型。

## License / 许可证

项目自有源码与文档采用 [Apache License 2.0](LICENSE)。第三方组件遵守各自许可证；数据集、分词器与未来模型权重的使用权需单独记录，不能因本仓库的代码许可证而自动获得授权。仓库仅含项目合成测试文本，没有发布预训练模型权重或外部训练语料。

[架构与边界](docs/ARCHITECTURE.md) · [原始源码来源](docs/SOURCE_PROVENANCE.md) · [开发说明](AGENTS.md)
