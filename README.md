# Nimbus-1

> A small language model trained from scratch. Open research by [Thundered Studios](https://thundered-studios.github.io).

Nimbus-1 is a decoder-only transformer language model (~124M parameters) trained completely from scratch on ~10B tokens of text. Built and researched by Arjun Katta (age 12) at Thundered Studios.

**Blog post:** [thundered-studios.github.io/blog/nimbus-1](https://thundered-studios.github.io/blog/nimbus-1)

---

## Model Specs

| Parameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~124M |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dim | 768 |
| Context length | 1,024 tokens |
| Vocabulary size | 50,257 (BPE) |
| Training tokens | ~10B |

---

## Training Setup

- **Dataset:** FineWeb (10B token sample)
- **Framework:** PyTorch
- **Optimizer:** AdamW (β1=0.9, β2=0.95)
- **Learning rate:** 6e-4, cosine decay with linear warmup
- **Batch size:** ~500k tokens/step (gradient accumulation)
- **Precision:** bfloat16
- **Hardware:** A100 / H100 (RunPod)

---

## Repo Structure

```
nimbus/
├── data/           # Dataset preparation & tokenization
├── model/          # Model architecture (transformer)
├── train.py        # Training loop
├── eval.py         # Evaluation & benchmarks
├── tokenizer/      # BPE tokenizer
└── configs/        # Training configs
```

> Code and weights coming as training progresses. Follow the [blog](https://thundered-studios.github.io/blog/nimbus-1) for updates.

---

## Philosophy

Everything is published — training code, weights, loss curves, eval results, and failures. This is open research. No secrets.

---

## License

MIT — use it, fork it, learn from it.

---

*Thundered Studios · [thundered-studios.github.io](https://thundered-studios.github.io)*
