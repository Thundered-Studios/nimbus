# Nimbus

**A powerful open language model by [Thundered Studios](https://thundered-studios.github.io).**

Nimbus is a family of language models with built-in reasoning and thinking capabilities. Available in multiple sizes from 0.6B to 235B parameters.

---

## Quickstart

```bash
pip install -r requirements.txt
python chat.py
```

Weights download automatically on first run (~2.5 GB for the default 4B model).

---

## Model Sizes

| Variant | Parameters | VRAM (fp16) | VRAM (4-bit) |
|---------|-----------|-------------|--------------|
| `0.6b`  | 0.6B      | ~2 GB       | ~1 GB        |
| `1.7b`  | 1.7B      | ~4 GB       | ~2 GB        |
| `4b`    | 4B        | ~8 GB       | ~4 GB        |
| `8b`    | 8B        | ~18 GB      | ~6 GB        |
| `14b`   | 14B       | ~30 GB      | ~10 GB       |
| `32b`   | 32B       | ~65 GB      | ~20 GB       |
| `30b`   | 30B (MoE) | ~60 GB      | ~18 GB       |
| `235b`  | 235B (MoE)| ~470 GB     | ~120 GB      |

---

## Usage

### Chat (CLI)

```bash
# Default (4B)
python chat.py

# Choose a size
python chat.py --variant 8b

# Low VRAM / CPU — quantized
python chat.py --variant 4b --4bit
```

### Python API

```python
from nimbus import Nimbus

nimbus = Nimbus.load(variant="4b")

# Single response
response = nimbus.chat("Explain transformers in simple terms")
print(response)

# Streaming
for chunk in nimbus.stream("Write a Python function to reverse a string"):
    print(chunk, end="", flush=True)

# Multi-turn conversation
history = []
while True:
    user = input("You: ")
    response = nimbus.chat(user, history=history)
    print(f"Nimbus: {response}")
    history += [{"role": "user", "content": user},
                {"role": "assistant", "content": response}]
```

### OpenAI-compatible API Server

```bash
python serve.py --variant 4b --port 8000
```

Then use any OpenAI SDK client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="nimbus")
response = client.chat.completions.create(
    model="nimbus",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Fine-tuning (LoRA)

Prepare a JSONL file:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Then fine-tune:
```bash
python finetune.py --data data/finetune.jsonl --variant 4b --output out/nimbus-ft
python finetune.py --data data/finetune.jsonl --variant 8b --4bit --output out/nimbus-ft
```

---

## Hardware Requirements

| Setup | Minimum |
|-------|---------|
| CPU only | 8 GB RAM (0.6B–1.7B model) |
| Single GPU (4-bit) | 4 GB VRAM (4B model) |
| Single GPU (fp16) | 8 GB VRAM (4B model) |
| Apple Silicon | 8 GB unified memory (4B model) |

---

## License

MIT — [Thundered Studios](https://thundered-studios.github.io)
