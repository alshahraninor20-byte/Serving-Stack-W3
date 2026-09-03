# Lab W3D2: inference anatomy, by hand

Start: **W3D1 checkpoint. A fresh Colab T4 runtime.**
Objective: Measure the two phases of LLM inference — **prefill and decode** — by hand using Transformers, then measure KV cache growth and the effect of static batching.

Today there is **no vLLM**. The point is to understand what the inference engine is doing before replacing the hand-rolled approach with vLLM on W3D3.

Time: about 3 hours. Steps carry a rough clock so you can pace it.

## Predict (by hand)

Fill this in before you run anything. Use what you learned from W2 and W3D1.

* A longer prompt should make **TTFT** higher because the model must process the whole prompt during prefill.
* For Qwen2.5-1.5B, the KV cache uses approximately **28 KB per token** in FP16.
* If the context grows from 512 to 4096 tokens, the KV cache should grow with it.
* Increasing batch size should improve throughput, but static batching can waste compute when requests have different output lengths.
* A batch containing one long request and several short requests will be limited by the longest request.

The interesting question today is what actually happens between the prompt and the generated tokens.

---

## The anatomy

You will load the model directly with **Transformers and Accelerate** and measure three things:

1. **TTFT** — Time to First Token
2. **TPOT** — Time Per Output Token
3. **KV Cache** — memory required to store previous keys and values

Then you will implement a small static batching queue and measure throughput and slot efficiency.

There is no vLLM today; that is deliberate. W3D3 will use these measurements as the baseline for the vLLM comparison.

---

### Cell 1: install pins and load the model (about 5 min)

Install only Transformers and Accelerate:

```python
import subprocess, sys

TRANSFORMERS_PIN = "4.46.*"
ACCELERATE_PIN = "1.1.*"

def pip_install(*specs):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *specs]
    print("installing:", " ".join(specs))
    subprocess.run(cmd, check=True)

pip_install(
    f"transformers=={TRANSFORMERS_PIN}",
    f"accelerate=={ACCELERATE_PIN}",
)

print("W3D2 install complete")
```

Load the model in FP16:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="cuda",
)

print("Model loaded:", MODEL)
print("GPU:", torch.cuda.get_device_name(0))
```

Expected:

```text
Model loaded: Qwen/Qwen2.5-1.5B-Instruct
GPU: Tesla T4
```

---

### Cell 2: TTFT and TPOT (about 30 min)

This cell measures streaming generation using `TextIteratorStreamer`.

The important distinction is:

* **TTFT:** time until the first generated token
* **TPOT:** average time between generated tokens

The prompt lengths are:

* 128 tokens
* 512 tokens
* 2048 tokens

```python
import time
import threading
from transformers import TextIteratorStreamer

def prompt_of_len(n_tokens: int) -> str:
    base = "Explain the following in detail.\n"
    filler = "A data center serves many inference requests at once. " * 600
    ids = tok(base + filler)["input_ids"][:n_tokens]
    return tok.decode(ids)

def measure_stream(prompt: str, new_tokens: int = 128):
    enc = tok(prompt, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(
        tok,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    kwargs = dict(
        **enc,
        max_new_tokens=new_tokens,
        do_sample=False,
        streamer=streamer,
    )

    th = threading.Thread(
        target=model.generate,
        kwargs=kwargs,
    )

    t0 = time.time()
    th.start()

    stamps = []

    for _ in streamer:
        stamps.append(time.time())

    th.join()

    ttft = stamps[0] - t0

    if len(stamps) > 1:
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        tpot = sum(gaps) / len(gaps)
    else:
        tpot = 0.0

    total = stamps[-1] - t0

    return {
        "ttft_s": round(ttft, 4),
        "tpot_s": round(tpot, 4),
        "total_s": round(total, 4),
        "n_tokens": len(stamps),
    }
```

Run the three prompt lengths:

```python
ttft_by_len = {}

for n in [128, 512, 2048]:
    r = measure_stream(prompt_of_len(n))
    ttft_by_len[str(n)] = r["ttft_s"]
    print(n, r)
```

### Results

The measured results were:

| Prompt Length | TTFT (s) | TPOT (s) |
| ------------: | -------: | -------: |
|           128 |   0.0336 |   0.0322 |
|           512 |   0.0693 |   0.0380 |
|          2048 |   0.3373 |   0.0320 |

TTFT rises as the prompt gets longer.

That is the **prefill phase**: the model must read the entire prompt before producing the first output token.

TPOT represents the **decode phase**, where tokens are generated one at a time.

---

### Cell 3: KV cache growth (about 30 min)

For Qwen2.5-1.5B, the KV cache formula is:

```text
2 × layers × KV heads × head dimension × bytes
```

Using:

```text
2 × 28 × 2 × 128 × 2
```

gives:

```text
28 KB/token
```

The following helper measures the theoretical KV cache and the memory observed during generation:

```python
import gc

def kv_formula_kb_per_token(
    layers=28,
    kv_heads=2,
    head_dim=128,
    dbytes=2,
):
    return 2 * layers * kv_heads * head_dim * dbytes / 1024

def cache_bytes(pkv):
    if hasattr(pkv, "key_cache"):
        tensors = list(pkv.key_cache) + list(pkv.value_cache)
    else:
        tensors = [t for layer in pkv for t in layer]

    return sum(
        t.numel() * t.element_size()
        for t in tensors
    )

def measure_kv(context: int, new_tokens: int = 256):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    enc = tok(
        prompt_of_len(context),
        return_tensors="pt",
    ).to("cuda")

    before = torch.cuda.memory_allocated()

    out = model.generate(
        **enc,
        max_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
    )

    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    total_tokens = out.sequences.shape[1]

    return {
        "context": context,
        "total_tokens": int(total_tokens),
        "peak_kb_per_token": round(
            (peak - before) / total_tokens / 1024,
            1,
        ),
        "kv_kb_per_token": round(
            cache_bytes(out.past_key_values)
            / total_tokens
            / 1024,
            1,
        ),
    }
```

Run it at:

* 512
* 2048
* 4096

```python
formula = kv_formula_kb_per_token()

print("formula KB/token:", formula)

kv_rows = [
    measure_kv(c)
    for c in [512, 2048, 4096]
]

for r in kv_rows:
    print(r, "vs formula", formula, "KB/token")
```

### Results

| Context | Total Tokens |      KV Cache |   Peak Memory |
| ------: | -----------: | ------------: | ------------: |
|     512 |          768 | 28.0 KB/token | 63.4 KB/token |
|    2048 |         2304 | 28.0 KB/token | 84.0 KB/token |
|    4096 |         4352 | 28.0 KB/token | 87.6 KB/token |

The theoretical value is:

```text
28 KB/token
```

and the measured KV cache was also:

```text
28.0 KB/token
```

The larger peak-memory value includes other memory used during inference, such as activations and allocator/workspace overhead.

Save the KV result:

```python
import json

with open("kv_check.json", "w") as f:
    json.dump(
        {
            "formula_kb_per_token": formula,
            "measured_kb_per_token": kv_rows[-1]["kv_kb_per_token"],
            "peak_kb_per_token": kv_rows[-1]["peak_kb_per_token"],
        },
        f,
        indent=2,
    )

print("kv_check.json saved")
```

---

### Cell 4: static batching (about 40 min)

Now build a simple static batching queue.

The queue contains mixed output lengths:

```python
QUEUE = [32, 32, 32, 256] * 6
```

This creates the **straggler effect**.

The shorter requests finish earlier, but the batch continues until the longest request finishes.

```python
QUEUE = [32, 32, 32, 256] * 6

def static_queue(
    batch: int,
    prompt: str = "Explain what an inference server does.",
):
    t0 = time.time()
    useful = 0
    slots = 0

    for i in range(0, len(QUEUE), batch):
        chunk = QUEUE[i:i + batch]
        n = max(chunk)

        enc = tok(
            [prompt] * len(chunk),
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        model.generate(
            **enc,
            max_new_tokens=n,
            do_sample=False,
        )

        useful += sum(chunk)
        slots += n * len(chunk)

    dt = time.time() - t0

    return {
        "batch": batch,
        "wall_s": round(dt, 2),
        "tokens_per_s": round(useful / dt, 1),
        "slot_efficiency": round(useful / slots, 3),
    }
```

Run batch sizes:

```python
batch_rows = {}

for n in [1, 4, 8]:
    r = static_queue(n)
    batch_rows[str(n)] = r["tokens_per_s"]
    print(r)
```

### Results

| Batch Size | Wall Time (s) | Tokens/s | Slot Efficiency |
| ---------: | ------------: | -------: | --------------: |
|          1 |         60.86 |     34.7 |           1.000 |
|          4 |         40.44 |     52.2 |           0.344 |
|          8 |         22.21 |     95.1 |           0.344 |

Throughput increases with batch size:

```text
Batch 1 → 34.7 tokens/s
Batch 4 → 52.2 tokens/s
Batch 8 → 95.1 tokens/s
```

But slot efficiency drops because the shorter requests are effectively waiting for the longer 256-token request.

This is the main limitation of **static batching**.

The result helps explain why inference engines such as vLLM use more advanced batching strategies.

---

### Cell 5: write baselines.json (about 10 min)

Save the main measurements for W3D3.

```python
import json

baselines = {
    "model": MODEL,
    "dtype": "fp16",
    "ttft_s": ttft_by_len,
    "tpot_s": measure_stream(
        prompt_of_len(512)
    )["tpot_s"],
    "batch": {
        k: v
        for k, v in batch_rows.items()
    },
}

with open("baselines.json", "w") as f:
    json.dump(
        baselines,
        f,
        indent=2,
    )

print(json.dumps(baselines, indent=2))
```

The saved baseline contains:

```text
model
dtype
ttft_s
tpot_s
batch
```

### Baseline Results

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "dtype": "fp16",
  "ttft_s": {
    "128": 0.0336,
    "512": 0.0693,
    "2048": 0.3373
  },
  "tpot_s": 0.0479,
  "batch": {
    "1": 34.7,
    "4": 52.2,
    "8": 95.1
  }
}
```

**Important:** Download `baselines.json` before closing Colab. W3D3 uses this file for the vLLM comparison.

```python
from google.colab import files

files.download("baselines.json")
```

---

## Verify (green check)

Run the W3D2 verifier as the last cell.

It checks:

* `baselines.json` exists and has the required fields.
* TTFT increases from 128 to 2048 tokens.
* Batch 8 throughput is higher than Batch 1.
* KV cache is within 2× of the theoretical **28 KB/token** value.

Run:

```python
!python verify_cell.py
```

Expected final line:

```text
GREEN CHECK: PASS
```

If it fails, the parenthesis tells you which rule failed. Fix the measurement or file rather than changing the verifier.

---

## Key Takeaways

* **TTFT increases with prompt length** because of prefill.
* **TPOT** measures the time between generated output tokens during decode.
* Qwen2.5-1.5B FP16 has a KV cache cost of approximately **28 KB/token**.
* KV cache grows with the number of tokens stored.
* **Batching increases throughput**.
* Static batching can waste capacity when requests have different output lengths.
* The **straggler effect** lowers slot efficiency.
* These measurements are the **baseline for W3D3**, where the same workload will be tested with vLLM.

---

## Files

The important W3D2 files are:

```text
baselines.json
kv_check.json
verify_cell.py
```

### `baselines.json`

Contains:

* TTFT
* TPOT
* Batch throughput

### `kv_check.json`

Contains:

* Theoretical KV cache
* Measured KV cache
* Peak memory

### `verify_cell.py`

Contains the W3D2 green-check verification.

---

## Before you close the tab

Colab runtimes disappear, so save the important artifacts before ending the session.

Download:

```python
from google.colab import files

for f_ in [
    "baselines.json",
    "kv_check.json",
]:
    files.download(f_)
```

The most important file for W3D3 is:

```text
baselines.json
```

Keep it safe because it is the **Transformers baseline** that will be compared with vLLM.
