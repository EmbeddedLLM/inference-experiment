# Sleep Mode in vLLM

To learn more about vLLM sleep mode, please visit the official vLLM blog post: [vLLM Sleep Mode](https://blog.vllm.ai/2025/10/26/sleep-mode.html)

You can find the experiment scripts of sleep mode under this directory.

## 1. Sleep Mode vs. No Sleep Mode

This test compares the end-to-end performance of sleep mode (enabled) versus nosleep mode (i.e., fully unloading the model).

- **Script:** `online_default.py`
- **Corresponding Result:** Table 1 (Sleep mode vs No sleep mode)

### Reproduction Commands

**Sleep Mode (Test Group):**
```bash
python online_default.py sleep > out_info_sleep.log 2>&1
```

**No Sleep Mode (Control Group):**
```bash
python online_default.py nosleep > out_info_nosleep.log 2>&1
```

---

## 2. Sleep Mode (No FP8) vs. Sleep Mode (FP8)

This test compares the performance difference between enabling and disabling FP8 quantization while in Sleep Mode.

- **Script:** `online_fp8.py`
- **Corresponding Result:** Table 2 (Sleep mode(without fp8 vs fp8))

### Reproduction Commands

**No FP8 (Test Group):**
```bash
python online_fp8.py > out_info.log 2>&1
```

**FP8 (Control Group):**
```bash
python online_fp8.py fp8 > out_info_fp8.log 2>&1
```

---

## 3. Sleep Mode (Warmup) vs. Sleep Mode (No Warmup)

This test compares the performance difference in Sleep Mode with and without a warmup phase.

- **Script:** `online_sleep1_nowarm.py`
- **Corresponding Result:** Table 3 (Sleep mode vs vLLM 0.11.0 no warmup)

### Reproduction Commands

**With Warmup (Test Group):**
```bash
python online_sleep1_nowarm.py sleep1 > out_info_sleep1.log 2>&1
```

**No Warmup (Control Group):**
```bash
python online_sleep1_nowarm.py nowarmup > out_info_nowarmup.log 2>&1
```

---

## 4. Sleep Mode (Level 1) vs. Sleep Mode (Level 2)

This test compares Sleep Mode Level 1 (default, retains weights) with Level 2 (weights and KV cache are offloaded to CPU).

- **Script:** `online_sleep1_2.py`
- **Corresponding Result:** Table 4 (Sleep mode(without fp8) vs Sleep level 2 wake + reload weights)

### Reproduction Commands

**Sleep Level 1 (Test Group):**
```bash
python online_sleep1_2.py sleep1 > out_info_sleep1.log 2>&1
```

**Sleep Level 2 (Control Group):**
```bash
python online_sleep1_2.py sleep2 > out_info_sleep2.log 2>&1
```

---

## 5. Sleep Mode (Level 2) vs. No Sleep Mode

This test compares the end-to-end performance of Sleep Mode Level 2 against nosleep mode (i.e., fully unloading the model).

- **Script:** `online_sleep2_nosleep.py`
- **Corresponding Result:** Table 5 (Sleep level 2 wake + reload weights vs No sleep mode)

### Reproduction Commands

**Sleep Level 2 (Test Group):**
```bash
python online_sleep2_nosleep.py sleep2 > out_info_sleep2.log 2>&1
```

**No Sleep Mode (Control Group):**
```bash
python online_sleep2_nosleep.py nosleep > out_info_nosleep.log 2>&1
```
