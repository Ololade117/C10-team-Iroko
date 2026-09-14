# Iroko —  Mental Health Support Chatbot

Iroko is a fine-tuned conversational AI built to provide culturally appropriate, safety-conscious mental health support for a Nigerian context. It fine-tunes Google's Gemma 4 E2B model on a mental health counseling dataset, with a pipeline spanning data preparation, training, safety evaluation, and deployment.

- Model (Hugging Face Hub): `Ololade117/gemma-4-e2b-iroko-mentalhealth-finetuned5`

## Dataset

The training data combines an existing mental-health counseling conversation dataset, from counsel health, mental health faq, anno mi, excov, with a purpose-built safety set, then splits into train/validation/test.

**Base counseling data** was cleaned before use:
- **HTML artifact stripping** — removed raw `&nbsp;`, `<br>` tags leaked from the source into responses.
- **Fabricated identity/credential scrubbing** — removed self-introduced fake names, workplaces, and false licensure claims (e.g. "Hi, I'm Karen, I work with family services...") and fabricated sign-offs.
- **Transcript artifact removal** — stripped real therapist names and markers like `[unintelligible]`/`[crosstalk]` from transcript-sourced examples.
- **Nigeria localization** — replaced US-centric crisis resources (988, 911) with local ones: **Mentally Aware Nigeria Initiative (MANI)** — 0809 111 6264 / 0811 1680 686 — and **112**, the national emergency line.

**Safety augmentation set** was generated, not scraped:
- Adversarial refusal examples were generated using the **clean, un-fine-tuned base model**, so the LoRA adapter learns to preserve refusal behavior rather than drift from it.
- Prompts targeted real failure categories: direct method-seeking (weapons, poison), "as a joke" framing, euphemisms, and third-person displacement.
- Every example was **manually reviewed** before inclusion, never trusted as ground truth automatically.
- Mixed into training at ~**20%** of examples, kept diverse rather than duplicated.

## Training Pipeline

**1. Data collection & preprocessing** — combine and clean the counseling data and generated safety set as above, then split into train/validation/test.

**2. Model & method:**
- Base: `google/gemma-4-E2B-it`, 4-bit quantized (NF4, bitsandbytes) to fit limited hardware.
- Fine-tuning: LoRA (rank 16, alpha 32) via PEFT, targeting attention/MLP projections only — vision/audio towers excluded since this is text-only.
- Precision: bf16 throughout, after fp16 caused `GradScaler` incompatibilities on this architecture.
- Trainer: TRL's `SFTTrainer`, with checkpointing, validation-loss tracking, and early stopping (patience = 3 on `eval_loss`).

**3. Architecture-specific fixes for Gemma 4 E2B:**
- Targeted norm-upcasting rather than blanket k-bit preparation.
- Manual unwrapping of `Gemma4ClippableLinear` layers before LoRA injection.
- Memory-constrained loading (`max_memory`, `low_cpu_mem_usage`) for free-tier GPUs.

**4. Hyperparameters** (LoRA rank/alpha, early-stopping patience, safety-data mixing ratio) were set from iterative runs on Colab/Kaggle, prioritizing stable convergence and preserved refusal behavior over raw loss reduction.

## Evaluation

Evaluation deliberately treats training loss as necessary but not sufficient for judging safety — early runs showed loss decreasing while safety failures were still present.

- **Validation loss** was tracked as a basic convergence check (2.283 → 2.093 in the latest run; training loss 6.25 → ~2.10–2.25; token accuracy 49.4% → 52.3%; entropy 2.22 → 2.09).
- **Manual log review** of real conversations surfaced six recurring failure categories: bypassable keyword filtering, the model engaging with harmful requests, fabricated clinical identities, inconsistent responses to near-identical risky prompts, text-degeneration loops, and uneven quality on adversarial prompts — each traced to a specific fix (regex-based risk detection, output-side moderation, greedy decoding under detected risk, repetition penalties).
- **Red-team evaluation** is the primary safety gate: a fixed set of paraphrased adversarial prompts, held out from training, run against each model version and manually scored pass/fail/borderline per category — the metric that actually answers whether an iteration improved safety, since loss alone does not.
- **Deployment checks**: crisis-detection recall, consistency rate (same input → same correct response), and latency (cold start ~145s, warm ~20–25s on ZeroGPU).


## Reproduction

1. **Prepare data** — run the cleaning scripts/notebooks in `data/`/`notebooks/` to strip HTML artifacts, scrub fabricated identities, remove transcript markers, and localize crisis resources; generate and manually review the adversarial safety set, then mix it in at ~20%.
2. **Split** the data into train/validation/test sets.
3. **Fine-tune** via the training notebook/script in `src/` (or `notebooks/`), which loads `google/gemma-4-E2B-it` in 4-bit, applies LoRA (rank 16, alpha 32), and trains with `SFTTrainer` (bf16, early stopping on `eval_loss`).
4. **Evaluate** the checkpoint against the held-out red-team prompt set (pass/fail/borderline per category); compare validation loss and token accuracy against the prior version.
5. **Push** the checkpoint to the Hugging Face Hub once it passes evaluation.
6. **Deploy** by wrapping the model in the multi-layer safety harness (input keyword/regex layers, output moderation, decoding controls) and serving via a Hugging Face Space (Gradio, ZeroGPU); configure the Google Sheets service account for logging.

Refer to `doc/` for supporting write-ups and `data/` for the underlying dataset files.

## Appendix

**Team members (Team Iroko):**
- Ololade Ogunleye
- Lawal Habib
- Stephanie Omolu
- Alemoh Rapheal

**Mentors:** 

## References

- Repository: https://github.com/Ololade117/C10-team-Iroko
- Model (Hugging Face Hub): https://huggingface.co/Ololade117/gemma-4-e2b-iroko-mentalhealth-finetuned5
