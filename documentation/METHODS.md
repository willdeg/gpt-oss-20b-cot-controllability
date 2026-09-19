# Methods and reproducibility details

## SDF corpus

One universe statement and five supporting key facts were used. For every key
fact, ten document types and ten ideas per type were produced, yielding 500
documents. The final corpus was reviewed and revised for realism and consistency.
Generation instructions explicitly prohibited worked CoT-control examples,
reasoning traces, prompt-response transcripts, and recognizable benchmark items.

Final corpus statistics after tokenization:

- Documents: 500
- Tokens: 302,703
- Mean tokens/document: 605.41
- Maximum tokens/document: 1,107
- Documents truncated at the 4,096-token limit: 0

## Stronger SDF fine-tune

- Base model: `openai/gpt-oss-20b`
- Method: causal-language-model training with LoRA
- Epochs: 5
- Learning rate: `1e-5`
- Per-device batch size: 1
- Gradient accumulation: 8
- Maximum sequence length: 4,096
- Precision: bfloat16
- Attention implementation: eager
- Gradient checkpointing: enabled
- LoRA rank/alpha/dropout: 64 / 128 / 0.0
- LoRA targets: all linear layers
- Seed: 42
- Hardware: one NVIDIA A100-SXM4-80GB
- Runtime: 3,601.8 seconds
- Final training loss: 1.8698

Recorded software versions were PyTorch 2.13.0+cu130, Transformers 4.57.6,
PEFT 0.20.0, Datasets 3.5.0, Accelerate 1.14.0, and Kernels 0.12.3.

The model checkpoint was loaded using `Mxfp4Config(dequantize=True)` and trained
in bfloat16. No held-out validation split was used.

## CoT-control evaluation

The 60-question sample was created with seed 42 and contains 20 questions each
from HLE, MMLU-Pro, and GPQA. Generation was deterministic (`do_sample=False`),
used medium reasoning effort, and was capped at 1,024 new tokens.

Uppercase compliance was computed over every Unicode alphabetic character in
the analysis channel. A trace was fully uppercase only when it contained at
least one alphabetic character and zero lowercase alphabetic characters.
Answers were parsed from the final channel, with a fallback to an `ANSWER: X`
string in the analysis channel.

## Belief evaluation

The paired MCQ-distinguish test contained 25 A/B questions spanning identity,
reasoning control, persistence, generalization, and answer-quality preservation.
The position of the inserted-universe answer was randomized. Each question was
asked once with the stronger adapter active and once with it disabled, in an
interleaved randomized schedule (seed 20260905). Both conditions used the same
identity system prompt.
