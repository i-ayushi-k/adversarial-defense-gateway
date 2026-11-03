# Baseline Research Implementation

## Reference Paper
Defenses Against Prompt Attacks Learn Surface Heuristics

## Original Repository
https://github.com/AiHeMaotai/MEval

This folder documents the external research baseline used in our capstone project.

The original implementation remains in its authors' repository. Our team uses it separately in Google Colab for research and experimentation.

Our own implementation and modifications will be maintained separately in this repository.

# Baseline: DistilBERT prompt-injection classifier

- Model: distilbert-base-uncased, fine-tuned on deepset/prompt-injections (491 train / 55 val / 116 test), seed 42, 3 epochs, CPU-only (~12 min)
- Train: `python baseline/train.py --model distilbert-base-uncased --bs 8`
- Over-defense eval: `python baseline/eval_overdefense.py` (NotInject, following Li et al., ACL 2026)

| Test | Result |
|---|---|
| Accuracy / Precision / Recall / F1 (deepset test) | 0.922 / 1.00 / 0.85 / 0.919 |
| NotInject false-positive rate (1 / 2 / 3 trigger words) | 0.867 / 0.496 / 0.858 |

Note: small training set, so over-defense may partly reflect limited benign diversity.
