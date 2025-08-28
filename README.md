# Selective Mixup for Debiasing Question Selection in Computerized Adaptive Testing

This is the official repository for the paper **Selective Mixup for Debiasing Question Selection in Computerized Adaptive Testing**.

Computerized Adaptive Testing (CAT) enables personalized assessment by dynamically selecting questions according to an examinee’s estimated proficiency, updating these estimates iteratively. However, this process is prone to **selection bias**: the choice of questions can become skewed for certain examinee groups, resulting in diagnostic models that are misaligned and biased.

To address this, we propose a debiasing framework that incorporates **Cross-Attribute Examinee Retrieval** and **Selective Mixup-based Regularization**. This approach increases the diversity of bias-conflicting samples and improves both the generalization and fairness of the question selection process.

---

## Environment Setup

This repository requires the following Python 3 packages. Additionally, the backbone implementation is based on [BOBCAT](https://github.com/arghosh/BOBCAT).

```bash
torch==2.4.0
tensorboard
scikit-learn
```

---

## Training Procedure

### 1. Pretraining

- Run `pretrain.py`, through `train.sh`.
- After pretraining, update the model save path in `utils/model_cfg.json`.

### 2. Training
- Run `train.py`, through `train.sh`.
---