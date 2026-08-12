import os
import sys

import wandb
from huggingface_hub import login

# Read credentials from the environment rather than hardcoding them, so they never land in git.
# Set them once in your shell (or a .env you do not commit):
#   export WANDB_API_KEY=...   export HF_TOKEN=...
if "WANDB_API_KEY" not in os.environ:
    raise SystemExit("WANDB_API_KEY is not set; export it before running this script")
wandb.login()

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise SystemExit("HF_TOKEN is not set; export it before running this script (push_to_hub needs it)")
login(hf_token)

from datasets import load_dataset
import numpy as np
from transformers import (
    ASTForAudioClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
import evaluate
from torch import nn

# ------------------------------
# 1. Load datasets
# ------------------------------

train_ds = load_dataset(
    sys.argv[-3],
    split="train"
)

val_ds = load_dataset(
    sys.argv[-2],
    split="train"
)

# val_ds = load_dataset(
#     "ramen-noodels/delta_z_val_spectrogram_labeled_unnormalized",
#     split="train"
# )


# ------------------------------
# 2. Compute normalization stats from TRAIN ONLY
# ------------------------------

all_train_specs = np.stack(train_ds["input_values"])  # shape: (N, T, F)
print(all_train_specs.shape)

time_dimension = all_train_specs.shape[1] 
print(time_dimension)
mean = float(all_train_specs.mean())
std = float(all_train_specs.std())

print("AST normalization mean =", mean)
print("AST normalization std  =", std)


# ------------------------------
# 3. AST-style normalization function
#    norm(x) = (x - mean) / (std * 2)
# ------------------------------

def normalize_ast(batch):
    arr = np.array(batch["input_values"], dtype=np.float32)
    batch["input_values"] = (arr - mean) / (std * 2)
    return batch

train_ds = train_ds.map(normalize_ast)
val_ds  = val_ds.map(normalize_ast)


# ------------------------------
# 4. Load AST model with positional interpolation
# ------------------------------

model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"

config = AutoConfig.from_pretrained(
    model_checkpoint,
    num_labels=1
)

# CRITICAL FIX: Update the config to match your actual input size
config.max_length = time_dimension

model = ASTForAudioClassification.from_pretrained(
    model_checkpoint,
    config=config,
    ignore_mismatched_sizes=True,  # Rescales positional embeddings for 300-frame spectrograms
)


# ------------------------------
# 5. Metrics
# ------------------------------

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Squeeze to ensure shapes match: (N, 1) -> (N,)
    logits = np.squeeze(logits)
    labels = np.squeeze(labels)
    
    # If logit > 0, prediction is 1, else 0
    preds = (logits > 0).astype(int)

    # POSITIVE CLASS = 0 (button pressed) - Keeping your logic
    tp = np.sum((preds == 0) & (labels == 0))
    fn = np.sum((preds == 1) & (labels == 0))
    fp = np.sum((preds == 0) & (labels == 1))

    # Metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "recall_pressed": recall,
        "precision_pressed": precision,
        "f1_pressed": f1,
        "false_negative_rate": fnr,
    }



class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Pop labels and ensure they are floats for BCE Loss
        labels = inputs.pop("labels").float()
        
        # 2. Forward pass
        outputs = model(**inputs)
        
        # 3. Get logits and squeeze them from shape (Batch, 1) to (Batch,)
        logits = outputs.logits.squeeze(-1)
        
        # 4. Compute Binary Cross Entropy Loss
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
        
# ------------------------------
# 6. TrainingArguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./" + sys.argv[-1],
    eval_strategy="epoch",            # Evaluate every epoch
    save_strategy="epoch",            # Save checkpoint every epoch
    learning_rate=1e-5,               # Low LR for finetuning (as per paper)
    per_device_train_batch_size=82,   # Adjust based on GPU VRAM
    per_device_eval_batch_size=82,
    num_train_epochs=10,              # As per paper
    load_best_model_at_end=True,      # Select best checkpoint
    metric_for_best_model="f1_pressed", 
    save_total_limit=1,               # Save space
    logging_steps=20,
    remove_unused_columns=False,      # Important when using custom inputs
    push_to_hub=True,
    report_to="wandb"
)

# ------------------------------
# 7. Trainer
# ------------------------------

trainer = BCETrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# ------------------------------
# 8. Train
# ------------------------------

trainer.train()


# ------------------------------
# 9. Evaluate
# ------------------------------

metrics = trainer.evaluate()
print("Final evaluation:", metrics)


# ------------------------------
# 10. Push model to Hub
# ------------------------------

trainer.push_to_hub()
