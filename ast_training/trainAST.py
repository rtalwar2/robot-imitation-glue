from datasets import load_dataset
import numpy as np
from transformers import (
    ASTForAudioClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
import evaluate

# ------------------------------
# 1. Load datasets
# ------------------------------

train_ds = load_dataset(
    "ramen-noodels/delta_z_train_spectrogram_labeled_unnormalized",
    split="train"
)

val_ds = load_dataset(
    "ramen-noodels/delta_z_val_spectrogram_labeled_unnormalized",
    split="train"
)


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
    num_labels=2
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
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)


# ------------------------------
# 6. TrainingArguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./ast_delta_z",
    eval_strategy="epoch",            # Evaluate every epoch
    save_strategy="epoch",            # Save checkpoint every epoch
    learning_rate=1e-5,               # Low LR for finetuning (as per paper)
    per_device_train_batch_size=45,    # Adjust based on GPU VRAM
    per_device_eval_batch_size=45,
    num_train_epochs=10,              # As per paper
    load_best_model_at_end=True,      # Select best checkpoint
    metric_for_best_model="accuracy", 
    save_total_limit=1,               # Save space
    logging_steps=20,
    remove_unused_columns=False,      # Important when using custom inputs
    push_to_hub=True,
)

# ------------------------------
# 7. Trainer
# ------------------------------

trainer = Trainer(
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



# import numpy as np
# import evaluate
# from transformers import ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# from datasets import load_dataset

# # ==========================================
# # 1. SETUP DUMMY DATA (Replace with your data)
# # ==========================================
# # Assuming fixed length. If not, you need to pad/truncate to max_length.
# # AST Base usually expects 1024 frames, but works with variable lengths if configured.
# train_ds = load_dataset("ramen-noodels/delta_z_spectrogram_labeled_fixeddz")
# print(train_ds.train)
# # ==========================================
# # 2. MODEL CONFIGURATION
# # ==========================================
# # We use the AudioSet pretrained model as a base
# model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return {"accuracy": accuracy_score(labels, predictions)}

# # ==========================================
# # 3. CROSS VALIDATION LOOP
# # ==========================================
# k_folds = 5
# skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
# fold_accuracies = []

# print(f"Starting {k_folds}-fold Cross Validation...")

# for fold, (train_idx, val_idx) in enumerate(skf.split(spectrograms, labels)):
#     print(f"\n--- Fold {fold + 1}/{k_folds} ---")
    
#     # Split data
#     train_specs, val_specs = spectrograms[train_idx], spectrograms[val_idx]
#     train_labels, val_labels = labels[train_idx], labels[val_idx]
    
#     # Create HF Datasets with normalization
#     train_ds = create_hf_dataset(train_specs, train_labels, dataset_mean, dataset_std)
#     val_ds = create_hf_dataset(val_specs, val_labels, dataset_mean, dataset_std)

#     # Load Pretrained Model
#     # ignore_mismatched_sizes=True allows replacing the 527-class head with a 2-class head
#     model = ASTForAudioClassification.from_pretrained(
#         model_checkpoint,
#         num_labels=2,
#         ignore_mismatched_sizes=True
#     )

#     # Configure Training
#     training_args = TrainingArguments(
#         output_dir=f"./results/fold_{fold}",
#         eval_strategy="epoch",            # Evaluate every epoch
#         save_strategy="epoch",            # Save checkpoint every epoch
#         learning_rate=1e-5,               # Low LR for finetuning (as per paper)
#         per_device_train_batch_size=4,    # Adjust based on GPU VRAM
#         per_device_eval_batch_size=4,
#         num_train_epochs=10,              # As per paper
#         load_best_model_at_end=True,      # Select best checkpoint
#         metric_for_best_model="accuracy", 
#         save_total_limit=1,               # Save space
#         logging_steps=10,
#         remove_unused_columns=False,      # Important when using custom inputs
#         push_to_hub=True,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         compute_metrics=compute_metrics,
#     )

#     # Train
#     trainer.train()

#     # Evaluate on the validation fold (Test performance for this fold)
#     metrics = trainer.evaluate()
#     acc = metrics["eval_accuracy"]
#     fold_accuracies.append(acc)
#     print(f"Fold {fold+1} Accuracy: {acc:.4f}")

# # ==========================================
# # 4. AGGREGATE RESULTS
# # ==========================================
# mean_acc = np.mean(fold_accuracies)
# std_acc = np.std(fold_accuracies)

# print("\n==============================")
# print("Final 5-Fold Cross-Validation Results")
# print("==============================")
# print(f"Accuracies: {fold_accuracies}")
# print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
