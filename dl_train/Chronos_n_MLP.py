"""
DL_ChronosClassifier.py

Connects Chronos-2's encoder hidden states to an MLP classifier for
bifurcation type classification (classes 0-3).

Architecture
------------
Chronos-2 encoder (frozen)
    |
    | pipeline.embed(context) -> (batch, num_patches+1, hidden_dim=512)
    |                            num_patches depends on series length/patch size
    v
Mean pooling across patch dimension -> (batch, 512)
    |
    v
MLP classifier head:
    Linear(512, 256) -> LayerNorm -> ReLU -> Dropout
    Linear(256, 64)  -> LayerNorm -> ReLU -> Dropout
    Linear(64, 4)    -> logits (CrossEntropyLoss applies softmax internally)

Why frozen encoder
------------------
Only the MLP head (~140k params) trains, not Chronos-2's 120M params.
This is fast, stable, and avoids destroying pretrained representations.
If this gives poor accuracy (<50%), set FREEZE_ENCODER=False to jointly
fine-tune the encoder — but expect much longer training time and the need
for a much lower encoder learning rate (see optimizer setup below).

Why a deeper MLP than before
-----------------------------
The old flat Linear(500->128->4) MLP operated on raw timestep values with
no temporal structure. This MLP operates on Chronos-2's 512-dim embedding,
which already encodes rich temporal context — a two-layer MLP with
LayerNorm should be enough to linearly separate classes in that space.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# ── Data source options ──────────────────────────────────────────────────────
# "library"    = existing dataset from DL_training.py
# "may"        = May's harvesting model simulated through its fold (Fold-only series)
# "porpoising" = toy F1 porpoising model (Hopf + Null series), see porpoising_ews.py
# "both"       = library + may, "all" = library + may + porpoising, or any
#                "+"-joined combination, e.g. "library+porpoising".
# TRAIN_DATA controls train + validation, TEST_DATA controls the test set.
# If you use "porpoising" as an out-of-distribution TEST set, keep it out of
# TRAIN_DATA here and out of the LoRA fine-tuning (DL_Choronotraining.py).
TRAIN_DATA = "library"   # "library" | "may" | "porpoising" | "both" | "all" | "a+b"
TEST_DATA = "porpoising"    # same options as TRAIN_DATA
N_MAY_SERIES = 2000       # May series simulated in total (split 80/10/10)
N_PORP_SERIES = 2000      # porpoising series in total (half Hopf, half Null; split 80/10/10)
from data_source import get_data
DL = get_data(train_data=TRAIN_DATA, test_data=TEST_DATA,
              n_may=N_MAY_SERIES, n_porp=N_PORP_SERIES)
from chronos import BaseChronosPipeline
from peft import PeftModel

# --- Config ---
FREEZE_ENCODER = True       # set False to jointly fine-tune encoder + head
LORA_ADAPTER_DIR = "chronos2_lora_adapter"     # set to "chronos2_lora_adapter" to load your
                             # fine-tuned LoRA adapter on top of base Chronos-2
                             # before freezing and extracting embeddings

HIDDEN_DIM = 512            # Chronos-2 base encoder hidden dim
MLP_DIM1 = 256
MLP_DIM2 = 64
NUM_CLASSES = 4
DROPOUT = 0.3

BATCH_SIZE = 64             # smaller than DL_newtraining since embeddings
                             # are precomputed and stored in RAM, so GPU
                             # memory isn't the bottleneck here
LEARNING_RATE = 3e-4        # for the MLP head; if FREEZE_ENCODER=False,
                             # a separate lower LR is used for the encoder
ENCODER_LR = 1e-5           # only used when FREEZE_ENCODER=False
EPOCH_PARAM = 200           # fewer epochs needed - head-only training
                             # converges faster than full model training
PATIENCE = 30
MIN_DELTA = 1e-4
SMOOTHING_WINDOW = 5

class_names = {0: "Fold", 1: "Hopf", 2: "Transcritical", 3: "Null"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running on: {device} ---")


# ── 1. Load Chronos-2 pipeline ───────────────────────────────────────────────

print("Loading Chronos-2 pipeline...")
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)

if LORA_ADAPTER_DIR is not None:
    from peft import PeftModel
    print(f"Attaching LoRA adapter from '{LORA_ADAPTER_DIR}'...")
    pipeline.model = PeftModel.from_pretrained(pipeline.model, LORA_ADAPTER_DIR)
    pipeline.model.eval()

if FREEZE_ENCODER:
    print("Freezing Chronos-2 encoder weights...")
    for param in pipeline.model.parameters():
        param.requires_grad = False
    pipeline.model.eval()


# ── 2. Pre-extract embeddings for all splits ─────────────────────────────────
# We run pipeline.embed() over the full dataset ONCE and cache the mean-pooled
# embeddings as plain tensors. This means:
#   - Training the MLP head only ever touches cached (batch, 512) vectors,
#     not the 120M-param encoder on every step - much faster per epoch.
#   - If FREEZE_ENCODER=True this is always correct (frozen encoder ->
#     identical embeddings every forward pass anyway).
#   - If FREEZE_ENCODER=False, the cached embeddings become stale after
#     each encoder update, so we re-extract every epoch instead (see below).

def extract_embeddings(series_array: np.ndarray, batch_size: int = 32) -> torch.Tensor:
    """
    Run pipeline.embed() on series_array in batches and mean-pool the patch
    embeddings into a single (N, hidden_dim) tensor.

    series_array: (N, ts_len, 1) numpy array  [from DL_training.py format]

    pipeline.embed() expects a list of 1D tensors or a (batch, time) tensor.
    We squeeze the trailing channel dim before passing in.

    Returns: (N, hidden_dim) float32 tensor on CPU.
    """
    all_embeddings = []
    n = len(series_array)

    pipeline.model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = series_array[start : start + batch_size]
            # squeeze (batch, ts_len, 1) -> (batch, ts_len) for embed()
            ctx = torch.tensor(batch, dtype=torch.float32).squeeze(-1)
            # embed() returns (batch, num_patches+1, hidden_dim)
            # last position is EOS token - include it in max pooling since
            # it aggregates global sequence information
            if ctx.ndim == 2:
                ctx = ctx[:,None,:]
            embeddings, _ = pipeline.embed(ctx)
            if isinstance(embeddings, list):
                embeddings = torch.stack(embeddings, dim=0)

            # embeddings: (batch, 1, num_patches, hidden_dim) -> squeeze variate dim
            if embeddings.ndim == 4:
                embeddings = embeddings.squeeze(1)   # -> (batch, num_patches, hidden_dim)

            pooled = embeddings.max(dim=1).values.cpu().float()   # -> (batch, hidden_dim)
            all_embeddings.append(pooled)

    return torch.cat(all_embeddings, dim=0)


print("\nExtracting Chronos-2 embeddings (this runs once per split)...")
print(f"  Train: {len(DL.train)} series...")
train_embeddings = extract_embeddings(DL.train, batch_size=BATCH_SIZE)
print(f"  Val:   {len(DL.validation)} series...")
val_embeddings = extract_embeddings(DL.validation, batch_size=BATCH_SIZE)
print(f"  Test:  {len(DL.test)} series...")
test_embeddings = extract_embeddings(DL.test, batch_size=BATCH_SIZE)
print(f"Embedding shape per split: {train_embeddings.shape}")

train_labels = torch.tensor(DL.train_target, dtype=torch.long)
val_labels = torch.tensor(DL.validation_target, dtype=torch.long)
test_labels = torch.tensor(DL.test_target, dtype=torch.long)


# ── 3. MLP classifier head ───────────────────────────────────────────────────

class ChronosMLPClassifier(nn.Module):
    """
    Two-hidden-layer MLP that operates on pre-extracted Chronos-2 embeddings.
    LayerNorm instead of BatchNorm since batch sizes can be small and
    LayerNorm is more stable across varying batch sizes during eval.
    """
    def __init__(self, input_dim=512, dim1=256, dim2=64, num_classes=4, dropout=0.3):
        super(ChronosMLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.LayerNorm(dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim1, dim2),
            nn.LayerNorm(dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


embed_dim = train_embeddings.shape[1]
print(f"\nBuilding MLP classifier head (input_dim={embed_dim})...")
classifier = ChronosMLPClassifier(
    input_dim=embed_dim,
    dim1=MLP_DIM1,
    dim2=MLP_DIM2,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(device)

total_head_params = sum(p.numel() for p in classifier.parameters())
print(f"MLP head trainable params: {total_head_params:,}")


# ── 4. Optimizer, loss, DataLoader ───────────────────────────────────────────

if FREEZE_ENCODER:
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
else:
    # Separate param groups: lower LR for encoder, higher for head.
    # Without this, the large LR that's appropriate for the head would
    # destroy Chronos-2's pretrained representations.
    optimizer = torch.optim.AdamW([
        {"params": pipeline.model.parameters(), "lr": ENCODER_LR},
        {"params": classifier.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=1e-4)

# Class-weighted loss: class 0 ("no bifurcation / stable") was completely
# missed by the Conv1D+LSTM model (0/25 correct). Weighting inversely by
# class frequency in the training set pushes the optimizer to treat it
# equally with the other classes rather than ignoring its harder signal.
class_counts = np.bincount(DL.train_target, minlength=NUM_CLASSES)
# Classes absent from the training set get weight 0 (not a huge weight, which
# would push the model toward classes it never sees) and the rest are
# normalised to average 1 over the classes actually present.
_inv = np.where(class_counts > 0, 1.0 / np.maximum(class_counts, 1), 0.0)
_inv = _inv / _inv.sum() * (class_counts > 0).sum()
class_weights = torch.tensor(_inv, dtype=torch.float32)
class_weights = class_weights.to(device)
print(f"Training class counts: {dict(zip(class_names.values(), class_counts))}")
print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

train_dataset = TensorDataset(train_embeddings, train_labels)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0,   # embeddings are already in RAM, no disk I/O needed
    pin_memory=(device.type == "cuda"),
)


# ── 5. Training loop ─────────────────────────────────────────────────────────

best_val_loss = float("inf")
best_smoothed_val_loss = float("inf")
epochs_no_improve = 0
val_loss_history = []

print("\n--- Starting MLP Head Training ---")

for epoch in range(EPOCH_PARAM):

    # If encoder is NOT frozen, re-extract embeddings every epoch since
    # encoder weights change each step. This is expensive (full encode pass)
    # but necessary for correctness. For frozen encoder this block is skipped.
    if not FREEZE_ENCODER:
        train_embeddings = extract_embeddings(DL.train, batch_size=BATCH_SIZE).to(device)
        val_embeddings = extract_embeddings(DL.validation, batch_size=BATCH_SIZE).to(device)
        train_dataset = TensorDataset(train_embeddings, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    classifier.train()
    if not FREEZE_ENCODER:
        pipeline.model.train()

    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_emb, batch_y in train_loader:
        batch_emb = batch_emb.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = classifier(batch_emb)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * batch_emb.size(0)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total += batch_emb.size(0)

    avg_train_loss = epoch_loss / total
    avg_train_acc = correct / total

    # Validation pass
    classifier.eval()
    with torch.no_grad():
        val_emb = val_embeddings.to(device)
        val_y = val_labels.to(device)
        val_logits = classifier(val_emb)
        val_loss = criterion(val_logits, val_y).item()
        val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()

    scheduler.step(val_loss)

    # Save best checkpoint on raw val loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(classifier.state_dict(), "best_chronos_classifier.pth")

    # Early stopping on smoothed val loss
    val_loss_history.append(val_loss)
    smoothed = np.mean(val_loss_history[-SMOOTHING_WINDOW:])
    if epoch == 0:
        best_smoothed_val_loss = smoothed
        epochs_no_improve = 0
    elif smoothed < best_smoothed_val_loss - MIN_DELTA:
        best_smoothed_val_loss = smoothed
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:03d}/{EPOCH_PARAM} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} (smoothed: {smoothed:.4f}) | Val Acc: {val_acc:.4f} | "
            f"No improve: {epochs_no_improve}/{PATIENCE}"
        )

    if epochs_no_improve >= PATIENCE:
        print(f"\n--- EARLY STOPPING AT EPOCH {epoch+1} ---")
        print(f"No improvement in smoothed val loss for {PATIENCE} consecutive epochs.")
        break


# ── 6. Final evaluation ───────────────────────────────────────────────────────

print("\nLoading best checkpoint for final evaluation...")
classifier.load_state_dict(torch.load("best_chronos_classifier.pth"))
classifier.eval()

import matplotlib.pyplot as plt

with torch.no_grad():
    test_emb = test_embeddings.to(device)
    test_y = test_labels.to(device)
    test_logits = classifier(test_emb)
    test_loss = criterion(test_logits, test_y).item()
    test_preds = test_logits.argmax(dim=1).cpu().numpy()
    test_labels_np = test_labels.numpy()
    test_accuracy = (test_preds == test_labels_np).mean()

print(f"\n--- Final Results ---")
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

# Confusion matrix
conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for true_c, pred_c in zip(test_labels_np, test_preds):
    conf_matrix[true_c, pred_c] += 1

class_totals = conf_matrix.sum(axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    per_class_acc = np.where(class_totals > 0, np.diag(conf_matrix) / class_totals, np.nan)

print("\nConfusion matrix (rows=true, cols=predicted):")
print(conf_matrix)
print("\nPer-class accuracy:")
for c in range(NUM_CLASSES):
    label = class_names[c]
    if class_totals[c] > 0:
        print(f"  Class {c} ({label}): {per_class_acc[c]:.4f}  ({conf_matrix[c,c]}/{class_totals[c]})")
    else:
        print(f"  Class {c} ({label}): no test examples")

# --- F1 score (per-class, macro, weighted) ---
# Per-class: how well each individual bifurcation type is identified.
# Macro: unweighted mean across classes (treats Fold/Hopf/Transcritical/Null
#        equally regardless of how many test examples each has).
# Weighted: mean weighted by class support (dominated by the more frequent
#           classes) - closer to overall accuracy when classes are imbalanced.
labels_present = [c for c in range(NUM_CLASSES) if class_totals[c] > 0]
target_names_present = [class_names[c] for c in labels_present]

per_class_f1 = f1_score(test_labels_np, test_preds, labels=labels_present, average=None, zero_division=0)
per_class_precision = precision_score(test_labels_np, test_preds, labels=labels_present, average=None, zero_division=0)
per_class_recall = recall_score(test_labels_np, test_preds, labels=labels_present, average=None, zero_division=0)
macro_f1 = f1_score(test_labels_np, test_preds, labels=labels_present, average="macro", zero_division=0)
weighted_f1 = f1_score(test_labels_np, test_preds, labels=labels_present, average="weighted", zero_division=0)

print("\nPer-class F1 (precision / recall / f1):")
for c, p, r, f in zip(labels_present, per_class_precision, per_class_recall, per_class_f1):
    print(f"  Class {c} ({class_names[c]}): precision={p:.4f}  recall={r:.4f}  f1={f:.4f}")

print(f"\nMacro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nsklearn classification_report:")
print(classification_report(
    test_labels_np, test_preds,
    labels=labels_present, target_names=target_names_present,
    zero_division=0,
))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

im = axes[0].imshow(conf_matrix, cmap="Blues")
axes[0].set_title("Confusion matrix — Chronos-2 + MLP (test set)")
axes[0].set_xlabel("Predicted class")
axes[0].set_ylabel("True class")
axes[0].set_xticks(range(NUM_CLASSES))
axes[0].set_yticks(range(NUM_CLASSES))
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
thresh = conf_matrix.max() / 2 if conf_matrix.max() > 0 else 0
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        axes[0].text(j, i, str(conf_matrix[i, j]), ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

valid_classes = [c for c in range(NUM_CLASSES) if class_totals[c] > 0]
valid_accs = [per_class_acc[c] for c in valid_classes]
x = np.arange(len(valid_classes))
width = 0.35
axes[1].bar(x - width/2, valid_accs, width, color="tab:blue", label="Accuracy")
axes[1].bar(x + width/2, per_class_f1, width, color="tab:orange", label="F1")
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("Class")
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{c}\n({class_names[c]})" for c in valid_classes])
axes[1].set_ylabel("Score")
axes[1].set_title("Per-class accuracy & F1 — Chronos-2 + MLP (test set)")
axes[1].axhline(test_accuracy, color="gray", linestyle="--", linewidth=1,
                label=f"Overall acc: {test_accuracy:.3f}")
axes[1].axhline(macro_f1, color="darkred", linestyle=":", linewidth=1,
                label=f"Macro F1: {macro_f1:.3f}")
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("confusion_matrix_chronos.png", dpi=120)
plt.close(fig)
print("\nSaved confusion matrix + F1 chart to 'confusion_matrix_chronos.png'")


# ── 7. Time series plot per class ────────────────────────────────────────────
# Shows a handful of example residual time series from each true class in the
# test set, so you can visually inspect what Fold/Hopf/Transcritical/Null
# series actually look like alongside the numeric metrics above.

N_CORRECT_PER_CLASS = 2
N_WRONG_PER_CLASS = 2

fig2, axes2 = plt.subplots(NUM_CLASSES, 1, figsize=(10, 3 * NUM_CLASSES), sharex=True)
if NUM_CLASSES == 1:
    axes2 = [axes2]

for c in range(NUM_CLASSES):
    ax = axes2[c]
    class_idx = np.where(test_labels_np == c)[0]

    if len(class_idx) == 0:
        ax.set_title(f"Class {c} ({class_names[c]}) — no test examples")
        continue

    correct_idx = class_idx[test_preds[class_idx] == c][:N_CORRECT_PER_CLASS]
    wrong_idx = class_idx[test_preds[class_idx] != c][:N_WRONG_PER_CLASS]
    chosen = np.concatenate([correct_idx, wrong_idx])

    for i in chosen:
        series = DL.test[i]              # (ts_len, 1)
        series = np.asarray(series).squeeze(-1)  # -> (ts_len,)
        pred_c = test_preds[i]
        correct = pred_c == c
        ax.plot(
            series,
            alpha=0.8,
            linewidth=1,
            linestyle="-" if correct else "--",
            label=f"idx {i} [{DL.test_source[i]}] ({'correct' if correct else f'pred={class_names[pred_c]}'})",
        )

    ax.set_title(
        f"Class {c} ({class_names[c]}) — {len(class_idx)} test examples, "
        f"showing {len(correct_idx)} correct + {len(wrong_idx)} wrong"
    )
    ax.set_ylabel("Residual value")
    ax.legend(fontsize=7, loc="upper right")

axes2[-1].set_xlabel("Timestep")
fig2.suptitle("Example residual time series by true bifurcation class (test set)", y=1.0)
fig2.tight_layout()
fig2.savefig("timeseries_by_class_chronos.png", dpi=120)
plt.close(fig2)
print("Saved per-class example time series to 'timeseries_by_class_chronos.png'")