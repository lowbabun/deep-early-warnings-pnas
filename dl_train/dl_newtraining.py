import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import random
# ── Data source options ──────────────────────────────────────────────────────
# "library"    = existing dataset from DL_training.py
# "may"        = May's harvesting model simulated through its fold (Fold-only series)
# "porpoising" = toy F1 porpoising model (Hopf + Null series), see porpoising_ews.py
# "both"       = library + may, "all" = library + may + porpoising, or any
#                "+"-joined combination, e.g. "library+porpoising".
# TRAIN_DATA controls train + validation, TEST_DATA controls the test set.
TRAIN_DATA = "library"   # "library" | "may" | "porpoising" | "both" | "all" | "a+b"
TEST_DATA = "may"        # same options as TRAIN_DATA
N_MAY_SERIES = 2000       # May series simulated in total (split 80/10/10)
N_PORP_SERIES = 2000      # porpoising series in total (half Hopf, half Null; split 80/10/10)
from data_source import get_data
DL = get_data(train_data=TRAIN_DATA, test_data=TEST_DATA,
              n_may=N_MAY_SERIES, n_porp=N_PORP_SERIES)

# --- Reproducibility ---
# Fixes the random seed across torch/numpy/python's random module so that
# repeated runs (e.g. when comparing hyperparameter changes) differ only
# because of the change you made, not because of a different random init,
# dropout mask sequence, or train_loader shuffle order.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# hyperparameter settings
CNN_layers = 1
LSTM_layers = 1
pool_size_param = 2
learning_rate_param = 0.0005
batch_param = 100
dropout_percent = 0.10
filters_param = 50
mem_cells = 50
mem_cells2 = 10
kernel_size_param = 12
epoch_param = 1500
initializer_param = "lecun_normal"

class_names = {0: "Fold", 1: "Hopf", 2: "Transcritical", 3: "Null"}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Training on: {device} ---")

class testClassifier(nn.Module):
    """
    Conv1D + LSTM time-series classifier, structurally matching the Keras
    model in DL_training.py:
        Conv1D(filters=50, kernel_size=12, padding="same") -> Dropout -> MaxPool1D(2)
        -> LSTM(50, return_sequences=True) -> Dropout
        -> LSTM(10)  [final hidden state only]
        -> Dropout -> Dense(4, softmax)

    Replacing the old flat Linear(500)->Linear(4) MLP, which had no notion
    of temporal locality at all (every timestep was treated as an
    independent feature, so it could only memorize fixed-index patterns -
    consistent with the high train / near-random val accuracy seen earlier).

    Input shape: (batch, time, 1)  - same convention as Keras's input_shape=(ts_len, 1)
    PyTorch's nn.Conv1d expects (batch, channels, time) instead, so we
    transpose internally rather than asking the caller to reshape data
    differently than the Keras pipeline already does.
    """
    def __init__(
        self,
        seq_len=500,
        filters=50,
        kernel_size=12,
        pool_size=2,
        lstm1_hidden=50,
        lstm2_hidden=10,
        dropout=0.10,
        output_dim=4,
    ):
        super(testClassifier, self).__init__()

        # padding="same" equivalent: for even kernel_size (e.g. 12), symmetric
        # padding of kernel_size // 2 on both sides actually adds one extra
        # timestep (PyTorch's Conv1d padding is symmetric, unlike Keras's
        # asymmetric "same" padding). We pad manually and trim the trailing
        # extra step in forward() to keep the sequence length exactly
        # unchanged through the conv layer, matching the Keras reference.
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self._trim_output = (kernel_size % 2 == 0)  # True when padding adds 1 extra timestep
        self.conv_dropout = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        self.lstm1 = nn.LSTM(input_size=filters, hidden_size=lstm1_hidden, batch_first=True)
        self.lstm1_dropout = nn.Dropout(p=dropout)

        self.lstm2 = nn.LSTM(input_size=lstm1_hidden, hidden_size=lstm2_hidden, batch_first=True)
        self.final_dropout = nn.Dropout(p=dropout)

        self.classifier = nn.Linear(lstm2_hidden, output_dim)

    def forward(self, x):
        # x: (batch, time, 1) -> Conv1d wants (batch, channels, time)
        x = x.transpose(1, 2)               # (batch, 1, time)
        x = torch.relu(self.conv(x))        # (batch, filters, time [+1 if kernel_size is even])
        if self._trim_output:
            x = x[:, :, :-1]                 # drop the extra trailing timestep to match Keras's "same" length
        x = self.conv_dropout(x)
        x = self.pool(x)                    # (batch, filters, time // pool_size)

        # LSTM wants (batch, time, features)
        x = x.transpose(1, 2)               # (batch, time // pool_size, filters)

        x, _ = self.lstm1(x)                # (batch, time, lstm1_hidden) - full sequence kept,
                                             # matching Keras's return_sequences=True
        x = self.lstm1_dropout(x)

        x, (h_n, _) = self.lstm2(x)         # h_n: (1, batch, lstm2_hidden) - final hidden state only,
        x = h_n.squeeze(0)                  # matching Keras's LSTM(...) without return_sequences

        x = self.final_dropout(x)
        return self.classifier(x)           # raw logits; CrossEntropyLoss applies softmax internally

def lecun_normal_init(module):
    """
    LeCun-normal initialization: weights ~ N(0, 1/fan_in). This was
    previously just a hyperparameter name (initializer_param) that was
    never actually applied anywhere - PyTorch's default inits (Kaiming-
    uniform-ish for Conv1d/Linear, uniform for LSTM) were silently running
    instead. This wires it up for real so the PyTorch model's init matches
    the intent of the Keras reference (lecun_normal).
    """
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        fan_in = module.weight.shape[1] * (module.weight.shape[2] if module.weight.dim() == 3 else 1)
        std = (1.0 / fan_in) ** 0.5
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                fan_in = param.shape[1]
                std = (1.0 / fan_in) ** 0.5
                nn.init.normal_(param, mean=0.0, std=std)
            elif "bias" in name:
                nn.init.zeros_(param)


model = testClassifier(
    seq_len=DL.ts_len,
    filters=filters_param,
    kernel_size=kernel_size_param,
    pool_size=pool_size_param,
    lstm1_hidden=mem_cells,
    lstm2_hidden=mem_cells2,
    dropout=dropout_percent,
).to(device)

if initializer_param == "lecun_normal":
    model.apply(lecun_normal_init)

training_data = torch.tensor(DL.train, dtype=torch.float32).view(len(DL.train), DL.ts_len, 1)
training_labels = torch.tensor(DL.train_target, dtype=torch.long)

test_data = torch.tensor(DL.test, dtype=torch.float32).view(len(DL.test), DL.ts_len, 1)
test_labels = torch.tensor(DL.test_target, dtype=torch.long)

# Validation data, used for early stopping below. Using held-out validation
# loss (rather than training loss) is what makes early stopping meaningful -
# training loss almost always keeps decreasing, so checking it tells you
# nothing about overfitting. DL.validation / DL.validation_target already
# exist (built by DL_training.py at import time) but weren't being used here.
val_data = torch.tensor(DL.validation, dtype=torch.float32).view(len(DL.validation), DL.ts_len, 1)
val_labels = torch.tensor(DL.validation_target, dtype=torch.long)

train_dataset = TensorDataset(training_data, training_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_param, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_param, weight_decay=1e-5)

# Class-weighted loss: class 0 ("no bifurcation / stable") was completely
# missed by this exact Conv1D+LSTM architecture in earlier runs (0/25
# correct - see DL_ChronosClassifier.py comments). Weighting inversely by
# class frequency in the training set pushes the optimizer to treat it
# equally with the other classes rather than ignoring its harder signal.
class_counts = np.bincount(DL.train_target, minlength=4)
# Classes absent from the training set get weight 0 (not a huge weight, which
# would push the model toward classes it never sees) and the rest are
# normalised to average 1 over the classes actually present.
_inv = np.where(class_counts > 0, 1.0 / np.maximum(class_counts, 1), 0.0)
_inv = _inv / _inv.sum() * (class_counts > 0).sum()
class_weights = torch.tensor(_inv, dtype=torch.float32)
class_weights = class_weights.to(device)
print(f"Training class counts: {dict(zip(class_names.values(), class_counts))}")
print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Decay LR when validation loss plateaus, rather than holding it fixed for
# the entire 1500-epoch budget - lets training settle tighter near the end
# instead of oscillating around a local minimum at a constant step size.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

best_val_loss = float('inf')
patience = 40  # was 10 - too tight given how noisy val loss is with only
                # ~100 validation examples; allow more epochs of fluctuation
                # before concluding the model has actually stopped improving
min_delta = 1e-4  # ignore improvements smaller than this - prevents the
                   # counter from resetting on negligible noise-level dips
epochs_no_improve = 0
val_loss_history = []
smoothing_window = 5  # epochs to average over for the stopping decision

print("--- Starting PyTorch Training Loop ---")

for epoch in range(epoch_param):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        # Clip gradients before stepping - LSTMs are prone to exploding
        # gradients over long sequences (250 timesteps post-pooling here),
        # and this was already the practice in DL_ChronosClassifier.py's
        # training loop but missing here.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
        correct_predictions += (outputs.argmax(dim=1) == batch_y).sum().item()
        total_samples += batch_X.size(0)

    #average loss and accuracy
    avg_loss = epoch_loss / total_samples
    avg_acc = correct_predictions / total_samples

    # --- Validation pass + early stopping (once per epoch, not per batch) ---
    # The previous version checked epoch_loss (a running, ever-growing
    # within-epoch sum) against a previous epoch's final sum, INSIDE the
    # batch loop - so it compared partial sums to full sums on every single
    # batch, and incremented/reset epochs_no_improve dozens of times per
    # epoch. That made it trigger almost immediately, before the model had
    # any real chance to train, and it was checking training loss rather
    # than a held-out validation signal anyway.
    model.eval()
    with torch.no_grad():
        val_X = val_data.to(device)
        val_y = val_labels.to(device)
        val_outputs = model(val_X)
        val_loss = criterion(val_outputs, val_y).item()
        val_acc = (val_outputs.argmax(dim=1) == val_y).float().mean().item()

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        # Save the best model state (based on raw validation loss, not training
        # loss) - this checkpoint-saving stays on the raw per-epoch loss so we
        # don't miss genuinely good epochs just because the smoothed signal
        # used for the *stopping decision* hasn't caught up yet.
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    # Smoothed validation loss for the early-stopping decision itself: a
    # single noisy epoch (common with a small ~100-example validation set)
    # can look like "no improvement" even when the model is still genuinely
    # getting better on average. Averaging over the last `smoothing_window`
    # epochs filters that out before deciding whether to increment the
    # no-improve counter.
    val_loss_history.append(val_loss)
    smoothed_val_loss = np.mean(val_loss_history[-smoothing_window:])

    if epoch == 0:
        best_smoothed_val_loss = smoothed_val_loss
        epochs_no_improve = 0
    elif smoothed_val_loss < best_smoothed_val_loss - min_delta:
        best_smoothed_val_loss = smoothed_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    #Print every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epoch_param} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} (smoothed: {smoothed_val_loss:.4f}) | Val Acc: {val_acc:.4f} "
            f"| LR: {current_lr:.2e} | No improve: {epochs_no_improve}/{patience}"
        )

    if epochs_no_improve >= patience:
        print(f"\n--- EARLY STOPPING TRIGGERED AT EPOCH {epoch+1} ---")
        print(f"No improvement in validation loss for {patience} consecutive epochs.")
        break

#evaluate on test set
# Load back the best checkpoint (lowest validation loss), not whatever the
# model happened to be at the final training epoch - those can differ since
# training continues for `patience` more epochs after the best point before
# stopping triggers.
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    # Send test data to GPU
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    
    test_outputs = model(test_data)
    test_loss = criterion(test_outputs, test_labels)
    test_accuracy = (test_outputs.argmax(dim=1) == test_labels).float().mean().item()
    
    print("\n--- Final Results ---")
    print(f"Test Loss: {test_loss.item():.4f} | Test Accuracy: {test_accuracy:.4f}")

    # --- Confusion matrix + per-class accuracy ---
    test_preds = test_outputs.argmax(dim=1).cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()
    num_classes = test_outputs.shape[1]  # 4, matches testClassifier(output_dim=4)

    # Build the confusion matrix manually (rows = true class, cols = predicted
    # class) so we don't need an extra sklearn dependency just for this plot.
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_c, pred_c in zip(test_labels_np, test_preds):
        conf_matrix[true_c, pred_c] += 1

    # Per-class accuracy = (correct predictions for class c) / (total true
    # instances of class c). Classes with zero test examples get NaN rather
    # than a divide-by-zero, and are skipped in the bar chart.
    class_totals = conf_matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.where(class_totals > 0, np.diag(conf_matrix) / class_totals, np.nan)

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(conf_matrix)
    print("\nPer-class accuracy:")
    for c in range(num_classes):
        if class_totals[c] > 0:
            print(f"  Class {c}: {per_class_acc[c]:.4f}  ({conf_matrix[c, c]}/{class_totals[c]})")
        else:
            print(f"  Class {c}: no test examples")

    # --- F1 score (per-class, macro, weighted) ---
    # Per-class: how well each individual bifurcation type is identified.
    # Macro: unweighted mean across classes (treats Fold/Hopf/Transcritical/
    #        Null equally regardless of test-set support).
    # Weighted: mean weighted by class support - closer to overall accuracy
    #           when classes are imbalanced.
    labels_present = [c for c in range(num_classes) if class_totals[c] > 0]
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

    # Panel 1: confusion matrix heatmap
    im = axes[0].imshow(conf_matrix, cmap="Blues")
    axes[0].set_title("Confusion matrix (test set)")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_xticks(range(num_classes))
    axes[0].set_yticks(range(num_classes))
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    # annotate each cell with its count
    thresh = conf_matrix.max() / 2 if conf_matrix.max() > 0 else 0
    for i in range(num_classes):
        for j in range(num_classes):
            axes[0].text(
                j, i, str(conf_matrix[i, j]),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
            )

    # Panel 2: per-class accuracy + F1 bar chart
    valid_classes = [c for c in range(num_classes) if class_totals[c] > 0]
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
    axes[1].set_title("Per-class accuracy & F1 (test set)")
    axes[1].axhline(test_accuracy, color="gray", linestyle="--", linewidth=1, label=f"Overall acc: {test_accuracy:.3f}")
    axes[1].axhline(macro_f1, color="darkred", linestyle=":", linewidth=1, label=f"Macro F1: {macro_f1:.3f}")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_path = "confusion_matrix.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved confusion matrix plot to '{out_path}'")


# ── Training-set evaluation + per-class example time series ─────────────────
# Same idea as the test-set plot above (2 correct + 2 misclassified examples
# per class, solid vs dashed lines), but run on the TRAINING set instead.
# Useful for sanity-checking fit: if training-set examples are still being
# misclassified in ways that look like the test-set failure modes, the model
# is underfitting rather than overfitting; if training predictions look
# near-perfect while test performance is much worse, that points to
# overfitting/generalization gap instead.
model.eval()
train_eval_loader = DataLoader(
    TensorDataset(training_data, training_labels), batch_size=batch_param, shuffle=False
)
train_preds_list = []
with torch.no_grad():
    for batch_X, _ in train_eval_loader:
        batch_X = batch_X.to(device)
        batch_preds = model(batch_X).argmax(dim=1).cpu().numpy()
        train_preds_list.append(batch_preds)
train_preds = np.concatenate(train_preds_list)
train_labels_np = training_labels.numpy()
train_accuracy = (train_preds == train_labels_np).mean()

print("\n--- Training Set Evaluation (best checkpoint, eval mode) ---")
print(f"Train Accuracy: {train_accuracy:.4f}")

N_CORRECT_PER_CLASS = 2
N_WRONG_PER_CLASS = 2

fig3, axes3 = plt.subplots(num_classes, 1, figsize=(10, 3 * num_classes), sharex=True)
if num_classes == 1:
    axes3 = [axes3]

for c in range(num_classes):
    ax = axes3[c]
    class_idx = np.where(train_labels_np == c)[0]

    if len(class_idx) == 0:
        ax.set_title(f"Class {c} ({class_names[c]}) — no training examples")
        continue

    correct_idx = class_idx[train_preds[class_idx] == c][:N_CORRECT_PER_CLASS]
    wrong_idx = class_idx[train_preds[class_idx] != c][:N_WRONG_PER_CLASS]
    chosen = np.concatenate([correct_idx, wrong_idx])

    for i in chosen:
        series = DL.train[i]                       # (ts_len, 1)
        series = np.asarray(series).squeeze(-1)     # -> (ts_len,)
        pred_c = train_preds[i]
        correct = pred_c == c
        ax.plot(
            series,
            alpha=0.8,
            linewidth=1,
            linestyle="-" if correct else "--",
            label=f"idx {i} [{DL.train_source[i]}] ({'correct' if correct else f'pred={class_names[pred_c]}'})",
        )

    ax.set_title(
        f"Class {c} ({class_names[c]}) — {len(class_idx)} training examples, "
        f"showing {len(correct_idx)} correct + {len(wrong_idx)} wrong"
    )
    ax.set_ylabel("Residual value")
    ax.legend(fontsize=7, loc="upper right")

axes3[-1].set_xlabel("Timestep")
fig3.suptitle("Example residual time series by true bifurcation class (training set)", y=1.0)
fig3.tight_layout()
fig3.savefig("timeseries_by_class_train.png", dpi=120)
plt.close(fig3)
print("Saved per-class example time series (training set) to 'timeseries_by_class_train.png'")