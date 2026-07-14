import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import DL_training as DL

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

model = testClassifier(
    seq_len=DL.ts_len,
    filters=filters_param,
    kernel_size=kernel_size_param,
    pool_size=pool_size_param,
    lstm1_hidden=mem_cells,
    lstm2_hidden=mem_cells2,
    dropout=dropout_percent,
).to(device)

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
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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
        print(
            f"Epoch {epoch+1}/{epoch_param} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} (smoothed: {smoothed_val_loss:.4f}) | Val Acc: {val_acc:.4f} "
            f"| No improve: {epochs_no_improve}/{patience}"
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

    # Panel 2: per-class accuracy bar chart
    valid_classes = [c for c in range(num_classes) if class_totals[c] > 0]
    valid_accs = [per_class_acc[c] for c in valid_classes]
    axes[1].bar([str(c) for c in valid_classes], valid_accs, color="tab:blue")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Per-class accuracy (test set)")
    axes[1].axhline(test_accuracy, color="gray", linestyle="--", linewidth=1, label=f"Overall: {test_accuracy:.3f}")
    axes[1].legend()

    fig.tight_layout()
    out_path = "confusion_matrix.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved confusion matrix plot to '{out_path}'")