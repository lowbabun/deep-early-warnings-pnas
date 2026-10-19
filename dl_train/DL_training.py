# Python script to train a neural network using Keras library.

import os
import zipfile

from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import pandas
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dropout, Conv1D, MaxPooling1D, Dense, LSTM  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore

from datetime import datetime

random.seed()

# model_type
# 0: no padding applied to time series (raw residual series used as-is)
# 1: both left and right sides of time series are padded
# 2: only left side of time series is padded

# Get command line parameters
model_type = 0
kk = 1  # index for NN
print("Train a classifier of type {} with index {}".format(model_type, kk))

# model_type = 1
# kk = 1

# Set size of training library and length of time series
(lib_size, ts_len) = (5000, 500)  # Or (500000, 500) for the 500-classifier

# keeps track of training metrics
f1_name = "training_results_{}_{}.txt".format(kk, model_type)
f2_name = "training_results_{}_{}.csv".format(kk, model_type)

f_results = open(f1_name, "w")
f_results2 = open(f2_name, "w")


if model_type == 0:
    # No padding: use the full-length raw series unchanged. pad_left/pad_right
    # stay at 0, which makes the padding loop below a no-op (pad_length =
    # int(0 * random.uniform(0,1)) == 0 for every series) without needing a
    # separate code path.
    pad_left = 0
    pad_right = 0

if model_type == 1:
    pad_left = 225 if ts_len == 500 else 725
    pad_right = 225 if ts_len == 500 else 725

if model_type == 2:
    pad_left = 450 if ts_len == 500 else 1450
    pad_right = 0

# train/validation/test split denotations - load this FIRST so we can pick
# which sequence_IDs to actually load from the zip, rather than assuming the
# first `lib_size` IDs are a representative sample. groups.csv is NOT
# shuffled: dataset_ID appears in long contiguous runs (e.g. the first ~2260
# sequence_IDs are ALL dataset_ID==1/train), and within each dataset_ID the
# class_label is also grouped in long runs. A naive
# tsid_vals = np.arange(1, lib_size+1) slice can therefore land entirely
# inside one dataset_ID/class block and produce an empty or single-class
# validation/test split, even though the full file is perfectly balanced
# (each dataset_ID has exactly 1/4 of its rows in each of the 4 classes).
df_groups_full = pandas.read_csv(
    r"C:\Users\19255\research\deep-early-warnings-pnas\training_data\output_full\ts_500\combined\groups.csv",
    index_col="sequence_ID",
)
df_targets_full = pandas.read_csv(
    r"C:\Users\19255\research\deep-early-warnings-pnas\training_data\output_full\ts_500\combined\labels.csv",
    index_col="sequence_ID",
)

# How many sequences to load per split. lib_size is split roughly
# proportionally to the full dataset's split ratios (~95% train / 4% val /
# 1% test), then each split is sampled to be class-balanced across the 4
# labels. Adjust these target counts directly if you want a different split
# size independent of the proportional split.
_split_fracs = {1: 0.80, 2: 0.10, 3: 0.10}  # train/val/test - more balanced
# than the raw 95/4/1 dataset ratio, since for a lib_size-limited subset you
# usually want enough val/test examples to evaluate on, not a proportional
# mirror of the full 500k-row dataset.

selected_ids_by_split = {}
rng = np.random.default_rng(seed=42)  # fixed seed -> reproducible split across runs

for dataset_id, frac in _split_fracs.items():
    target_n = max(1, int(lib_size * frac))
    ids_in_split = df_groups_full.index[df_groups_full["dataset_ID"] == dataset_id]
    labels_in_split = df_targets_full.loc[ids_in_split, "class_label"]

    # Stratified sample: aim for an equal number of each class within target_n
    classes = sorted(labels_in_split.unique())
    per_class_n = max(1, target_n // len(classes))
    picked = []
    for c in classes:
        ids_c = labels_in_split.index[labels_in_split == c].to_numpy()
        n = min(per_class_n, len(ids_c))
        picked.extend(rng.choice(ids_c, size=n, replace=False))
    selected_ids_by_split[dataset_id] = sorted(picked)

tsid_vals = np.array(
    selected_ids_by_split[1] + selected_ids_by_split[2] + selected_ids_by_split[3]
)
print(
    f"Selected {len(selected_ids_by_split[1])} train / "
    f"{len(selected_ids_by_split[2])} val / "
    f"{len(selected_ids_by_split[3])} test sequence IDs "
    f"(stratified by class, out of requested lib_size={lib_size})"
)

# get zipfile of time series
print("Load in zip file containing training data")
zf = zipfile.ZipFile(r"C:\Users\19255\research\deep-early-warnings-pnas\training_data\output_full\ts_500\combined\output_resids.zip")
text_files = zf.infolist()
sequences = list()


print("Extract time series from zip file")
for tsid in tqdm(tsid_vals, desc="Loading CSVs"):
    df = pandas.read_csv(zf.open("output_resids/resids" + str(int(tsid)) + ".csv"))
    values = df[["Residuals"]].values
    sequences.append(values)

sequences = np.array(sequences)

# Get target labels for each data sample
df_targets = df_targets_full

# train/validation/test split denotations
df_groups = df_groups_full

# Padding input sequences
# With model_type=0, pad_left=pad_right=0 above, so pad_length is always 0
# here and every series passes through unmodified (no zeroed-out region).
for i, tsid in enumerate(tsid_vals):
    # for i in range(lib_size):
    pad_length = int(pad_left * random.uniform(0, 1))
    for j in range(0, pad_length):
        sequences[i, j] = 0

    pad_length = int(pad_right * random.uniform(0, 1))
    for j in range(ts_len - pad_length, ts_len):
        sequences[i, j] = 0

# normalizing input time series by the average.
for i, tsid in enumerate(tsid_vals):
    # for i in range(lib_size):
    values_avg = 0.0
    count_avg = 0
    for j in range(0, ts_len):
        if sequences[i, j] != 0:
            values_avg = values_avg + abs(sequences[i, j])
            count_avg = count_avg + 1
    if count_avg != 0:
        values_avg = values_avg / count_avg
        for j in range(0, ts_len):
            if sequences[i, j] != 0:
                sequences[i, j] = sequences[i, j] / values_avg

final_seq = sequences

# apply train/test/validation labels
train = [
    final_seq[i]
    for i, tsid in enumerate(tsid_vals)
    if df_groups["dataset_ID"].loc[tsid] == 1
]
validation = [
    final_seq[i]
    for i, tsid in enumerate(tsid_vals)
    if df_groups["dataset_ID"].loc[tsid] == 2
]
test = [
    final_seq[i]
    for i, tsid in enumerate(tsid_vals)
    if df_groups["dataset_ID"].loc[tsid] == 3
]


train_target = [
    df_targets["class_label"].loc[tsid]
    for i, tsid in enumerate(tsid_vals)
    if df_groups["dataset_ID"].loc[tsid] == 1
]
validation_target = [
    df_targets["class_label"].loc[tsid]
    for i, tsid in enumerate(tsid_vals)
    if df_groups["dataset_ID"].loc[tsid] == 2
]
test_target = [
    df_targets["class_label"].loc[tsid]
    for i, tsid in enumerate(tsid_vals)
    if df_groups["dataset_ID"].loc[tsid] == 3
]


train = np.array(train)
validation = np.array(validation)
test = np.array(test)

train_target = np.array(train_target)
validation_target = np.array(validation_target)
test_target = np.array(test_target)


# hyperparameter settings
CNN_layers = 1
LSTM_layers = 1
pool_size_param = 2
learning_rate_param = 0.0005
batch_param = 500
dropout_percent = 0.10
filters_param = 50
mem_cells = 50
mem_cells2 = 10
kernel_size_param = 12
epoch_param = 500
initializer_param = "lecun_normal"

if __name__ == "__main__":
    model = Sequential()

    # add layers
    if CNN_layers == 1:
        model.add(
            Conv1D(
                filters=filters_param,
                kernel_size=kernel_size_param,
                activation="relu",
                padding="same",
                input_shape=(ts_len, 1),
                kernel_initializer=initializer_param,
            )
        )
    elif CNN_layers == 2:
        model.add(
            Conv1D(
                filters=filters_param,
                kernel_size=kernel_size_param,
                activation="relu",
                padding="same",
                input_shape=(ts_len, 1),
            )
        )
        model.add(
            Conv1D(
                filters=2 * filters_param,
                kernel_size=kernel_size_param,
                activation="relu",
                padding="same",
            )
        )

    model.add(Dropout(dropout_percent))
    model.add(MaxPooling1D(pool_size=pool_size_param))

    if LSTM_layers == 1:
        model.add(
            LSTM(mem_cells, return_sequences=True, kernel_initializer=initializer_param)
        )
        model.add(Dropout(dropout_percent))
    elif LSTM_layers == 2:
        model.add(LSTM(mem_cells, return_sequences=True))
        model.add(LSTM(mem_cells, return_sequences=True))
        model.add(Dropout(dropout_percent))

    model.add(LSTM(mem_cells2, kernel_initializer=initializer_param))
    model.add(Dropout(dropout_percent))
    model.add(Dense(4, activation="softmax", kernel_initializer=initializer_param))

    # name for output pickle file containing model info
    model_name = "best_model_{}_{}_length{}.pkl".format(kk, model_type, ts_len)

    # Set up optimiser
    adam = Adam(learning_rate=learning_rate_param)
    chk = ModelCheckpoint(
        model_name, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=adam,
        metrics=["accuracy", "sparse_categorical_accuracy"],
    )

    # Train model
    history = model.fit(
        train,
        train_target,
        epochs=epoch_param,
        batch_size=batch_param,
        callbacks=[chk],
        validation_data=(validation, validation_target),
    )


    model = load_model(model_name)

    # generate test metrics
    from sklearn.metrics import accuracy_score

    
    test_probs = model.predict(test)
    test_preds = np.argmax(test_probs, axis=1)
    accuracy_score(test_target, test_preds)

    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import classification_report

    print(classification_report(test_target, test_preds, digits=3))
    print(history.history["accuracy"])
    print(history.history["val_accuracy"])
    print(history.history["loss"])
    print(history.history["val_loss"])
    print("F1 score:", f1_score(test_target, test_preds, average="macro"))
    print("Precision: ", precision_score(test_target, test_preds, average="macro"))
    print("Recall: ", recall_score(test_target, test_preds, average="macro"))

    conf_matrix = confusion_matrix(test_target, test_preds)
    print("Confusion matrix: \n", conf_matrix)

    # --- Confusion matrix + per-class accuracy plot ---
    num_classes = conf_matrix.shape[0]
    class_totals = conf_matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.where(class_totals > 0, np.diag(conf_matrix) / class_totals, np.nan)
    overall_accuracy = accuracy_score(test_target, test_preds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: confusion matrix heatmap
    im = axes[0].imshow(conf_matrix, cmap="Blues")
    axes[0].set_title("Confusion matrix (test set)")
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_xticks(range(num_classes))
    axes[0].set_yticks(range(num_classes))
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
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
    axes[1].axhline(overall_accuracy, color="gray", linestyle="--", linewidth=1, label=f"Overall: {overall_accuracy:.3f}")
    axes[1].legend()

    fig.tight_layout()
    conf_matrix_path = "confusion_matrix_keras.png"
    fig.savefig(conf_matrix_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved confusion matrix plot to '{conf_matrix_path}'")

    f_results.write(
        "Simulation %d macro f1: %f. macro avg precision %f.  macro avg recall %f. Kernel size %d.  Filters %d.  Batch size %d.  Epochs %d.  seq_len %d.  Set size %d.  Mem_cells %d. Mem_cells2 %d. dropout %f. CNN layers %d, LSTM layers %d, pool_size_param %d, learning_rate_param %f.  initializer %s. pad_left %d.  pad_right %d \r\n"
        % (
            kk,
            f1_score(test_target, test_preds, average="macro"),
            precision_score(test_target, test_preds, average="macro"),
            recall_score(test_target, test_preds, average="macro"),
            kernel_size_param,
            filters_param,
            batch_param,
            epoch_param,
            ts_len,
            lib_size,
            mem_cells,
            mem_cells2,
            dropout_percent,
            CNN_layers,
            LSTM_layers,
            pool_size_param,
            learning_rate_param,
            initializer_param,
            pad_left,
            pad_right,
        )
    )
    f_results.flush()

    f_results2.write(
        "%d, %f, %f, %f, %d, %d, %d, %d, %d, %d, %d, %d, %f, %d, %d, %d, %f, %s, %d, %d \r\n"
        % (
            kk,
            f1_score(test_target, test_preds, average="macro"),
            precision_score(test_target, test_preds, average="macro"),
            recall_score(test_target, test_preds, average="macro"),
            kernel_size_param,
            filters_param,
            batch_param,
            epoch_param,
            ts_len,
            lib_size,
            mem_cells,
            mem_cells2,
            dropout_percent,
            CNN_layers,
            LSTM_layers,
            pool_size_param,
            learning_rate_param,
            initializer_param,
            pad_left,
            pad_right,
        )
    )
    f_results2.flush()

    f_results.close()
    f_results2.close()