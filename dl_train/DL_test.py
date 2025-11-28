# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    DL_test.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tbury <tbury>                              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/11/28 14:15:51 by tbury             #+#    #+#              #
#    Updated: 2025/11/28 15:23:13 by tbury            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
Get performance of DL models on test sets
"""

import os
import zipfile
import sys


import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random

from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dropout, Conv1D, MaxPooling1D, Dense, LSTM  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore

from datetime import datetime

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

ts_len = 1500
model_type = 2
kk = 3

path_to_test_data = f"/Users/tbury/Downloads/ts_{ts_len}/combined/"

# Get target labels for each data sample
df_targets = pd.read_csv(
    path_to_test_data + "labels.csv",
    index_col="sequence_ID",
)

# train/validation/test split denotations
df_groups = pd.read_csv(
    path_to_test_data + "groups.csv",
    index_col="sequence_ID",
)


list_tsid_test = df_groups[df_groups["dataset_ID"] == 3].index
df_targets_test = df_targets[df_targets.index.isin(list_tsid_test)]

(lib_size, ts_len) = (200000, 1500)

# Load in test data
zf = zipfile.ZipFile(path_to_test_data + "output_resids.zip")
text_files = zf.infolist()
sequences = list()

for tsid in list_tsid_test:
    df = pd.read_csv(zf.open("output_resids/resids" + str(tsid) + ".csv"))
    values = df[["Residuals"]].values
    sequences.append(values)

sequences = np.array(sequences).reshape(len(list_tsid_test), ts_len)
print(sequences.shape)

###########################
if model_type == 1:
    pad_left = 225 if ts_len == 500 else 725
    pad_right = 225 if ts_len == 500 else 725

if model_type == 2:
    pad_left = 450 if ts_len == 500 else 1450
    pad_right = 0


# Padding input sequences

for i, tsid in enumerate(list_tsid_test):
    # for i in range(lib_size):
    pad_length = int(pad_left * random.uniform(0, 1))
    for j in range(0, pad_length):
        sequences[i, j] = 0

    pad_length = int(pad_right * random.uniform(0, 1))
    for j in range(ts_len - pad_length, ts_len):
        sequences[i, j] = 0

# normalizing input time series by the average.
for i, tsid in enumerate(list_tsid_test):
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


# apply test labels
test = [
    final_seq[i]
    for i, tsid in enumerate(list_tsid_test)
    if df_groups["dataset_ID"].loc[tsid] == 3
]
test_target = [
    df_targets["class_label"].loc[tsid]
    for i, tsid in enumerate(list_tsid_test)
    if df_groups["dataset_ID"].loc[tsid] == 3
]

inputs_test = np.array(test)
targets_test = np.array(test_target)


######################################
###### Load model and test

path_to_models = f"dl_train/best_models_tf215/len{ts_len}/"
model_name = path_to_models + f"best_model_{kk}_{model_type}_len1500.keras"


# model_name = "dl_train/best_models_tf215/len1500/best_model_1_1_len1500.keras"
model = load_model(model_name)


# Get predictions
preds_prob = model.predict(inputs_test)  # prediction probabilities
preds = np.argmax(preds_prob, axis=1)  # class prediction (max prob)

# Show performance stats
# print(classification_report(targets_test, preds, digits=3))
print("Performance of model 1\n")
print("F1 score: {:.3f}".format(f1_score(targets_test, preds, average="macro")))
print("Accuracy: {:3f}".format(accuracy_score(targets_test, preds)))
print("Confusion matrix: \n")
print(confusion_matrix(targets_test, preds))

# Make confusion matrix plot
class_names = ["Fold", "Hopf", "Branch", "Null"]
disp = ConfusionMatrixDisplay.from_predictions(
    targets_test, preds, display_labels=class_names, cmap=plt.cm.Blues, normalize="true"
)
ax = disp.ax_
ax.images[0].colorbar.remove()
plt.text(x=-1.7, y=-0.1, s="A", fontdict={"size": 14})
plt.show()


# -----------
# Confusion matrix for model type 1 - binary classificaiton problem
# ------------


def full_to_binary(bif_class):
    if bif_class in [0, 1, 2]:
        out = 1
    else:
        out = 0
    return out


targets_test_binary = np.array(list(map(full_to_binary, targets_test)))
preds_binary = np.array(list(map(full_to_binary, preds)))

# Show performance stats
# print(classification_report(targets_test, preds, digits=3))
print("Performance of model 1\n")
print(
    "F1 score: {:.3f}".format(
        f1_score(targets_test_binary, preds_binary, average="macro")
    )
)
print("Accuracy: {:3f}".format(accuracy_score(targets_test_binary, preds_binary)))
print("Confusion matrix: \n")
print(confusion_matrix(targets_test_binary, preds_binary))

# Make confusion matrix plot
class_names = ["Null", "Bifurcation"]
disp = ConfusionMatrixDisplay.from_predictions(
    targets_test_binary,
    preds_binary,
    display_labels=class_names,
    cmap=plt.cm.Blues,
    normalize="true",
)
ax = disp.ax_
ax.images[0].colorbar.remove()
plt.yticks(rotation=90, va="center")
plt.text(x=-0.9, y=-0.35, s="C", fontdict={"size": 14})
plt.ylabel(ylabel="True label", labelpad=15)
plt.show()
