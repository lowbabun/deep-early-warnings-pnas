"""
DL_Choronotraining.py

Fine-tunes Chronos-2 with LoRA on your real residual time series data
(loaded via DL_training.py), replacing the original np.random.rand placeholders.

Data flow
---------
DL_training.py produces:
    DL.train        : (N_train, 500, 1)  float64  - residual series, padded+normalized
    DL.validation   : (N_val,   500, 1)  float64
    DL.train_target : (N_train,)         int      - class labels 0-3 (NOT used here)
    DL.validation_target: same

Chronos-2 fit() needs:
    list of {"target": np.ndarray shape (500,)}  - flat 1D per series
    No separate future-values array needed; fit() carves out the
    prediction window internally from the context series.

So the only transformation needed is: squeeze the trailing (,1) channel dim.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List

# --- Import your real data pipeline ---
# This runs DL_training.py's module-level code (loads zip/npy, pads,
# normalizes, stratified-samples, splits into train/val/test arrays).
# Make sure DL_training.py is in the same directory or on PYTHONPATH.
import DL_training as DL

from chronos import BaseChronosPipeline, Chronos2Pipeline


def build_chronos2_inputs(series_array: np.ndarray) -> List[Dict[str, np.ndarray]]:
    """
    Convert a (N, ts_len, 1) or (N, ts_len) numpy array into the
    list-of-dict format Chronos2Pipeline.fit() expects.

    fit() only needs the historical target per series — it learns to
    predict the next `prediction_length` steps internally. Class labels
    are not passed in here; Chronos-2 is fine-tuned as a forecaster,
    not a classifier.

    Args:
        series_array: (N, ts_len[, 1]) array from DL_training.py

    Returns:
        list of {"target": (ts_len,) float32 array}, one per series
    """
    # squeeze trailing channel dim if present: (N, 500, 1) -> (N, 500)
    if series_array.ndim == 3 and series_array.shape[2] == 1:
        series_array = series_array.squeeze(-1)

    inputs = []
    for i in range(len(series_array)):
        inputs.append({"target": np.asarray(series_array[i], dtype=np.float32)})
    return inputs


def train_chronos2(
    train_contexts: np.ndarray,
    val_contexts: np.ndarray,
    model_name: str = "amazon/chronos-2",
    prediction_length: int = 50,
    num_steps: int = 1000,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    finetune_mode: str = "lora",
    lora_config: Optional[Dict[str, Any]] = None,
    logging_steps: int = 100,
    save_dir: str = "chronos2_lora_adapter",
    device_map: Optional[str] = None,
):
    """
    Fine-tune Chronos-2 with LoRA on real residual time series.

    fit() is step-based (not epoch-based), evaluates every `logging_steps`
    steps when validation_inputs is given, and automatically reloads the
    best checkpoint at the end — no manual early stopping needed.

    Args:
        train_contexts : (N_train, ts_len[, 1]) array  - DL.train
        val_contexts   : (N_val,   ts_len[, 1]) array  - DL.validation
        prediction_length: how many steps ahead Chronos-2 learns to forecast.
                           Set to ~10% of ts_len (500 steps -> 50 step horizon).
    """
    if device_map is None:
        device_map = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Chronos-2 pipeline from '{model_name}' on device='{device_map}'...")
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device_map,
    )

    print(f"Building inputs from {len(train_contexts)} train / {len(val_contexts)} val series...")
    train_inputs = build_chronos2_inputs(train_contexts)
    val_inputs   = build_chronos2_inputs(val_contexts)
    print(f"  Series length (context): {len(train_inputs[0]['target'])} steps")
    print(f"  Prediction horizon:      {prediction_length} steps")

    if lora_config is None:
        lora_config = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": [
                "self_attention.q",
                "self_attention.v",
                "self_attention.k",
                "self_attention.o",
                "output_patch_embedding.output_layer",
            ],
            "lora_dropout": 0.05,
            "bias": "none",
        }

    print(f"\nStarting {finetune_mode.upper()} fine-tuning for {num_steps} steps...")
    finetuned_pipeline = pipeline.fit(
        inputs=train_inputs,
        validation_inputs=val_inputs,
        prediction_length=prediction_length,
        finetune_mode=finetune_mode,
        lora_config=lora_config if finetune_mode == "lora" else None,
        learning_rate=learning_rate,
        num_steps=num_steps,
        batch_size=batch_size,
        logging_steps=logging_steps,
    )

    print(f"\nTraining complete. Saving adapter to '{save_dir}'...")
    finetuned_pipeline.save_pretrained(save_dir)
    print("Saved.")

    return finetuned_pipeline


if __name__ == "__main__":

    print(f"Train series:      {DL.train.shape}")       # (N, 500, 1)
    print(f"Validation series: {DL.validation.shape}")  # (N, 500, 1)
    print(f"Series length:     {DL.ts_len}")            # 500
    print(f"Train class dist:  {np.bincount(DL.train_target)}")

    # prediction_length: how many steps at the end of each 500-step series
    # Chronos-2 learns to forecast. 50 = last 10% of the series, which
    # covers the region closest to (or containing) the bifurcation point —
    # the most informative part for early-warning signal research.
    # Adjust if you want a longer/shorter forecast horizon.
    PREDICTION_LENGTH = 50

    finetuned_pipeline = train_chronos2(
        train_contexts=DL.train,
        val_contexts=DL.validation,
        model_name="amazon/chronos-2",
        prediction_length=PREDICTION_LENGTH,
        num_steps=1000,
        learning_rate=1e-4,
        batch_size=32,
        finetune_mode="lora",
        save_dir="chronos2_lora_adapter",
    )