"""
This script:
  1. Loads the real Chronos2Pipeline.
  2. Converts your numpy arrays into the dict/array input format the pipeline
     actually accepts (no custom torch.utils.data.Dataset needed).
  3. Calls pipeline.fit(..., finetune_mode="lora") with sane LoRA defaults.
  4. Saves and reloads the fine-tuned pipeline (adapter weights only).


"""

import torch
import numpy as np
from chronos import BaseChronosPipeline, Chronos2Pipeline



def build_chronos2_inputs(raw_contexts: np.ndarray, raw_targets: np.ndarray):
    """
    Convert plain (N, context_len) / (N, horizon) numpy arrays into the list-of-dict
    format Chronos2Pipeline.fit() / .predict_quantiles() accept for univariate series.

    Chronos-2's `fit` only needs the historical `target` per series; it learns to
    predict the next `prediction_length` steps from context during training, so we
    only pass `target` here (NOT future_values — fit handles the train/predict
    split internally from context length and prediction_length).
    """
    
    inputs = []
    for i in range(len(raw_contexts)):
        inputs.append({"target": np.asarray(raw_contexts[i], dtype=np.float32)})
    return inputs


def train_chronos2(
    train_contexts,
    train_targets,
    val_contexts,
    val_targets,
    model_name: str = "amazon/chronos-2",
    prediction_length: int = 10,
    num_steps: int = 1000,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    finetune_mode: str = "lora",
    lora_config: dict | None = None,
    logging_steps: int = 100,
    save_dir: str = "chronos2_lora_adapter",
    device_map: str | None = None,
):
    """
    Fine-tune Chronos-2 with LoRA using the pipeline's built-in `fit` method.

    Note on validation: `fit()` evaluates every 100 steps when `validation_inputs`
    is given, and automatically reloads the best (lowest eval_loss) checkpoint at
    the end — that replaces the manual early-stopping/patience logic from the
    original script. `num_steps` is the stopping criterion you control directly
    (this is step-based, not epoch-based, since `fit` batches series randomly
    rather than iterating a fixed-size DataLoader for a fixed number of epochs).
    """
    if device_map is None:
        device_map = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Chronos-2 pipeline from '{model_name}' on device='{device_map}'...")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=device_map,
    )

    train_inputs = build_chronos2_inputs(train_contexts, train_targets)
    val_inputs = build_chronos2_inputs(val_contexts, val_targets)

    # Default LoRA config matches the package's own LoRA fine-tuning defaults
    # (r=8, alpha=16, targeting the attention + output-projection layers) unless
    # the caller overrides it.
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

    # Simulated data — replace with your DL_training.py data pipeline.
    # Shapes: (num_series, context_length) and (num_series, prediction_length).
    # NOTE: fit() only consumes the historical `target` per series (see
    # build_chronos2_inputs above) — the held-out future window is generated
    # internally from context_length/prediction_length during training, so
    # raw_y_train/raw_y_val aren't fed in directly. Keep them around if you need
    # them for your own post-hoc evaluation against finetuned_pipeline.predict_*.
    raw_X_train = np.random.rand(100, 50)
    raw_y_train = np.random.rand(100, 10)
    raw_X_val = np.random.rand(20, 50)
    raw_y_val = np.random.rand(20, 10)

    PREDICTION_LENGTH = raw_y_train.shape[1]  # 10

    finetuned_pipeline = train_chronos2(
        train_contexts=raw_X_train,
        train_targets=raw_y_train,
        val_contexts=raw_X_val,
        val_targets=raw_y_val,
        model_name="amazon/chronos-2",
        prediction_length=PREDICTION_LENGTH,
        num_steps=1000,
        learning_rate=1e-4,   # 1e-5–1e-4 recommended for LoRA fine-tuning
        batch_size=32,
        finetune_mode="lora",
    )

    # Example: use the fine-tuned pipeline for inference afterward
    # quantiles, mean = finetuned_pipeline.predict_quantiles(
    #     raw_X_val, prediction_length=PREDICTION_LENGTH, quantile_levels=[0.1, 0.5, 0.9]
    # )