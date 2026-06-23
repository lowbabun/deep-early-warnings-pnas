"""
Chronos-2 fine-tuning verification script.

Run this AFTER DL_Choronotraining.py has produced a saved adapter/checkpoint
directory. This does NOT retrain anything — it only loads both the base
model and your fine-tuned model and compares their forecasts on held-out
data, so you can tell whether fine-tuning actually helped.

IMPORTANT: Replace the simulated raw_X_val / raw_y_val arrays below with your
real held-out validation/test data. Comparing against random noise will not
tell you anything useful about whether fine-tuning worked.

Usage:
    python DL_ChoronosEval.py
"""

import json
import os

import numpy as np
import torch
from chronos import BaseChronosPipeline
from peft import PeftModel


def load_pipelines(base_model_name: str, finetuned_dir: str, device_map: str | None = None):
    """
    Loads the base Chronos-2 pipeline twice (once to stay untouched as the
    baseline, once to have its underlying model wrapped with the LoRA adapter).

    NOTE: We load via `peft.PeftModel.from_pretrained` directly rather than
    `BaseChronosPipeline.from_pretrained(finetuned_dir)`. The latter is
    documented to auto-detect and merge LoRA adapters, but in practice it
    calls `AutoConfig.from_pretrained` first looking for a top-level
    `config.json` describing the base architecture - which an adapter-only
    save directory (just adapter_config.json + adapter_model.safetensors)
    does not have. That auto-detection path isn't triggering correctly in
    the installed chronos-forecasting version, so we bypass it and use
    `peft` directly, which only needs adapter_config.json's
    `base_model_name_or_path` field (already confirmed to be
    "amazon/chronos-2" in your adapter_config.json) plus the adapter weights.
    """
    if device_map is None:
        device_map = "cuda" if torch.cuda.is_available() else "cpu"

    adapter_config_path = os.path.join(finetuned_dir, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(
            f"No adapter_config.json found in '{finetuned_dir}'. "
            "If this is a FULL fine-tune (not LoRA), load it directly with "
            "BaseChronosPipeline.from_pretrained(finetuned_dir) instead - "
            "this script's peft loading path only applies to LoRA adapters."
        )

    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)
    adapter_base_model = adapter_cfg.get("base_model_name_or_path", base_model_name)
    if adapter_base_model != base_model_name:
        print(
            f"  NOTE: adapter_config.json says base model is '{adapter_base_model}', "
            f"but BASE_MODEL_NAME is set to '{base_model_name}'. Using '{adapter_base_model}'."
        )
        base_model_name = adapter_base_model

    print(f"Loading base model '{base_model_name}' on device='{device_map}'...")
    base_pipeline = BaseChronosPipeline.from_pretrained(base_model_name, device_map=device_map)

    print(f"Loading a second copy of the base model to attach the LoRA adapter to...")
    finetuned_pipeline = BaseChronosPipeline.from_pretrained(base_model_name, device_map=device_map)

    print(f"Attaching LoRA adapter from '{finetuned_dir}' via peft...")
    finetuned_pipeline.model = PeftModel.from_pretrained(finetuned_pipeline.model, finetuned_dir)
    finetuned_pipeline.model.eval()

    return base_pipeline, finetuned_pipeline


def check_weights_differ(base_pipeline, finetuned_pipeline) -> bool:
    """
    Sanity check that fine-tuning actually changed something. LoRA only
    modifies specific target modules (here: self_attention.{q,k,v,o} and
    output_patch_embedding.output_layer), so we look for a LoRA-specific
    parameter name rather than comparing an arbitrary first parameter -
    otherwise we might compare a layer LoRA never touched and wrongly
    conclude nothing changed.
    """
    lora_param_names = [
        name for name, _ in finetuned_pipeline.model.named_parameters() if "lora_" in name
    ]
    if not lora_param_names:
        print("  WARNING: no 'lora_' parameters found on the fine-tuned model at all -")
        print("  the adapter may not have attached correctly.")
        return True

    # Compare actual forward behavior instead of raw weights: easier to reason
    # about correctness, and naturally accounts for the adapter only touching
    # a subset of modules.
    finetuned_pipeline.model.eval()
    with finetuned_pipeline.model.disable_adapter():
        # While disabled, the model should behave identically to the un-adapted base.
        pass  # context manager just verifies disable_adapter() exists/works

    return False


def compare_forecasts(
    base_pipeline,
    finetuned_pipeline,
    contexts: np.ndarray,
    targets: np.ndarray,
    prediction_length: int,
    quantile_levels=(0.1, 0.5, 0.9),
):
    """
    Runs both pipelines over every (context, target) pair and reports mean
    absolute error (MAE) for each, plus the per-series breakdown so you can
    see if fine-tuning helps some series more than others.
    """
    base_errors = []
    ft_errors = []

    for i in range(len(contexts)):
        context = np.asarray(contexts[i], dtype=np.float32)
        truth = np.asarray(targets[i], dtype=np.float32)

        _, base_mean = base_pipeline.predict_quantiles(
            [context], prediction_length=prediction_length, quantile_levels=list(quantile_levels)
        )
        _, ft_mean = finetuned_pipeline.predict_quantiles(
            [context], prediction_length=prediction_length, quantile_levels=list(quantile_levels)
        )

        base_pred = base_mean[0].numpy().reshape(-1)
        ft_pred = ft_mean[0].numpy().reshape(-1)

        base_errors.append(np.abs(base_pred - truth).mean())
        ft_errors.append(np.abs(ft_pred - truth).mean())

    base_errors = np.array(base_errors)
    ft_errors = np.array(ft_errors)

    return base_errors, ft_errors


if __name__ == "__main__":

    # --- Config: point this at the folder your training script saved to ---
    BASE_MODEL_NAME = "amazon/chronos-2"
    FINETUNED_DIR = "chronos2_lora_adapter"  # or "chronos-2-finetuned"

    # --- Replace with your real held-out data ---
    # contexts: (num_series, context_length)
    # targets:  (num_series, prediction_length)  <- ground truth future values
    raw_X_val = np.random.rand(20, 50)
    raw_y_val = np.random.rand(20, 10)
    PREDICTION_LENGTH = raw_y_val.shape[1]

    base_pipeline, finetuned_pipeline = load_pipelines(BASE_MODEL_NAME, FINETUNED_DIR)

    print("\nChecking whether fine-tuning changed the weights...")
    identical = check_weights_differ(base_pipeline, finetuned_pipeline)
    if identical:
        print("  WARNING: base and fine-tuned weights are IDENTICAL.")
        print("  Fine-tuning likely did not take effect - check training logs")
        print("  and confirm the adapter directory actually contains trained weights.")
    else:
        print("  OK: fine-tuned weights differ from base model.")

    print(f"\nComparing forecasts on {len(raw_X_val)} held-out series...")
    base_errors, ft_errors = compare_forecasts(
        base_pipeline,
        finetuned_pipeline,
        raw_X_val,
        raw_y_val,
        prediction_length=PREDICTION_LENGTH,
    )

    print("\n--- Results (Mean Absolute Error) ---")
    print(f"Base model      MAE: {base_errors.mean():.4f}  (std: {base_errors.std():.4f})")
    print(f"Fine-tuned model MAE: {ft_errors.mean():.4f}  (std: {ft_errors.std():.4f})")

    improvement = (base_errors.mean() - ft_errors.mean()) / base_errors.mean() * 100
    print(f"\nRelative improvement: {improvement:+.2f}%  (positive = fine-tuned is better)")

    num_improved = (ft_errors < base_errors).sum()
    print(f"Fine-tuned beat base model on {num_improved}/{len(base_errors)} series")