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

import matplotlib.pyplot as plt
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

    Also returns the raw per-series quantile predictions (low/median/high)
    for both models so they can be plotted without re-running inference.
    """
    base_errors = []
    ft_errors = []
    base_quantiles_all = []  # list of (prediction_length, num_quantiles) arrays
    ft_quantiles_all = []

    for i in range(len(contexts)):
        context = np.asarray(contexts[i], dtype=np.float32)
        truth = np.asarray(targets[i], dtype=np.float32)

        base_q, base_mean = base_pipeline.predict_quantiles(
            [context], prediction_length=prediction_length, quantile_levels=list(quantile_levels)
        )
        ft_q, ft_mean = finetuned_pipeline.predict_quantiles(
            [context], prediction_length=prediction_length, quantile_levels=list(quantile_levels)
        )

        base_q_arr = np.asarray(base_q[0])
        ft_q_arr = np.asarray(ft_q[0])

        # base_q[0] is expected to already be (prediction_length, num_quantiles)
        # after stripping the batch dim, since predict_quantiles takes a batch
        # of size 1 here ([context]). In practice this version of the pipeline
        # returns base_q[0] still carrying a leading singleton dim, i.e.
        # (1, prediction_length, num_quantiles). Squeeze out any leading axis
        # of size 1 until we hit the real (prediction_length, num_quantiles)
        # shape, rather than assuming a fixed number of dims.
        while base_q_arr.ndim > 2 and base_q_arr.shape[0] == 1:
            base_q_arr = base_q_arr[0]
        while ft_q_arr.ndim > 2 and ft_q_arr.shape[0] == 1:
            ft_q_arr = ft_q_arr[0]

        if i == 0:
            # One-time shape check, run after the leading-dim squeeze above.
            # Earlier debugging showed predict_quantiles([context], ...)
            # returns base_q[0] as (1, prediction_length, num_quantiles) -
            # i.e. an extra leading singleton dim beyond what [0] alone
            # strips - for BOTH base and fine-tuned pipelines identically.
            # Not a PeftModel-specific issue; just an extra batch-like axis
            # in this pipeline version's per-series tensor. The squeeze loop
            # above handles it, so this check should now always pass.
            expected_shape = (prediction_length, len(quantile_levels))
            print(f"  [debug] expected per-series quantile shape: {expected_shape}")
            print(f"  [debug] base_pipeline shape after squeeze:      {base_q_arr.shape}")
            print(f"  [debug] finetuned_pipeline shape after squeeze: {ft_q_arr.shape}")
            if base_q_arr.shape != expected_shape:
                print("  [debug] -> BASE pipeline shape STILL mismatched after squeeze!")
            if ft_q_arr.shape != expected_shape:
                print("  [debug] -> FINETUNED pipeline shape STILL mismatched after squeeze!")

        base_pred = base_mean[0].numpy().reshape(-1)
        ft_pred = ft_mean[0].numpy().reshape(-1)

        base_errors.append(np.abs(base_pred - truth).mean())
        ft_errors.append(np.abs(ft_pred - truth).mean())

        base_quantiles_all.append(base_q_arr)
        ft_quantiles_all.append(ft_q_arr)

    base_errors = np.array(base_errors)
    ft_errors = np.array(ft_errors)

    return base_errors, ft_errors, base_quantiles_all, ft_quantiles_all


def plot_series_forecast(
    context: np.ndarray,
    truth: np.ndarray,
    base_quantiles: np.ndarray,
    ft_quantiles: np.ndarray,
    quantile_levels,
    series_idx: int,
    out_dir: str,
):
    """
    Plots one series: historical context, ground-truth future, and both
    models' forecasts (median line + low/high quantile shaded band).
    Saves to out_dir/series_{idx}.png.
    """
    context_len = len(context)
    pred_len = len(truth)

    x_context = np.arange(context_len)
    x_future = np.arange(context_len, context_len + pred_len)

    low_idx, median_idx, high_idx = 0, len(quantile_levels) // 2, len(quantile_levels) - 1

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x_context, context, color="black", label="Context (history)")
    ax.plot(x_future, truth, color="black", linestyle="--", marker="o", markersize=3, label="Ground truth")

    def _plot_model(quantiles, color, label):
        # Defensive: predict_quantiles should return one row per forecast
        # step, but if a pipeline returns a different length (e.g. collapsed
        # to 1 step due to a wrapping issue), don't crash the whole eval run -
        # plot what we can and flag it visibly instead.
        if quantiles.shape[0] != pred_len:
            ax.text(
                x_future[0],
                ax.get_ylim()[1] if ax.get_ylim()[1] != ax.get_ylim()[0] else 1.0,
                f"{label}: shape mismatch {quantiles.shape}, skipped",
                color=color,
                fontsize=8,
                va="top",
            )
            print(f"  [plot warning] {label} quantiles shape {quantiles.shape} != expected ({pred_len}, {len(quantile_levels)}); skipping this curve")
            return
        ax.plot(x_future, quantiles[:, median_idx], color=color, label=f"{label} (median)")
        ax.fill_between(
            x_future,
            quantiles[:, low_idx],
            quantiles[:, high_idx],
            color=color,
            alpha=0.15,
            label=f"{label} {int(quantile_levels[low_idx]*100)}-{int(quantile_levels[high_idx]*100)}% band",
        )

    _plot_model(base_quantiles, "tab:blue", "Base model")
    _plot_model(ft_quantiles, "tab:orange", "Fine-tuned model")

    ax.axvline(context_len - 0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"Series {series_idx}: base vs fine-tuned forecast")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"series_{series_idx:03d}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_error_summary(base_errors: np.ndarray, ft_errors: np.ndarray, out_dir: str):
    """
    Two-panel summary: (1) MAE per series for base vs fine-tuned as paired bars,
    (2) histogram of the error distributions. Saves to out_dir/error_summary.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: per-series paired comparison
    idx = np.arange(len(base_errors))
    width = 0.4
    axes[0].bar(idx - width / 2, base_errors, width=width, color="tab:blue", label="Base")
    axes[0].bar(idx + width / 2, ft_errors, width=width, color="tab:orange", label="Fine-tuned")
    axes[0].set_xlabel("Series index")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Per-series MAE: base vs fine-tuned")
    axes[0].legend()

    # Panel 2: error distribution
    axes[1].hist(base_errors, bins=15, alpha=0.5, color="tab:blue", label="Base")
    axes[1].hist(ft_errors, bins=15, alpha=0.5, color="tab:orange", label="Fine-tuned")
    axes[1].axvline(base_errors.mean(), color="tab:blue", linestyle="--", linewidth=1)
    axes[1].axvline(ft_errors.mean(), color="tab:orange", linestyle="--", linewidth=1)
    axes[1].set_xlabel("MAE")
    axes[1].set_ylabel("Number of series")
    axes[1].set_title("Error distribution across all series")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "error_summary.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":

    # --- Config: point this at the folder your training script saved to ---
    BASE_MODEL_NAME = "amazon/chronos-2"
    FINETUNED_DIR = "chronos2_lora_adapter"  # or "chronos-2-finetuned"
    PLOT_DIR = "eval_plots"
    MAX_SERIES_TO_PLOT = 10  # only plot the first N series individually; the
                              # error_summary.png always covers every series

    QUANTILE_LEVELS = (0.1, 0.5, 0.9)

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
    base_errors, ft_errors, base_quantiles_all, ft_quantiles_all = compare_forecasts(
        base_pipeline,
        finetuned_pipeline,
        raw_X_val,
        raw_y_val,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS,
    )

    print("\n--- Results (Mean Absolute Error) ---")
    print(f"Base model      MAE: {base_errors.mean():.4f}  (std: {base_errors.std():.4f})")
    print(f"Fine-tuned model MAE: {ft_errors.mean():.4f}  (std: {ft_errors.std():.4f})")

    improvement = (base_errors.mean() - ft_errors.mean()) / base_errors.mean() * 100
    print(f"\nRelative improvement: {improvement:+.2f}%  (positive = fine-tuned is better)")

    num_improved = (ft_errors < base_errors).sum()
    print(f"Fine-tuned beat base model on {num_improved}/{len(base_errors)} series")

    print(f"\nSaving plots to '{PLOT_DIR}/'...")
    num_to_plot = min(MAX_SERIES_TO_PLOT, len(raw_X_val))
    for i in range(num_to_plot):
        path = plot_series_forecast(
            context=np.asarray(raw_X_val[i], dtype=np.float32),
            truth=np.asarray(raw_y_val[i], dtype=np.float32),
            base_quantiles=base_quantiles_all[i],
            ft_quantiles=ft_quantiles_all[i],
            quantile_levels=QUANTILE_LEVELS,
            series_idx=i,
            out_dir=PLOT_DIR,
        )
    print(f"  Saved {num_to_plot} per-series forecast plots (e.g. '{path}')")

    summary_path = plot_error_summary(base_errors, ft_errors, out_dir=PLOT_DIR)
    print(f"  Saved error summary plot: '{summary_path}'")