#!/usr/bin/env python3
"""
ML Motion Detection - Feature Set Sweep

Run a reproducible feature-selection campaign for the MLP detector without
overwriting production exports during intermediate experiments.

The script evaluates:
  - baseline feature set
  - targeted single-feature ablations
  - curated candidate feature sets
  - seed robustness for the best candidates

Final promotion exports the winning model only after it clears the paired
real-data gate and the long-recording holdout gate.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from statistics import median
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

ROOT_DIR = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT_DIR / "tools"
SRC_DIR = ROOT_DIR / "src"
TESTS_DIR = ROOT_DIR / "tests"
MODELS_DIR = ROOT_DIR / "models"

for path in (TOOLS_DIR, SRC_DIR, TESTS_DIR):
    sys.path.insert(0, str(path))

from config import DEFAULT_SUBCARRIERS, SEG_WINDOW_SIZE  # noqa: E402
from csi_utils import find_dataset, load_npz_as_packets  # noqa: E402
from features import DEFAULT_FEATURES, extract_features_by_name  # noqa: E402
from segmentation import SegmentationContext  # noqa: E402
from conftest import get_available_long_test_datasets  # noqa: E402


BASELINE_FEATURES = list(DEFAULT_FEATURES)
ABLATION_FEATURES = [
    "turb_entropy",
    "turb_slope",
    "turb_skewness",
    "turb_kurtosis",
    "turb_iqr",
    "turb_mad",
]

CANDIDATE_FEATURE_SETS = {
    "baseline-12": BASELINE_FEATURES,
    "drop-entropy-slope": [
        name for name in BASELINE_FEATURES
        if name not in {"turb_entropy", "turb_slope"}
    ],
    "drop-entropy-slope-mad": [
        name for name in BASELINE_FEATURES
        if name not in {"turb_entropy", "turb_slope", "turb_mad"}
    ],
    "drop-entropy-slope-iqr": [
        name for name in BASELINE_FEATURES
        if name not in {"turb_entropy", "turb_slope", "turb_iqr"}
    ],
    "drop-entropy-slope-kurtosis": [
        name for name in BASELINE_FEATURES
        if name not in {"turb_entropy", "turb_slope", "turb_kurtosis"}
    ],
}

PAIRED_CHIPS = ["C3", "C5", "C6", "ESP32", "S3"]
LONG_CHIPS = ["C3", "C5", "C6", "S3"]
BASELINE_SEED = 20260518
INITIAL_SEEDS = [20260518, 20260519, 20260520]
FINAL_SEEDS = [20260518, 20260519, 20260520, 20260521, 20260522]


def _load_train_module():
    train_path = TOOLS_DIR / "10_train_ml_model.py"
    spec = importlib.util.spec_from_file_location("train_ml_model_sweep", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = _load_train_module()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full MLP feature-set optimization campaign."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "ml_feature_sweep_results.json",
        help="JSON file where experiment results are saved",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "robust", "clipped_standard"],
        default=TRAIN.DEFAULT_SCALER_MODE,
        help="Feature normalization mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN.DEFAULT_BATCH_SIZE,
        help="Mini-batch size for Keras training",
    )
    parser.add_argument(
        "--fp-weight",
        type=float,
        default=TRAIN.DEFAULT_FP_WEIGHT,
        help="Multiplier for the IDLE class weight",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Optional environment filter (same syntax as 10_train_ml_model.py)",
    )
    parser.add_argument(
        "--exclude-chip",
        type=str,
        default=",".join(TRAIN.DEFAULT_EXCLUDED_CHIPS),
        help="Optional chip exclusions (same syntax as 10_train_ml_model.py)",
    )
    parser.add_argument(
        "--positive-chip-boost",
        type=str,
        default=None,
        help="Optional motion boost factors (same syntax as 10_train_ml_model.py)",
    )
    parser.add_argument(
        "--skip-final-export",
        action="store_true",
        help="Stop after selecting the winning feature set",
    )
    return parser.parse_args()


def ensure_runtime():
    with TRAIN.suppress_stderr():
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        try:
            import absl.logging

            absl.logging.set_verbosity(absl.logging.ERROR)
            absl.logging.set_stderrthreshold(absl.logging.ERROR)
        except ImportError:
            pass


def build_dataset(feature_names, args, seed):
    ensure_runtime()
    TRAIN.set_global_determinism(seed)

    all_packets, stats = TRAIN.load_all_data(
        environment_filter=args.environment,
        excluded_chips=args.exclude_chip,
    )
    if not stats["chips"]:
        raise RuntimeError("No datasets found for ML training")

    X, y, actual_feature_names, sample_context = TRAIN.extract_features(
        all_packets,
        subcarriers=DEFAULT_SUBCARRIERS,
        feature_names=feature_names,
    )
    tuning_map = TRAIN.build_gridsearch_tuning_map(
        TRAIN.load_dataset_info(),
        DEFAULT_SUBCARRIERS,
        default_threshold=1.0,
    )
    sample_weights = TRAIN.compute_mvs_guided_sample_weights(
        all_packets, tuning_map, window_size=SEG_WINDOW_SIZE
    )
    sample_weights, _ = TRAIN.apply_positive_chip_boost(
        sample_weights,
        sample_context,
        y,
        args.positive_chip_boost,
    )
    return all_packets, stats, X, y, actual_feature_names, sample_context, sample_weights


def train_feature_set(feature_names, args, seed):
    (
        all_packets,
        stats,
        X,
        y,
        actual_feature_names,
        sample_context,
        sample_weights,
    ) = build_dataset(feature_names, args, seed)

    eval_groups = sample_context[TRAIN.DEFAULT_PRIMARY_GROUP_KEY]
    with TRAIN.suppress_stderr():
        cv_results = TRAIN.cross_validate(
            X,
            y,
            hidden_layers=list(TRAIN.DEFAULT_HIDDEN_LAYERS),
            n_folds=TRAIN.DEFAULT_CV_FOLDS,
            max_epochs=TRAIN.DEFAULT_MAX_EPOCHS,
            fp_weight=args.fp_weight,
            sample_weight=sample_weights,
            groups=eval_groups,
            sample_context=sample_context,
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            block_stride=SEG_WINDOW_SIZE,
            block_group_key=TRAIN.DEFAULT_BLOCK_GROUP_KEY,
            report_group_keys=TRAIN.DEFAULT_REPORT_GROUP_KEYS,
            seed=seed,
        )

    scaler = TRAIN.build_preprocessor(args.scaler)
    X_scaled = scaler.fit_transform(X)
    with TRAIN.suppress_stderr():
        model = TRAIN.train_model(
            X_scaled,
            y,
            hidden_layers=list(TRAIN.DEFAULT_HIDDEN_LAYERS),
            max_epochs=TRAIN.DEFAULT_MAX_EPOCHS,
            fp_weight=args.fp_weight,
            sample_weight=sample_weights,
            batch_size=args.batch_size,
            seed=TRAIN.derive_seed(seed, 10_000),
        )

    return {
        "packets": all_packets,
        "stats": stats,
        "X": X,
        "y": y,
        "feature_names": actual_feature_names,
        "sample_context": sample_context,
        "sample_weights": sample_weights,
        "cv": cv_results,
        "model": model,
        "scaler": scaler,
    }


def summarize_gate(by_chip):
    rows = list(by_chip.values())
    pass_count = sum(
        1 for row in rows if row["recall"] > 95.0 and row["fp_rate"] < 5.0
    )
    return {
        "by_chip": by_chip,
        "pass_count": int(pass_count),
        "mean_recall": float(sum(row["recall"] for row in rows) / len(rows)),
        "worst_chip_recall": float(min(row["recall"] for row in rows)),
        "max_fp_rate": float(max(row["fp_rate"] for row in rows)),
        "mean_f1": float(sum(row["f1"] for row in rows) / len(rows)),
        "worst_chip_f1": float(min(row["f1"] for row in rows)),
        "total_fp": int(sum(row["fp"] for row in rows)),
    }


class StreamingEvaluator:
    """Evaluate a trained Keras model with the runtime-equivalent feature path."""

    def __init__(self, model, scaler, feature_names):
        self.feature_names = list(feature_names)
        self.center, self.scale = TRAIN.get_preprocessor_arrays(scaler)
        raw_weights = model.get_weights()
        self.layers = []
        for idx in range(0, len(raw_weights), 2):
            weights = raw_weights[idx]
            biases = raw_weights[idx + 1]
            is_output = idx == len(raw_weights) - 2
            self.layers.append((weights, biases, is_output))
        self.context = SegmentationContext(
            window_size=SEG_WINDOW_SIZE,
            threshold=1.0,
            enable_hampel=True,
            hampel_window=7,
            hampel_threshold=5.0,
        )
        self.context.use_cv_normalization = False
        self.current_amplitudes = None

    def _predict_probability(self, features):
        activations = (np.asarray(features, dtype=np.float32) - self.center) / self.scale
        for weights, biases, is_output in self.layers:
            activations = activations @ weights + biases
            if not is_output:
                activations = activations.clip(min=0.0)

        logit = float(activations.reshape(-1)[0]) / float(TRAIN.DEFAULT_ML_TEMPERATURE)
        if logit < -20.0:
            return 0.0
        if logit > 20.0:
            return 1.0
        return 1.0 / (1.0 + np.exp(-logit))

    def process_packet(self, csi_data):
        turbulence, amplitudes = self.context.calculate_spatial_turbulence(
            csi_data,
            DEFAULT_SUBCARRIERS,
            return_amplitudes=True,
        )
        self.current_amplitudes = amplitudes
        self.context.add_turbulence(turbulence)
        if self.context.buffer_count < self.context.window_size:
            return None

        idx = self.context.buffer_index
        turb_list = (
            self.context.turbulence_buffer[idx:]
            + self.context.turbulence_buffer[:idx]
        )
        features = extract_features_by_name(
            turb_list,
            len(turb_list),
            amplitudes=self.current_amplitudes,
            feature_names=self.feature_names,
        )
        return float(self._predict_probability(features))


def evaluate_split(model, scaler, feature_names, baseline_packets, movement_packets):
    evaluator = StreamingEvaluator(model, scaler, feature_names)

    warmup = SEG_WINDOW_SIZE
    baseline_eval_count = max(len(baseline_packets) - warmup, 0)
    movement_eval_count = max(len(movement_packets) - warmup, 0)
    baseline_motion_packets = 0
    movement_with_motion = 0
    movement_without_motion = 0

    for i, pkt in enumerate(baseline_packets):
        prob = evaluator.process_packet(pkt["csi_data"])
        if i >= warmup and prob is not None and prob > 0.5:
            baseline_motion_packets += 1

    for i, pkt in enumerate(movement_packets):
        prob = evaluator.process_packet(pkt["csi_data"])
        if i >= warmup and prob is not None:
            if prob > 0.5:
                movement_with_motion += 1
            else:
                movement_without_motion += 1

    tp = movement_with_motion
    fn = movement_without_motion
    fp = baseline_motion_packets
    tn = max(baseline_eval_count - baseline_motion_packets, 0)
    recall = tp / (tp + fn) * 100.0 if (tp + fn) else 0.0
    precision = tp / (tp + fp) * 100.0 if (tp + fp) else 0.0
    fp_rate = fp / baseline_eval_count * 100.0 if baseline_eval_count else 0.0
    f1 = (
        2 * (precision / 100.0) * (recall / 100.0) / ((precision + recall) / 100.0) * 100.0
        if (precision + recall)
        else 0.0
    )
    return {
        "recall": float(recall),
        "precision": float(precision),
        "fp_rate": float(fp_rate),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "baseline_eval_count": int(baseline_eval_count),
        "movement_eval_count": int(movement_eval_count),
    }


def evaluate_paired_gate(model, scaler, feature_names):
    by_chip = {}
    for chip in PAIRED_CHIPS:
        try:
            baseline_path, movement_path, _ = find_dataset(chip=chip, num_sc=64)
        except FileNotFoundError:
            continue
        baseline_packets = load_npz_as_packets(baseline_path)[300:]
        movement_packets = load_npz_as_packets(movement_path)
        by_chip[chip] = evaluate_split(
            model,
            scaler,
            feature_names,
            baseline_packets,
            movement_packets,
        )
    return summarize_gate(by_chip)


def evaluate_long_gate(model, scaler, feature_names):
    by_chip = {}
    for _, baseline_packets, movement_packets, _, chip, _ in get_available_long_test_datasets(
        chips=LONG_CHIPS
    ):
        by_chip[chip] = evaluate_split(
            model,
            scaler,
            feature_names,
            baseline_packets,
            movement_packets,
        )
    return summarize_gate(by_chip)


def slim_cv_result(cv_results):
    session_report = cv_results.get("group_reports", {}).get("session_group", {})
    chip_report = cv_results.get("group_reports", {}).get("chip", {})
    return {
        "f1_mean": float(cv_results["f1_mean"]),
        "f1_std": float(cv_results["f1_std"]),
        "oof_f1": float(cv_results["oof_f1"]),
        "recall_mean": float(cv_results["recall_mean"]),
        "fp_rate_mean": float(cv_results["fp_rate_mean"]),
        "candidate_key": list(TRAIN.build_candidate_key(cv_results)),
        "worst_session_recall": float(
            session_report.get("worst_recall", {}).get("recall", 0.0)
        ),
        "worst_session_fp_rate": float(
            session_report.get("worst_fp_rate", {}).get("fp_rate", 0.0)
        ),
        "worst_chip_recall": float(
            chip_report.get("worst_recall", {}).get("recall", 0.0)
        ),
    }


def evaluate_feature_set(name, feature_names, seed, args):
    trained = train_feature_set(feature_names, args, seed)
    paired = evaluate_paired_gate(
        trained["model"], trained["scaler"], trained["feature_names"]
    )
    long_gate = evaluate_long_gate(
        trained["model"], trained["scaler"], trained["feature_names"]
    )
    return {
        "name": name,
        "seed": int(seed),
        "features": list(trained["feature_names"]),
        "n_features": len(trained["feature_names"]),
        "cv": slim_cv_result(trained["cv"]),
        "paired": paired,
        "long": long_gate,
    }


def campaign_rank_key(result):
    long_gate = result["long"]
    paired = result["paired"]
    cv = result["cv"]
    return (
        long_gate["pass_count"],
        paired["pass_count"],
        long_gate["worst_chip_f1"],
        paired["worst_chip_f1"],
        -long_gate["max_fp_rate"],
        -paired["max_fp_rate"],
        cv["oof_f1"],
        cv["f1_mean"],
    )


def write_results(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def select_top_feature_sets(results, limit):
    ordered = sorted(results, key=campaign_rank_key, reverse=True)
    return [item["name"] for item in ordered[:limit]]


def aggregate_seed_runs(name, runs):
    long_worst_f1 = [run["long"]["worst_chip_f1"] for run in runs]
    long_max_fp = [run["long"]["max_fp_rate"] for run in runs]
    paired_pass = [run["paired"]["pass_count"] for run in runs]
    return {
        "name": name,
        "seeds": [run["seed"] for run in runs],
        "median_long_worst_chip_f1": float(median(long_worst_f1)),
        "median_long_max_fp_rate": float(median(long_max_fp)),
        "median_paired_pass_count": float(median(paired_pass)),
        "best_single_run": max(runs, key=campaign_rank_key),
        "runs": runs,
    }


def aggregate_rank_key(summary):
    return (
        summary["median_paired_pass_count"],
        summary["median_long_worst_chip_f1"],
        -summary["median_long_max_fp_rate"],
        campaign_rank_key(summary["best_single_run"]),
    )


def export_and_validate_winner(feature_names, seed, args):
    exit_code, _, cv_results = TRAIN.train_all(
        fp_weight=args.fp_weight,
        seed=seed,
        feature_names=feature_names,
        hidden_layers=list(TRAIN.DEFAULT_HIDDEN_LAYERS),
        scaler_mode=args.scaler,
        batch_size=args.batch_size,
        export_artifacts=True,
        environment_filter=args.environment,
        excluded_chips=args.exclude_chip,
        positive_chip_boost=args.positive_chip_boost,
    )
    if exit_code != 0:
        raise RuntimeError("Final export training failed")

    paired_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_validation_real_data.py::TestPerformanceMetrics::test_ml_detection_accuracy",
        "-v",
        "-s",
    ]
    long_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_validation_long_recordings.py::TestLongRecordings::test_ml_vs_test_recordings",
        "-v",
        "-s",
    ]

    paired_proc = TRAIN.subprocess.run(
        paired_cmd,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    long_proc = TRAIN.subprocess.run(
        long_cmd,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    return {
        "cv": slim_cv_result(cv_results),
        "paired_returncode": int(paired_proc.returncode),
        "paired_output": paired_proc.stdout,
        "long_returncode": int(long_proc.returncode),
        "long_output": long_proc.stdout,
    }


def main():
    args = parse_args()
    args.positive_chip_boost = TRAIN.parse_positive_chip_boost(args.positive_chip_boost)
    results = {
        "config": {
            "scaler": args.scaler,
            "batch_size": args.batch_size,
            "fp_weight": args.fp_weight,
            "environment": args.environment,
            "exclude_chip": args.exclude_chip,
            "positive_chip_boost": args.positive_chip_boost,
        },
        "baseline": None,
        "ablations": [],
        "candidate_sets": [],
        "seed_filter": [],
        "seed_finalists": [],
        "winner": None,
        "final_export": None,
    }

    print("== Baseline ==")
    baseline = evaluate_feature_set(
        "baseline-12",
        CANDIDATE_FEATURE_SETS["baseline-12"],
        BASELINE_SEED,
        args,
    )
    results["baseline"] = baseline
    write_results(args.output, results)
    print(
        f"baseline-12 seed={BASELINE_SEED} | "
        f"OOF={baseline['cv']['oof_f1']:.1f}% | "
        f"paired pass={baseline['paired']['pass_count']} | "
        f"long pass={baseline['long']['pass_count']} | "
        f"long worst F1={baseline['long']['worst_chip_f1']:.1f}%"
    )

    print("\n== Ablations ==")
    for removed in ABLATION_FEATURES:
        name = f"drop-{removed}"
        feature_names = [item for item in BASELINE_FEATURES if item != removed]
        run = evaluate_feature_set(name, feature_names, BASELINE_SEED, args)
        results["ablations"].append(run)
        write_results(args.output, results)
        print(
            f"{name} | OOF={run['cv']['oof_f1']:.1f}% | "
            f"paired pass={run['paired']['pass_count']} | "
            f"long pass={run['long']['pass_count']} | "
            f"long worst F1={run['long']['worst_chip_f1']:.1f}%"
        )

    print("\n== Candidate feature sets ==")
    for name, feature_names in CANDIDATE_FEATURE_SETS.items():
        run = evaluate_feature_set(name, feature_names, BASELINE_SEED, args)
        results["candidate_sets"].append(run)
        write_results(args.output, results)
        print(
            f"{name} | features={len(feature_names)} | OOF={run['cv']['oof_f1']:.1f}% | "
            f"paired pass={run['paired']['pass_count']} | "
            f"long pass={run['long']['pass_count']} | "
            f"long worst F1={run['long']['worst_chip_f1']:.1f}%"
        )

    top2 = select_top_feature_sets(results["candidate_sets"], limit=2)
    print(f"\nTop 2 after single-seed screening: {', '.join(top2)}")

    print("\n== 3-seed robustness filter ==")
    for name in top2:
        feature_names = CANDIDATE_FEATURE_SETS[name]
        runs = [
            evaluate_feature_set(name, feature_names, seed, args)
            for seed in INITIAL_SEEDS
        ]
        summary = aggregate_seed_runs(name, runs)
        results["seed_filter"].append(summary)
        write_results(args.output, results)
        print(
            f"{name} | median long worst F1={summary['median_long_worst_chip_f1']:.1f}% | "
            f"median long max FP={summary['median_long_max_fp_rate']:.1f}%"
        )

    finalists = [
        item["name"]
        for item in sorted(results["seed_filter"], key=aggregate_rank_key, reverse=True)[:2]
    ]
    print(f"\nFinalists for 5-seed head-to-head: {', '.join(finalists)}")

    print("\n== 5-seed final comparison ==")
    for name in finalists:
        feature_names = CANDIDATE_FEATURE_SETS[name]
        runs = [
            evaluate_feature_set(name, feature_names, seed, args)
            for seed in FINAL_SEEDS
        ]
        summary = aggregate_seed_runs(name, runs)
        results["seed_finalists"].append(summary)
        write_results(args.output, results)
        print(
            f"{name} | median long worst F1={summary['median_long_worst_chip_f1']:.1f}% | "
            f"median long max FP={summary['median_long_max_fp_rate']:.1f}% | "
            f"median paired pass={summary['median_paired_pass_count']:.1f}"
        )

    winner = max(results["seed_finalists"], key=aggregate_rank_key)
    results["winner"] = {
        "name": winner["name"],
        "features": CANDIDATE_FEATURE_SETS[winner["name"]],
        "seed": winner["best_single_run"]["seed"],
        "summary": winner,
    }
    write_results(args.output, results)
    print(
        f"\nWinner: {winner['name']} "
        f"(seed {winner['best_single_run']['seed']}, "
        f"median long worst F1={winner['median_long_worst_chip_f1']:.1f}%)"
    )

    if args.skip_final_export:
        return 0

    print("\n== Final export and repo tests ==")
    final_export = export_and_validate_winner(
        CANDIDATE_FEATURE_SETS[winner["name"]],
        winner["best_single_run"]["seed"],
        args,
    )
    results["final_export"] = final_export
    write_results(args.output, results)
    print(
        f"Final export done | paired rc={final_export['paired_returncode']} | "
        f"long rc={final_export['long_returncode']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
