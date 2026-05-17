"""
Bug Condition Exploration Test — MLDetector Production Instability

Task 1.1: Verify that the training dataset (with ESP32 files included) produces
heterogeneous turbulence distributions for gain_locked=False vs gain_locked=True
packets. This test is EXPECTED TO FAIL on unfixed code, confirming the bug.

The bug: extract_features() uses `use_cv_norm = not pkt.get('gain_locked', True)`,
so ESP32 packets (gain_locked=False) produce CV-normalised turbulence (~0.05–0.15)
while gain-lock chip packets (gain_locked=True) produce raw-std turbulence (~2–8).
The StandardScaler is then fitted on this mixed distribution, corrupting the
ML_FEATURE_MEAN / ML_FEATURE_SCALE parameters exported to ml_weights.h.

Expected counterexample:
  - gain_locked=False (ESP32, CV norm): turbulence mean ~0.05–0.15
  - gain_locked=True  (C3/S3/C5/C6, raw std): turbulence mean ~2–8
  The two groups have significantly different distributions → scaler is corrupted.

**Validates: Requirements 1.1 (bug condition exploration)**
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Add src and tools to path
SRC_PATH = Path(__file__).parent.parent / 'src'
TOOLS_PATH = Path(__file__).parent.parent / 'tools'
sys.path.insert(0, str(TOOLS_PATH))
sys.path.insert(0, str(SRC_PATH))

from config import DEFAULT_SUBCARRIERS, SEG_WINDOW_SIZE, HAMPEL_WINDOW, HAMPEL_THRESHOLD
from segmentation import SegmentationContext


# ---------------------------------------------------------------------------
# Helpers — replicate extract_features() turbulence extraction logic
# ---------------------------------------------------------------------------

def _extract_turbulence_per_packet(packets, subcarriers=None):
    """
    Replicate the per-packet turbulence extraction from extract_features() in
    10_train_ml_model.py, returning one turbulence value per packet together
    with the gain_locked flag for that packet.

    This mirrors the exact logic at the heart of the bug:
        use_cv_norm = not pkt.get('gain_locked', True)
        turb, _ = SegmentationContext.compute_spatial_turbulence(
            csi_data, subcarriers, use_cv_normalization=use_cv_norm
        )

    Returns:
        list of (turbulence: float, gain_locked: bool)
    """
    if subcarriers is None:
        subcarriers = DEFAULT_SUBCARRIERS

    results = []
    for pkt in packets:
        csi_data = pkt.get('csi_data')
        if csi_data is None:
            continue
        gain_locked = bool(pkt.get('gain_locked', True))
        use_cv_norm = not gain_locked  # ← the buggy line from extract_features()
        turb, _ = SegmentationContext.compute_spatial_turbulence(
            csi_data, subcarriers, use_cv_normalization=use_cv_norm
        )
        results.append((float(turb), gain_locked))
    return results


# ---------------------------------------------------------------------------
# Test 1.1 — Mixed normalization distribution check
# ---------------------------------------------------------------------------

class TestBugConditionExploration:
    """
    Exploration tests that confirm the bug exists on unfixed code.

    These tests are EXPECTED TO FAIL on unfixed code.
    Failure = bug confirmed (SUCCESS for an exploration test).
    """

    def test_turbulence_distributions_are_homogeneous(self):
        """
        EXPLORATION TEST (expected to FAIL on unfixed code).

        Loads the full training dataset (including ESP32 files), extracts
        turbulence values using the current extract_features() logic, then
        asserts that the distributions for gain_locked=False and gain_locked=True
        packets are statistically homogeneous (same mean/std range).

        On unfixed code this assertion FAILS because:
          - gain_locked=False (ESP32, CV norm) → turbulence ~0.05–0.15
          - gain_locked=True  (C3/S3/C5/C6, raw std) → turbulence ~2–8

        **Validates: Requirements 1.1**
        """
        # ----------------------------------------------------------------
        # 1. Load the full training dataset (ESP32 included — no exclusions)
        # ----------------------------------------------------------------
        from csi_utils import load_npz_as_packets, DATA_DIR
        import json

        dataset_info_path = DATA_DIR / 'dataset_info.json'
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)

        # Build filename → gain_locked mapping from dataset_info
        gain_locked_map = {}
        for label, file_list in dataset_info.get('files', {}).items():
            for entry in file_list:
                fname = entry.get('filename', '')
                if fname:
                    gain_locked_map[fname] = bool(entry.get('gain_locked', True))

        all_packets = []
        for subdir in sorted(DATA_DIR.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith('.'):
                continue
            for npz_file in sorted(subdir.glob('*.npz')):
                try:
                    packets = load_npz_as_packets(npz_file)
                    if not packets:
                        continue
                    label = packets[0].get('label', subdir.name).lower()
                    # Keep only training labels (baseline / movement)
                    if label not in ('baseline', 'movement'):
                        continue
                    # Apply gain_locked from dataset_info (authoritative source)
                    gain_locked = gain_locked_map.get(npz_file.name,
                                                      bool(packets[0].get('gain_locked', True)))
                    for pkt in packets:
                        pkt['gain_locked'] = gain_locked
                    all_packets.extend(packets)
                except Exception as e:
                    print(f"  Warning: could not load {npz_file.name}: {e}")

        assert len(all_packets) > 0, "No training packets loaded — check data/ directory"

        # ----------------------------------------------------------------
        # 2. Extract turbulence per packet using the current (buggy) logic
        # ----------------------------------------------------------------
        turb_results = _extract_turbulence_per_packet(all_packets)
        assert len(turb_results) > 0, "No turbulence values extracted"

        # Separate by gain_locked flag
        cv_norm_turbulences = [t for t, gl in turb_results if not gl]   # gain_locked=False → CV
        raw_std_turbulences = [t for t, gl in turb_results if gl]        # gain_locked=True  → raw std

        assert len(cv_norm_turbulences) > 0, (
            "No gain_locked=False (ESP32) packets found — "
            "ensure ESP32 files are present in data/ and not excluded"
        )
        assert len(raw_std_turbulences) > 0, (
            "No gain_locked=True packets found"
        )

        cv_mean  = float(np.mean(cv_norm_turbulences))
        cv_std   = float(np.std(cv_norm_turbulences))
        raw_mean = float(np.mean(raw_std_turbulences))
        raw_std  = float(np.std(raw_std_turbulences))

        # ----------------------------------------------------------------
        # 3. Report the actual distributions (visible in pytest output)
        # ----------------------------------------------------------------
        print(f"\n--- Turbulence distribution report ---")
        print(f"  gain_locked=False (ESP32, CV norm):  n={len(cv_norm_turbulences):5d}  "
              f"mean={cv_mean:.4f}  std={cv_std:.4f}")
        print(f"  gain_locked=True  (C3/S3/C5/C6, raw std): n={len(raw_std_turbulences):5d}  "
              f"mean={raw_mean:.4f}  std={raw_std:.4f}")
        print(f"  Ratio of means (raw/cv): {raw_mean / cv_mean:.1f}x" if cv_mean > 0 else "")
        print(f"--------------------------------------")

        # ----------------------------------------------------------------
        # 4. Assert homogeneity — WILL FAIL on unfixed code
        #
        # Homogeneity criterion: the means of the two groups must be within
        # a factor of 2 of each other (i.e. |raw_mean / cv_mean| < 2).
        # On unfixed code the ratio is ~20–100x, so this assertion fails.
        # ----------------------------------------------------------------
        ratio = raw_mean / cv_mean if cv_mean > 1e-9 else float('inf')

        assert ratio < 2.0, (
            f"BUG CONFIRMED: turbulence distributions are NOT homogeneous.\n"
            f"  gain_locked=False (CV norm):  mean={cv_mean:.4f}, std={cv_std:.4f}\n"
            f"  gain_locked=True  (raw std):  mean={raw_mean:.4f}, std={raw_std:.4f}\n"
            f"  Ratio of means (raw/cv): {ratio:.1f}x  (expected < 2.0 for homogeneous dataset)\n"
            f"  The StandardScaler is fitted on this mixed distribution, corrupting\n"
            f"  ML_FEATURE_MEAN / ML_FEATURE_SCALE in ml_weights.h."
        )

    def test_scaler_corrupted_by_mixed_distribution(self):
        """
        EXPLORATION TEST (expected to FAIL on unfixed code).

        Fits a StandardScaler on the MIXED dataset (all packets, both
        gain_locked=True and False) and on the GAIN-LOCK-ONLY dataset
        (only gain_locked=True packets), then asserts that the scaler
        ``mean_`` parameters are within 10% of each other.

        On unfixed code this assertion FAILS because:
          - Mixed scaler mean is pulled down by CV-norm values (~0.18)
          - Gain-lock-only scaler mean reflects raw std values (~5.38)
          The two means differ by >> 10%, confirming the scaler is corrupted.

        **Validates: Requirements 1.2**
        """
        from sklearn.preprocessing import StandardScaler
        from csi_utils import load_npz_as_packets, DATA_DIR
        import json

        # ----------------------------------------------------------------
        # 1. Load the full training dataset (ESP32 included — no exclusions)
        # ----------------------------------------------------------------
        dataset_info_path = DATA_DIR / 'dataset_info.json'
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)

        # Build filename → gain_locked mapping from dataset_info
        gain_locked_map = {}
        for label, file_list in dataset_info.get('files', {}).items():
            for entry in file_list:
                fname = entry.get('filename', '')
                if fname:
                    gain_locked_map[fname] = bool(entry.get('gain_locked', True))

        all_packets = []
        for subdir in sorted(DATA_DIR.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith('.'):
                continue
            for npz_file in sorted(subdir.glob('*.npz')):
                try:
                    packets = load_npz_as_packets(npz_file)
                    if not packets:
                        continue
                    label = packets[0].get('label', subdir.name).lower()
                    if label not in ('baseline', 'movement'):
                        continue
                    gain_locked = gain_locked_map.get(npz_file.name,
                                                      bool(packets[0].get('gain_locked', True)))
                    for pkt in packets:
                        pkt['gain_locked'] = gain_locked
                    all_packets.extend(packets)
                except Exception as e:
                    print(f"  Warning: could not load {npz_file.name}: {e}")

        assert len(all_packets) > 0, "No training packets loaded — check data/ directory"

        # ----------------------------------------------------------------
        # 2. Extract turbulence per packet using the current (buggy) logic
        # ----------------------------------------------------------------
        turb_results = _extract_turbulence_per_packet(all_packets)
        assert len(turb_results) > 0, "No turbulence values extracted"

        all_turbulences   = np.array([t for t, _  in turb_results]).reshape(-1, 1)
        gainlock_only     = np.array([t for t, gl in turb_results if gl]).reshape(-1, 1)

        assert len(gainlock_only) > 0, (
            "No gain_locked=True packets found — cannot fit gain-lock-only scaler"
        )
        assert len(all_turbulences) > len(gainlock_only), (
            "No gain_locked=False (ESP32) packets found — "
            "ensure ESP32 files are present in data/ and not excluded"
        )

        # ----------------------------------------------------------------
        # 3. Fit scalers on mixed and gain-lock-only datasets
        # ----------------------------------------------------------------
        scaler_mixed     = StandardScaler()
        scaler_gainlock  = StandardScaler()

        scaler_mixed.fit(all_turbulences)
        scaler_gainlock.fit(gainlock_only)

        mixed_mean    = float(scaler_mixed.mean_[0])
        gainlock_mean = float(scaler_gainlock.mean_[0])

        # ----------------------------------------------------------------
        # 4. Report the actual scaler mean_ values
        # ----------------------------------------------------------------
        print(f"\n--- Scaler mean_ report ---")
        print(f"  Mixed dataset scaler mean_:      {mixed_mean:.4f}")
        print(f"  Gain-lock-only scaler mean_:     {gainlock_mean:.4f}")
        if gainlock_mean > 1e-9:
            diff_pct = abs(mixed_mean - gainlock_mean) / abs(gainlock_mean) * 100
            print(f"  Relative difference:             {diff_pct:.1f}%  (threshold: 10%)")
        print(f"---------------------------")

        # ----------------------------------------------------------------
        # 5. Assert means are within 10% — WILL FAIL on unfixed code
        #
        # On unfixed code the mixed scaler mean is pulled down by CV-norm
        # values (~0.18) while the gain-lock-only mean is ~5.38, giving a
        # relative difference >> 10%.  The assertion fails, confirming the
        # scaler is corrupted by the mixed distribution.
        # ----------------------------------------------------------------
        assert gainlock_mean > 1e-9, "Gain-lock-only scaler mean is near zero — unexpected"

        relative_diff = abs(mixed_mean - gainlock_mean) / abs(gainlock_mean)

        assert relative_diff < 0.10, (
            f"BUG CONFIRMED: StandardScaler is corrupted by mixed distribution.\n"
            f"  Mixed dataset scaler mean_:      {mixed_mean:.4f}\n"
            f"  Gain-lock-only scaler mean_:     {gainlock_mean:.4f}\n"
            f"  Relative difference:             {relative_diff * 100:.1f}%  (expected < 10%)\n"
            f"  The mixed scaler mean is pulled down by CV-norm values (~0.18) while\n"
            f"  the gain-lock-only mean is ~5.38.  ML_FEATURE_MEAN in ml_weights.h\n"
            f"  reflects the corrupted mixed mean, causing systematic normalisation\n"
            f"  errors on C3/S3/C5/C6 devices in production."
        )
