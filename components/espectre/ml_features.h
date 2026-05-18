/*
 * ESPectre - ML Feature Extraction
 * 
 * Extracts 9 non-redundant features from CSI data for ML-based motion
 * detection. Port of micro-espectre/src/features.py to C++.
 * 
 * All 9 features are computed from the turbulence buffer (100 samples).
 * 
 * Features (in order):
 *  0. turb_mean      - Mean of turbulence buffer
 *  1. turb_std       - Standard deviation
 *  2. turb_max       - Maximum value
 *  3. turb_min       - Minimum value
 *  4. turb_iqr       - Interquartile range
 *  5. turb_skewness  - Fisher's skewness (3rd moment)
 *  6. turb_autocorr  - Lag-1 autocorrelation
 *  7. turb_mad       - Median absolute deviation
 *  8. waveform_length - Sum of absolute first differences
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include "utils.h"

namespace esphome {
namespace espectre {

// Number of features extracted
constexpr uint8_t ML_NUM_FEATURES = 9;

// Number of entropy bins
constexpr uint8_t ML_ENTROPY_BINS = 10;

// Maximum buffer size for sorting (MAD calculation)
constexpr uint16_t ML_MAX_SORT_SIZE = 200;

/**
 * Calculate Fisher's skewness (third standardized moment).
 * 
 * @param values Array of values
 * @param count Number of values
 * @param mean Pre-computed mean (must be valid)
 * @param std_dev Pre-computed standard deviation (must be valid)
 * @return Skewness coefficient
 */
inline float calc_skewness(const float* values, uint16_t count, float mean, float std_dev) {
    if (count < 3 || std_dev < 1e-10f) return 0.0f;
    
    float m3 = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        m3 += diff * diff * diff;
    }
    m3 /= count;
    
    return m3 / (std_dev * std_dev * std_dev);
}

/**
 * Calculate Fisher's kurtosis (fourth standardized moment, excess).
 * 
 * @param values Array of values
 * @param count Number of values
 * @param mean Pre-computed mean (must be valid)
 * @param std_dev Pre-computed standard deviation (must be valid)
 * @return Excess kurtosis (0 for normal distribution)
 */
inline float calc_kurtosis(const float* values, uint16_t count, float mean, float std_dev) {
    if (count < 4 || std_dev < 1e-10f) return 0.0f;
    
    float m4 = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        float diff2 = diff * diff;
        m4 += diff2 * diff2;
    }
    m4 /= count;
    
    float std4 = std_dev * std_dev * std_dev * std_dev;
    return (m4 / std4) - 3.0f;  // Excess kurtosis
}

/**
 * Calculate Shannon entropy of values.
 * 
 * @param values Array of values
 * @param count Number of values
 * @return Shannon entropy in bits
 */
inline float calc_entropy(const float* values, uint16_t count) {
    if (count < 2) return 0.0f;
    
    // Find min/max
    float min_val = values[0];
    float max_val = values[0];
    for (uint16_t i = 1; i < count; i++) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }
    
    float range = max_val - min_val;
    if (range < 1e-10f) return 0.0f;
    
    // Create histogram
    uint16_t bins[ML_ENTROPY_BINS] = {0};
    float bin_width = range / ML_ENTROPY_BINS;
    
    for (uint16_t i = 0; i < count; i++) {
        int bin_idx = static_cast<int>((values[i] - min_val) / bin_width);
        if (bin_idx >= ML_ENTROPY_BINS) bin_idx = ML_ENTROPY_BINS - 1;
        bins[bin_idx]++;
    }
    
    // Calculate entropy
    float entropy = 0.0f;
    float log2 = std::log(2.0f);
    for (uint8_t i = 0; i < ML_ENTROPY_BINS; i++) {
        if (bins[i] > 0) {
            float p = static_cast<float>(bins[i]) / count;
            entropy -= p * std::log(p) / log2;
        }
    }
    
    return entropy;
}

inline float interpolate_sorted_percentile(const float* sorted_values, uint16_t count,
                                           float percentile) {
    if (count == 0 || sorted_values == nullptr) return 0.0f;
    if (count == 1) return sorted_values[0];

    float position = (count - 1) * (percentile / 100.0f);
    uint16_t lower_idx = static_cast<uint16_t>(position);
    uint16_t upper_idx = lower_idx + 1;
    if (upper_idx >= count) return sorted_values[count - 1];

    float fraction = position - lower_idx;
    float lower = sorted_values[lower_idx];
    float upper = sorted_values[upper_idx];
    return lower * (1.0f - fraction) + upper * fraction;
}

/**
 * Calculate interquartile range (P75 - P25).
 *
 * Robust measure of spread, less sensitive to oscillatory sign flips than
 * zero-crossing rate on quiet-but-noisy windows.
 *
 * @param values Array of values
 * @param count Number of values
 * @return IQR value
 */
inline float calc_iqr(const float* values, uint16_t count) {
    if (count < 2 || count > ML_MAX_SORT_SIZE) return 0.0f;

    float sorted[ML_MAX_SORT_SIZE];
    for (uint16_t i = 0; i < count; i++) {
        sorted[i] = values[i];
    }
    std::sort(sorted, sorted + count);

    float q1 = interpolate_sorted_percentile(sorted, count, 25.0f);
    float q3 = interpolate_sorted_percentile(sorted, count, 75.0f);
    return q3 - q1;
}

/**
 * Calculate lag-k autocorrelation coefficient.
 * 
 * Measures temporal correlation between values separated by 'lag' samples.
 * Higher lag captures longer-term temporal patterns in motion.
 * 
 * @param values Array of values
 * @param count Number of values
 * @param mean Pre-computed mean
 * @param variance Pre-computed variance
 * @param lag Number of samples to lag (default: 1)
 * @return Autocorrelation coefficient (-1.0 to 1.0)
 */
inline float calc_autocorrelation(const float* values, uint16_t count, float mean, float variance, uint16_t lag = 1) {
    if (count < lag + 2 || variance < 1e-10f) return 0.0f;
    
    float autocovariance = 0.0f;
    for (uint16_t i = 0; i < count - lag; i++) {
        autocovariance += (values[i] - mean) * (values[i + lag] - mean);
    }
    autocovariance /= (count - lag);
    
    return autocovariance / variance;
}

/**
 * Calculate Median Absolute Deviation (MAD).
 * 
 * Robust measure of variability, less sensitive to outliers than std.
 * Uses insertion sort (efficient for small n, e.g. 50).
 * 
 * @param values Array of values
 * @param count Number of values
 * @return MAD value
 */
inline float calc_mad(const float* values, uint16_t count) {
    if (count < 2 || count > ML_MAX_SORT_SIZE) return 0.0f;
    
    // Copy for sorting (stack allocation, max 200 floats = 800 bytes)
    float sorted[ML_MAX_SORT_SIZE];
    for (uint16_t i = 0; i < count; i++) {
        sorted[i] = values[i];
    }
    
    // Calculate median using utils.h helper
    float median = calculate_median_float(sorted, count);
    
    // Calculate absolute deviations
    float abs_devs[ML_MAX_SORT_SIZE];
    for (uint16_t i = 0; i < count; i++) {
        abs_devs[i] = std::fabs(values[i] - median);
    }
    
    // Return median of absolute deviations
    return calculate_median_float(abs_devs, count);
}

/**
 * Calculate waveform length (sum of absolute first differences).
 *
 * Captures total trajectory variation and oscillation activity over time.
 *
 * @param values Array of values
 * @param count Number of values
 * @return Waveform length
 */
inline float calc_waveform_length(const float* values, uint16_t count) {
    if (count < 2 || values == nullptr) return 0.0f;

    float total = 0.0f;
    float prev = values[0];
    for (uint16_t i = 1; i < count; i++) {
        float curr = values[i];
        total += std::fabs(curr - prev);
        prev = curr;
    }
    return total;
}

/**
 * Extract all 9 ML features from turbulence buffer and amplitudes.
 * 
 * All 9 features are computed from the turbulence buffer (typically 100 samples).
 * 
 * @param turb_buffer Turbulence buffer
 * @param turb_count Number of valid values in turbulence buffer
 * @param amplitudes Subcarrier amplitudes (unused, kept for API compatibility)
 * @param amp_count Number of amplitude values
 * @param features_out Output array for 9 features (must be pre-allocated)
 */
inline void extract_ml_features(const float* turb_buffer, uint16_t turb_count,
                                const float* amplitudes, uint8_t amp_count,
                                float* features_out) {
    (void) amplitudes;
    (void) amp_count;
    // Initialize to zero
    for (uint8_t i = 0; i < ML_NUM_FEATURES; i++) {
        features_out[i] = 0.0f;
    }
    
    if (turb_count < 2) return;
    
    // Calculate turbulence statistics (single pass for sum, min, max)
    float turb_sum = 0.0f;
    float turb_min = turb_buffer[0];
    float turb_max = turb_buffer[0];
    
    for (uint16_t i = 0; i < turb_count; i++) {
        float val = turb_buffer[i];
        turb_sum += val;
        if (val < turb_min) turb_min = val;
        if (val > turb_max) turb_max = val;
    }
    
    float turb_mean = turb_sum / turb_count;
    
    // Calculate variance (second pass)
    float var_sum = 0.0f;
    for (uint16_t i = 0; i < turb_count; i++) {
        float diff = turb_buffer[i] - turb_mean;
        var_sum += diff * diff;
    }
    float turb_var = var_sum / turb_count;
    float turb_std = std::sqrt(turb_var);
    
    // Interquartile range
    float turb_iqr = calc_iqr(turb_buffer, turb_count);
    
    // Skewness (pre-computed mean/std passed to avoid redundant calculation)
    float turb_skewness = calc_skewness(turb_buffer, turb_count, turb_mean, turb_std);
    
    // Lag-1 autocorrelation
    float turb_autocorr = calc_autocorrelation(turb_buffer, turb_count, turb_mean, turb_var, 1);
    
    // Median absolute deviation
    float turb_mad = calc_mad(turb_buffer, turb_count);
    
    // Temporal variation feature
    float waveform_length = calc_waveform_length(turb_buffer, turb_count);
    
    // Fill output array in correct order (matches Python DEFAULT_FEATURES)
    features_out[0] = turb_mean;       // 0
    features_out[1] = turb_std;        // 1
    features_out[2] = turb_max;        // 2
    features_out[3] = turb_min;        // 3
    features_out[4] = turb_iqr;        // 4
    features_out[5] = turb_skewness;   // 5
    features_out[6] = turb_autocorr;   // 6
    features_out[7] = turb_mad;        // 7
    features_out[8] = waveform_length; // 8
}

}  // namespace espectre
}  // namespace esphome
