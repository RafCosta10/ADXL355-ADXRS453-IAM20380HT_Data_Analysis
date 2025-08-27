# ADXL355-ADXRS453-IAM20380HT_Data_Analysis

Data analysis tools and workflows for position sensors:
- **ADXL355** (3-axis accelerometer)  
- **ADXRS453** (SPI gyroscope)  
- **IAM20380HT** (I2C gyroscope)  

This repository provides Python utilities for loading, processing, analyzing, and visualizing raw binary sensor data. The code is designed to evaluate sensor performance metrics such as sampling rate, drift, noise, dropouts, and stability.

---

## Features

- **Data Loading**
  - Parse binary `.bin` files from accelerometers and gyroscopes.
  - Automatic record format detection (old chunked and new single-file formats supported).
  - Timestamp normalization and time-indexed pandas DataFrames.

- **Preprocessing**
  - Mean-centering of signals.
  - High-pass filtering to remove DC drift.
  - Despiking (Hampel filter) for outlier removal.
  - Dropout detection and reporting.

- **Integration & Drift Analysis**
  - Successive trapezoidal integration (acceleration → velocity → displacement).
  - 1-second windowed displacement drift analysis.
  - Long-term drift evaluation with high-pass bias suppression.
  - Calm-segment detection for controlled analysis.

- **Visualization**
  - Sampling rate histograms per sensor.
  - Complete sensor data plots (accelerometers and gyroscopes).
  - Temperature trend plots for I2C gyroscope.
  - Drift histograms and time series.

- **Reporting**
  - Outlier and dropout reports saved as text files.
  - Statistical summaries (mean, std, percentiles, worst-case).

---

## Repository Structure

ADXL355-ADXRS453-IAM20380HT_Data_Analysis/
├── data_analysis.py # Core analysis functions and utilities
└──README.md # Project documentation

---

## Requirements

- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
