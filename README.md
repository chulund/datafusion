# Multi-Sensor Data Fusion for Wildfire Detection

This repository provides MATLAB implementations of various data fusion methods for background temperature estimation in wildfire detection applications. The primary script, `datafusion_KF_fireRobust_weighted.m`, combines a fire-robust Kalman filter with adaptive per-pixel weighting to fuse mid-infrared (MIR) data from multiple satellite sources.

The methods are designed to handle fire-pixel exclusion and spatial/spectral/temporal weighting for robust background estimation.

## Overview

This code supports the following fusion methods (to be included progressively):

- Kalman Filter (KF)
- Ensemble Kalman Filter (EnKF)
- Particle Filter (SIR)
- Weak Constraint Four-Dimensional Variational Assimilation (4DVar)

Implemented sensors:

- Himawari-9 AHI (Japan Meteorological Agency / Bureau of Meteorology)
- GeoKompsat-2A AMI (Korea Meteorological Administration)
- Sentinel-3 SLSTR (bands S7 and F1, from both S3A and S3B)

All data used in this research are publicly available (see below). This repository only includes the fusion scripts; no raw data are provided.

## Data Sources

The following public datasets were used:

- Himawari-9 AHI:
  - Bureau of Meteorology Satellite Observations (2021): [https://doi.org/10.25914/61A609F9E7FFA](https://doi.org/10.25914/61A609F9E7FFA)
  - Bureau of Meteorology Satellite Derived Products (2022): [https://dx.doi.org/10.25914/5QRS-QB54](https://dx.doi.org/10.25914/5QRS-QB54)

- Sentinel-3 SLSTR:
  - Accessed on 1 January 2025 from [https://registry.opendata.aws/sentinel-3](https://registry.opendata.aws/sentinel-3)

- GeoKompsat-2A AMI:
  - Korea Meteorological Administration Data Service Center: [https://datasvc.nmsc.kma.go.kr/datasvc/html/data/listData.do](https://datasvc.nmsc.kma.go.kr/datasvc/html/data/listData.do)

All observations are level-1B radiances converted to brightness temperatures (BT) using sensor-specific calibrations. Cloud-masked and saturated pixels are excluded.

## Usage

### Example

```matlab
% Load your KalmanInput structure with sensor data here
load('KalmanInput_example.mat');

% Run the Kalman filter fusion
xhat = datafusion_KF_fireRobust_weighted(KalmanInput, 'variance_base', 5.8, 'deltaT_day', 85);
```
