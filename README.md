# Clustering & Anomaly Detection on Olivetti Faces

## Overview

This project explores unsupervised learning techniques for clustering and anomaly detection using the Olivetti Faces dataset.

The project uses PCA for dimensionality reduction, hierarchical clustering and Gaussian Mixture Models (GMM) for clustering, and GMM log-likelihood scores for anomaly detection.

The project is currently being refactored from the original course assignment into a structured machine learning pipeline.

## Tech Stack

* Python
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn
* SciPy
* Scikit-image

## Current Pipeline

1. Load and visualize the Olivetti Faces dataset.
2. Split the dataset into training, validation, and test sets.
3. Flatten facial images and apply PCA while retaining 99% of the explained variance.
4. Apply hierarchical clustering.
5. Train and evaluate Gaussian Mixture Models.
6. Generate synthetic faces using the fitted GMM.
7. Detect anomalous facial images using GMM log-likelihood scores.

## Project Status

Currently refactoring and reorganizing the original implementation.
