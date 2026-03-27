# Loan Risk Prediction: Production-Ready MLOps Pipeline

This repository implements a modular, scalable machine learning pipeline for loan risk prediction, adhering to modern MLOps principles.

## 🚀 Key Features
- **Data Versioning**: Integrated with **DVC** to ensure full reproducibility and data lineage.
- **Modular Architecture**: Decoupled components for data ingestion, validation, feature encoding, and model training.
- **Config-Driven**: Centralized hyperparameters and path management via `config/params.yaml`.
- **Validation Layers**: Built-in data validation for both raw and processed datasets to ensure high-quality training.
- **Multi-Model Support**: Pre-configured support for various classifiers including RandomForest, SVC, and Decision Trees.

## 🛠️ Project Structure
- `src/data`: Data loading and validation logic.
- `src/features`: Category encoding and feature engineering.
- `src/models`: Data preparation, model training, and building.
- `src/pipelines`: Orchestration of the preprocessing and training workflows.
- `config/`: Configuration files for reproducibility.

## ⚙️ Quick Start
1. **Initialize DVC**: `dvc init`
2. **Reproduce Pipeline**: `dvc repro`
3. **Visualize DAG**: `dvc dag`