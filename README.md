# Multimodal Emotion Recognition (EEG + Video)

This repository contains a Multimodal Emotion Recognition system that utilizes Electroencephalography (EEG) signals and Video processing to predict the presence of negative emotion. 

## Overview
The application uses a Logistic Regression model trained on fused multimodal features:
- **EEG Features**: Alpha, Beta, Theta means and correlation variances.
- **Video Features**: Mouth motion, eye motion, and head motion variances.

Based on these combined parameters, the model predicts a binary label: **1 (Negative Emotion)** or **0 (Non-Negative)**.

## Project Structure
- `data/`: Contains raw and processed feature files.
- `models/`: Destination for saved `.joblib` model artifacts.
- `notebooks/`: Jupyter Notebooks (e.g., `Eav_ML.ipynb` containing early experimentation).
- `src/`: Core Python source code.
  - `export_model.py`: Script to train and export the `model.joblib` artifact.

## Setup and Running
1. Install requirements using `pip install -r requirements.txt`. (Soon to be added)
2. Run `python src/export_model.py` to train and export the fallback model locally.
3. Start the Streamlit frontend with `streamlit run app.py` (Soon to be added).
