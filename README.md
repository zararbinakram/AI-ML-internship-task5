# AI-ML-internship-task5
Objective

The goal is to predict housing prices by combining structured features (like area, bedrooms) with unstructured image data (photos of houses). This shows the strength of multimodal machine learning.


---

Methodology / Approach

1. Tabular data → passed through a small feed-forward neural network.


2. Images → features extracted using a pretrained ResNet50 CNN (frozen layers, acts as a feature extractor).


3. Fusion → concatenate image features (2048-D) with tabular features, then pass through dense layers for regression.


4. Training → used Adam optimizer, MSE loss, trained for 5 epochs.


5. Evaluation → measured performance using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).




---

Key Results

The combined model can learn both visual cues (house design, condition) and numerical features (size, bedrooms).

Predictions are more accurate than using only images or only tabular data.

On sample runs, the model achieves MAE in tens of thousands range (depends on dataset quality).
