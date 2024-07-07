# Suspicious URL Detection Module

## Overview

The Suspicious URL Detection Module leverages machine learning techniques to identify potential phishing URLs. This module is based on the methodology introduced in the paper ["A new hybrid ensemble feature selection framework for machine learning-based phishing detection system"](https://doi.org/10.1016/j.ins.2019.01.064) by Kang Leng Chiew et al., 2019.

The dataset used in this module comprises 5,000 phishing webpages and 5,000 legitimate webpages. It includes 48 features extracted from webpage URLs and HTML source codes, covering aspects like the presence of suspicious characters, subdomain levels, URL lengths, and more.

## Dataset

The dataset used in this project is detailed in the aforementioned paper. The dataset (Phishing_Legitimate_full.csv) is downloaded from Kaggle (https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning/data).

## Models

Building on top of the pre-existing work, we trained both Random Forest and Neural Network (NN) models using this dataset. The performance metrics for both models are summarized below:

| Model          | Precision | Recall  | F1 Score | Accuracy |
|----------------|-----------|---------|----------|----------|
| Random Forest  | 0.983     | 0.983   | 0.983    | 0.983    |
| Neural Network | 0.956     | 0.957   | 0.957    | 0.957    |

The trained Random Forest model (`random_forest_model.pkl`) and Neural Network model (`model.pth`) are included in this repository. 

## References

- Kang Leng Chiew, et al. ["A new hybrid ensemble feature selection framework for machine learning-based phishing detection system."](https://doi.org/10.1016/j.ins.2019.01.064)
- [GitHub Repository for Scripts and Models](https://github.com/periscopegithub/Suspicious-URL-Detector.git)

