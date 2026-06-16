# SOD-FE: Supervised Outlier Detection and Feature Engineering

[![Published](https://img.shields.io/badge/Published-Article-green)](https://doi.org/10.1038/s41598-026-56591-6)

This repository contains the official implementation of the paper: **"SOD-FE: A Supervised Outlier Detection and Feature Engineering Approach for Student Dropout Prediction"**.

📄 **Read the published paper:** [Click here to view the article](https://doi.org/10.1038/s41598-026-56591-6)

## Overview
Student dropout prediction is critical for early intervention in higher education. This repository provides the custom code for the SOD-FE approach, which combines label-aware outlier detection (IQR-based with mutual information and Pearson correlation) with feature engineering to enhance machine learning dropout prediction models.

The methodology has been tested on two real-world datasets (Portugal and Slovakia).

## Repository Structure
- `src/sodfe.py`: Contains the core `SODFE` class, including data preprocessing, EDA, the custom outlier detection algorithm, feature engineering (RFI, SHAP, RFE), and model evaluation methods.
- `main.py`: The execution script demonstrating how to load data, apply the SOD-FE methodology, and train/evaluate the classifiers (RF, XGB, LR, SVM, MLP).
- `requirements.txt`: List of dependencies required to run the code.

## Installation
To run this code, ensure you have Python 3.9+ installed. Clone this repository and install the dependencies:

```bash
git clone [https://github.com/AlzabidiEissa/SOD-FE_Student_Dropout.git](https://github.com/AlzabidiEissa/SOD-FE_Student_Dropout.git)
cd SOD-FE_Student_Dropout
pip install -r requirements.txt
