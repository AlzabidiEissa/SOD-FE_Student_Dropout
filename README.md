# SOD-FE: Supervised Outlier Detection and Feature Engineering

[![Preprint](https://img.shields.io/badge/Preprint-Research_Square-blue)](https://www.researchsquare.com/article/rs-6889300/v1)

This repository contains the official implementation of the paper: **"SOD-FE: A Supervised Outlier Detection and Feature Engineering Approach for Student Dropout Prediction"**.

📄 **Read the preprint on Research Square:** [Click here to view the article](https://www.researchsquare.com/article/rs-6889300/v1)

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
git clone https://github.com/AlzabidiEissa/SOD-FE_Student_Dropout.git
cd SOD-FE_Student_Dropout
pip install -r requirements.txt
```
## Data Availability
The datasets used in this study are publicly available:
1. **Portugal Dataset:** [Predict students' dropout and academic success (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
2. **Slovakia Dataset:** [Students Dropout Prediction (GitHub)](https://github.com/JKabathova/Students-Dropout-Prediction)

*Note: Please download the datasets and place them in the root directory (or update the file paths in `main.py`) before running the code.*

## Usage
To reproduce the experiments and evaluate the models, simply run the main script:

```bash
python main.py
```
This script will load the datasets, apply the SOD-FE preprocessing and feature selection steps, train the classifiers, and output the performance metrics (Accuracy, F1 Score, Recall, AUC).

## Citation
If you find this code or research helpful, please consider citing our preprint:

**APA Format:**
Sabah Saad, Önder Yakut, Eissa Alzabidi et al. SOD-FE: A Supervised Outlier Detection and Feature Engineering Approach for Student Dropout Prediction, 19 June 2025, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-6889300/v1]

**BibTeX:**
```bibtex
@article{saad2025sodfe,
  title={SOD-FE: A Supervised Outlier Detection and Feature Engineering Approach for Student Dropout Prediction},
  author={Saad, Sabah and Yakut, {\"O}nder and Alzabidi, Eissa and F{\i}nd{\i}k, O{\u{g}}uz},
  journal={Research Square},
  year={2025},
  note={PREPRINT (Version 1)},
  doi={10.21203/rs.3.rs-6889300/v1},
  url={[https://doi.org/10.21203/rs.3.rs-6889300/v1](https://doi.org/10.21203/rs.3.rs-6889300/v1)}
}
```

## License
This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).




