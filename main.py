import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from src.sodfe import SODFE  

def run_sodfe_pipeline(dataset_name, data_path):
    """
    Function to run the complete SOD-FE pipeline for a given dataset.
    
    Parameters:
    - dataset_name (str): Name of the dataset (e.g., 'Portugal' or 'Slovakia').
    - data_path (str): Path to the dataset CSV file.
    
    Returns:
    - metrics_results (dict): The final evaluation metrics for the dataset.
    """
    print(f"\n{'='*50}")
    print(f"=== Starting SOD-FE Pipeline for {dataset_name} Dataset ===")
    print(f"{'='*50}")

    # 1. Initialize and Load Data
    print("\n[Step 1] Initializing and Loading Data...")
    pipeline = SODFE(data_path=data_path, random_state=0, target_col='Target')
    
    df_raw = pipeline.load_data()

    # 2. Data Preprocessing
    print("\n[Step 2] Preprocessing Data...")
    if "portugal" in data_path:
        df_processed = pipeline.preprocess_data()
    else:
        pipeline.df['Target'] = df_raw['Target'].map({0: 1, 1: 0})

    # Feature and Target Separation
    X_full = pipeline.df.drop(['Target'], axis=1)
    y_full = pipeline.df['Target']

    # Scale Features
    X_scaled = pipeline.scale_features(X_full)
    pipeline.X = X_scaled
    pipeline.y = y_full

    # Train/Test Split (80:20)
    pipeline.split_data(pipeline.X, pipeline.y, test_size=0.2)

    # Apply SMOTE to training data
    print("\n--- Applying SMOTE to address class imbalance ---")
    smote_sampler = SMOTE(random_state=0)
    pipeline.X_train, pipeline.y_train = smote_sampler.fit_resample(pipeline.X_train, pipeline.y_train)

    # 3. Baseline Model Evaluation (Before SOD-FE)
    print("\n[Step 3] Baseline Evaluation (Without SOD-FE)...")
    pipeline.models_bulding()
    _, _, baseline_metrics = pipeline.train_and_evaluate_all_models(evaluation_method='holdout')

    # 4. Supervised Outlier Detection (SOD)
    print("\n[Step 4] Applying SOD (Supervised Outlier Detection)...")
    
    # Merge train and test temporarily to apply SOD methodology
    df_train_temp = pd.concat([pipeline.X_train, pipeline.y_train], axis=1)
    df_test_temp = pd.concat([pipeline.X_test, pipeline.y_test], axis=1)
    pipeline.df = df_train_temp
    
    # Optimize and detect outliers
    pipeline.optimize_outlier_feature_selection(method="combined")
    
    df_merged_temp = pipeline.df.merge(df_test_temp, how='outer', indicator=True)
    df_merged_temp = df_merged_temp[df_merged_temp['_merge'] != 'both'].drop(columns=['_merge'])
    df_clean, df_outliers, outlier_mask = pipeline.outliers_by_sof(df_merged_temp)
    
    # Add 'outlier_score' feature (1 for normal, -1 for anomaly)
    anomaly_scores = outlier_mask.map({True: 1, False: -1})
    df_merged_temp['outlier_score'] = anomaly_scores
    
    # Update X and y with the cleaned data incorporating outlier scores
    pipeline.X = df_merged_temp.drop(['Target'], axis=1)
    pipeline.y = df_merged_temp['Target']
    pipeline.split_data(pipeline.X, pipeline.y, test_size=0.2)

    # 5. Feature Engineering (FE) - Selecting Top Features
    print("\n[Step 5] Feature Engineering (Selecting Top Features using RFI)...")
    best_features_rfi = pipeline.features_selected(name="rfi")
    print(f"--- Top Features Selected for {dataset_name}: {len(best_features_rfi)} features ---")
    
    # Filter dataset to keep only the best features
    pipeline.X = pipeline.X[best_features_rfi]
    pipeline.split_data(pipeline.X, pipeline.y, test_size=0.2)

    # 6. Final Model Evaluation (After SOD-FE)
    print("\n[Step 6] Final Evaluation (With SOD-FE) using K-Fold Cross Validation...")
    pipeline.models_bulding()
    best_model_name, best_model, final_metrics = pipeline.train_and_evaluate_all_models(
        evaluation_method='kfold')
    
    print(f"\n=== Pipeline Completed for {dataset_name}. Best Model: {best_model_name} ===")
    
    return pipeline, baseline_metrics, final_metrics

def main():
    # Define file paths (Update these paths to match your local setup)
    PORTUGAL_DATA_PATH = 'data/portugal_dataset.csv'
    SLOVAKIA_DATA_PATH = 'data/slovakia_dataset.csv'

    # 1. Run Pipeline for Portugal Dataset
    portugal_pipeline, pt_baseline, pt_final = run_sodfe_pipeline(
        dataset_name='Portugal',
        data_path=PORTUGAL_DATA_PATH,
    )
    
    # 2. Run Pipeline for Slovakia Dataset
    slovakia_pipeline, sk_baseline, sk_final = run_sodfe_pipeline(
        dataset_name='Slovakia',
        data_path=SLOVAKIA_DATA_PATH,
    )
    
    # 3. Compare Results
    print("\n" + "="*50)
    print("=== Final Comparison: Portugal vs Slovakia ===")
    print("="*50)
    
    # Using the existing compare_models function from your class
    comparison_df = portugal_pipeline.compare_models(pt_final, sk_final)
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()