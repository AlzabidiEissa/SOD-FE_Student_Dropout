# =====================================================
# SOD-FE: Supervised Outlier Detection and Feature Engineering
# =====================================================

import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba
import shap
from sklearn.decomposition import PCA
# Statistics & Data Preprocessing
from sklearn.preprocessing import (StandardScaler, MinMaxScaler)
# Data Splitting & Model Selection
from sklearn.model_selection import (train_test_split, StratifiedKFold)
# Feature Selection
from sklearn.feature_selection import (mutual_info_classif, RFE)
from sklearn.inspection import permutation_importance
# Metrics & Evaluation
from sklearn.metrics import (precision_score, recall_score, accuracy_score, 
                             f1_score, roc_auc_score, roc_curve)
# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

class SODFE:
    
    def __init__(self, data_path, random_state=0, target_col='Target'):
        self.data_path = data_path
        self.df = None
        self.save_path = os.path.join(os.getcwd(), "outputs")
        pd.options.display.max_columns = 500
        
        # Evaluate initial model performance by K-fold Cross-validation
        self.K = 5
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        self.best_model = RandomForestClassifier(random_state=self.random_state)
        self.best_name = " RF"
        self.target_col = target_col
        
        self.results = []
        self.best_sc = 0
        self.best_features = None
        
        self.selected_features = None
        self.target_class = None
        self.subgroup_class = None
        self.models = None

        self.X = None
        self.y = None
              
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.num_non_dropout = None
        self.num_dropout = None
        self.results_df = pd.DataFrame(columns=["Target_Class", "Subgroup_Class", "num_Selected_Features", "Slope", "F1_Score", "num_non_dropout", "num_dropout", "Removiated_data_ratio"])

        self.metrics_results = {}
        self.metrics_results1 = {}
        self.metrics_results2 = {}
        self.y_pred_results = {}
        
    # =====================================================
    # Step 1: Load & Prepare Data
    # =====================================================
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop_duplicates()
        return self.df

    def preprocess_data(self):
        ## Data Cleaning
        self.df.rename(columns = {"Nacionality": "Nationality", 
                                "Mother's qualification": "Mother_qualification", 
                                "Father's qualification": "Father_qualification", 
                                "Mother's occupation": "Mother_occupation",
                                "Father's occupation": "Father_occupation", 
                                "Age at enrollment": "Age"}, inplace = True)

        self.df.columns = self.df.columns.str.replace(' ', '_')
        self.df.columns = self.df.columns.str.replace('(', '').str.replace(')', '')

        self.df.drop(self.df[self.df['Target'] == 'Enrolled'].index, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df['Target'] = self.df['Target'].map({'Graduate': 0, 'Dropout': 1})
        
        # cleaning and handling of unrelated features
        delete_features = ['Educational_special_needs', 'International', 'Nationality']
        self.df = self.df.drop(columns=delete_features, errors='ignore')
        
        return self.df

    # =====================================================
    # Step 2: Visualization - Exploratory Data Analysis (EDA)
    # =====================================================
    def plot_pie(self, target_col='Target', figsize=(8, 8), name="class_pie.png", df= None):
        
        if df is None: df = self.df
        
        plt.figure(figsize=figsize)
        plt.title("Students Dropout", fontsize=18)
        
        if sum(df["Target"] == 0) > sum(df["Target"] == 1):
            lbl = ['Non-dropout', 'Dropout']
            colors=['green', 'red']
        else:
            lbl = ['Dropout', 'Non-Dropout']
            colors=['red', 'green']

        wedges, texts, autotexts = plt.pie(df[target_col].value_counts(), labels=lbl, colors=colors, explode=(0.1, 0.0), autopct='%1.2f%%', shadow=True, textprops={'fontsize': 16})
        plt.legend(wedges, lbl, loc='upper right', fontsize=12)
        
        for text in texts: text.set_fontsize(16)
        
        if self.save_path:
            full_path = os.path.join(self.save_path, name)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
                    
        plt.show()

    def plot_pca(self, target_col='Target', figsize=(10, 8), name="2d_PCA.png", df= None):
        
        if df is None: df = self.df
            
        features = df.drop(columns=[target_col])
        numeric_features = features.select_dtypes(include='number')
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        target = df[target_col]

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)

        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df[target_col] = target.values
        pca_df[target_col] = pca_df[target_col].map({0: 'Non-Dropout', 1: 'Dropout'})

        sns.set(style='whitegrid', font_scale=1.4)
        palette = sns.color_palette("Set1", n_colors=len(pca_df[target_col].unique()))

        plt.figure(figsize=figsize, dpi=300) 
        scatter = sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=target_col, palette=palette, s=80, edgecolor='k')

        plt.title('PCA Projection (2D)', fontsize=16, weight='bold')
        plt.grid(False)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.legend(title=target_col, title_fontsize=12, fontsize=11, loc='best', frameon=True)
        plt.tight_layout()
        
        if self.save_path:
            full_path = os.path.join(self.save_path, name)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
                    
        plt.show()

    def plot_dist(self, target_col='Target', figsize=(25, 30), name="data_dist.png", df=None):
        
        if df is None: df = self.df.copy()
            
        df[target_col] = df[target_col].map({0:'Non-dropout', 1:'Dropout'})
        
        nrows = len(df.columns) // 5 + (len(df.columns) % 5 > 0)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if target_col in numeric_cols: numeric_cols.remove(target_col)        
        
        ncols = 5
        y = nrows * 5
        figsize = (25, y)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()
        custom_palette = {'Non-dropout': 'green', 'Dropout': 'red'}

        for i, col in enumerate(numeric_cols):
            ax = axs[i] if i < len(df.columns) else axs[-1]
            sns.histplot(data=df, x=col, hue=target_col, palette=custom_palette, ax=ax, stat="density", element="step", common_norm=False)
            for class_label in df[target_col].unique():
                subset = df[df[target_col] == class_label]
                if subset[col].nunique() > 1: 
                    sns.kdeplot(data=subset, x=col, ax=ax, color=custom_palette[class_label], fill=True, label=f"{class_label} (KDE)")

            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.grid(False)
        
        for ax in axs[len(numeric_cols):]: ax.remove()
        
        plt.tight_layout()
        
        if self.save_path:
            full_path = os.path.join(self.save_path, name)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
                    
        plt.show()    
        
    def plot_correlation_heatmap(self, target_col='Target', name='Correlation_Heatmap.png', df= None):
        
        if df is None: df = self.df
            
        corr_matrix = df.corr(method="pearson")
        corr_matrix['Target'] = abs(corr_matrix[target_col])
        corr_matrix = corr_matrix.sort_values(by='Target', ascending=False)

        groups = {
            '>=0.4': corr_matrix[corr_matrix['Target'] >= 0.4].index.tolist()[1:],
            '>=0.2': corr_matrix[(corr_matrix['Target'] >= 0.2) & (corr_matrix['Target'] < 0.4)].index.tolist(),
            '>=0.1': corr_matrix[(corr_matrix['Target'] >= 0.1) & (corr_matrix['Target'] < 0.2)].index.tolist(),
            '<0.1': corr_matrix[corr_matrix['Target'] < 0.1].index.tolist(),
            'Target': ['Target']
        }

        group_colors = {'>=0.4': 'blue', '>=0.2': 'green', '>=0.1': 'purple', '<0.1': 'brown', 'Target': 'red'}
        ordered_columns = [col for group in groups.values() for col in group]
        df_selected = df[ordered_columns]
        corr = df_selected.corr(method="pearson")

        n = len(ordered_columns)
        color_matrix = np.zeros((n, n, 4))
        col_to_group = {col: grp for grp, cols in groups.items() for col in cols}

        for i in range(n):
            for j in range(n):
                col_i = ordered_columns[i]
                col_j = ordered_columns[j]
                group_i = col_to_group.get(col_i)
                group_j = col_to_group.get(col_j)

                if group_i == group_j or 'Target' in group_i:
                    base_color = group_colors[group_i]
                else:
                    if 'Target' in group_j:
                        base_color = group_colors['Target']
                    else:
                        base_color = 'gray'

                alpha = abs(corr.iloc[i, j])
                rgba = to_rgba(base_color, alpha=alpha)
                color_matrix[i, j] = rgba

        fig, ax = plt.subplots(figsize=(20, 15))
        ax.imshow(color_matrix, aspect='equal')
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(ordered_columns, rotation=90)
        ax.set_yticklabels(ordered_columns)
        ax.grid(False)
        plt.title('Correlation Heatmap', fontsize=20, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        
        if self.save_path:
            full_path = os.path.join(self.save_path, name)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
                    
        plt.show()

    def statistics_summary(self, df= None):
        
        if df is None: df = self.df
        
        summary = df.describe().T 
        formatted_summary = summary.style.format("{:.2f}")
        
        return formatted_summary 
         
    def plot_roc_curves(self, name='ROC.png'):
        """
        Plots ROC curves for all models.
        """
        colors = ['#003366', '#006600', '#990000', 
                '#660066', '#006666', '#333333', '#800000']

        plt.figure(figsize=(10, 8))
        
        for idx, (model_name, y_pred_prob) in enumerate(self.y_pred_results.items()):
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
            roc_auc = "{:.3f}".format(float(self.metrics_results[model_name]['roc_auc'].split("±")[0])) 
            
            plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {roc_auc})')

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)  # Random guess line

        plt.xlim([-0.01, 1.01])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate', fontsize=15, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=15, weight='bold')
        plt.title('Comparison of ROC Curves', fontsize=25, weight='bold')
        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()
        plt.grid(False)
        
        # Save if a path is provided
        if self.save_path:
            full_path = os.path.join(self.save_path, name)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
    
        plt.show()    
    
    # =====================================================
    # Step 3: Outlier Removal & Optimization
    # =====================================================
    def outliers_by_sof(self, df):
        
        df_cleaned = df.copy()
        mask = pd.Series(True, index=df_cleaned.index)

        if self.target_class == 2:
            class_df = df_cleaned
        else:
            class_df = df_cleaned[df_cleaned[self.target_col] == self.target_class]
        
        Q1 = class_df[self.selected_features].quantile(0.25)
        Q3 = class_df[self.selected_features].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR        
        
        current_mask = ((df_cleaned[self.selected_features] >= lower_bound) & 
                (df_cleaned[self.selected_features] <= upper_bound)).all(axis=1)
        
        if self.subgroup_class == 2:
            mask = current_mask
        else:
            label_mask = (df_cleaned[self.target_col] == self.subgroup_class)
            mask = current_mask | label_mask 
        
        df_cleaned = df_cleaned[mask]
        df_outliers = df[~mask]
        outliers_dict = df_outliers['Target'].value_counts().to_dict()
        self.num_non_dropout = outliers_dict[0] if 0 in outliers_dict.keys() else 0
        self.num_dropout = outliers_dict[1] if 1 in outliers_dict.keys() else 0

        return df_cleaned, df_outliers, mask

    def evaluate_model_outliers(self, model, X, y, test_size=0.2):
        
        scores_kfold_holdout = {key: [] for key in ["accuracy", "f1"]}
        
        for train_idx, val_idx in self.cv.split(X, y):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_val_pred = model.predict(X_fold_val)
            
            acc_val = accuracy_score(y_fold_val, y_val_pred)
            f1_val = f1_score(y_fold_val, y_val_pred, average='weighted')
            scores_kfold_holdout['accuracy'].append(acc_val) 
            scores_kfold_holdout['f1'].append(f1_val) 

        score = np.mean([np.mean(scores_kfold_holdout["accuracy"]), np.mean(scores_kfold_holdout["f1"])])  
        
        return score
  
    def optimize_outlier_feature_selection(self, method="mutual_info", name="best_feature_selection"):
        
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != self.target_col]
        X_temp, y_temp= self.df[numeric_cols], self.df[self.target_col] 
        cols = X_temp.columns.values
        
        correlations = abs(X_temp[cols].corrwith(y_temp))
        thresholds = correlations.sort_values(ascending=False).values
        mi_scores = mutual_info_classif(X_temp, y_temp, discrete_features='auto', random_state=0)
        mi_series = pd.Series(mi_scores, index=cols).sort_values(ascending=False)

        rank_corr = correlations.rank(ascending=False)
        rank_mi = mi_series.rank(ascending=False)
        combined_rank = ((rank_corr + rank_mi) / 2).dropna().sort_values()

        model = self.best_model
        original_size = len(self.df)
        best_score = -np.inf

        def process_combination(target_class, subgroup_class, idx, th):
            
            self.target_class = target_class
            self.subgroup_class = subgroup_class

            if method == "correlation":
                self.selected_features = correlations[correlations >= th].index.tolist()
            elif method == "mutual_info":
                self.selected_features = mi_series.head(idx+1).index.tolist()
            elif method == "combined":
                self.selected_features = combined_rank.head(idx+1).index.tolist()
            else:
                raise ValueError("Invalid method. Choose 'correlation', 'mutual_info', or 'combined'.")

            if not self.selected_features: return None
            
            df_filtered, _, _ = self.outliers_by_sof(self.df)
            X = df_filtered.drop(columns=self.target_col)
            y = df_filtered[self.target_col]
            removal_size = self.num_non_dropout + self.num_dropout
            removal_rate = 100 * removal_size / original_size
            mean_score = self.evaluate_model_outliers(model, X, y)

            row = {
                "Target_Class": target_class, "Subgroup_Class": subgroup_class,
                "num_Selected_Features": len(self.selected_features), "F1_Score": f"{100 * mean_score:.2f}",
                "num_non_dropout": self.num_non_dropout, "num_dropout": self.num_dropout, "Removiated_data_ratio": f"{removal_rate:.2f}%"
            }
            
            return mean_score, list(self.selected_features), target_class, subgroup_class, removal_size, row

        results = Parallel(n_jobs=-1)(delayed(process_combination)(tc, sgc, idx, th) for tc in [0] for sgc in [1] for idx, th in enumerate(thresholds))

        f1_scores = [] 
        data_sizes = []
        alfa = 0
        beta = 0
        f1 = 0
        
        for i, res in enumerate(results):
            
            if i%len(thresholds) == 0 and i !=0:
                f1_scores = [] 
                data_sizes = []
                alfa = 0
                beta = 0
                f1 = 0
                
            if res is None:
                continue
            mean_score, features, tc, sgc, removal_size, row = res
            
            if mean_score > f1:
                alfa = sgc
                beta = len(features)
                f1 = mean_score
            
            f1_scores.append(100 * mean_score)
            removal_rate = 100 * removal_size / original_size
            data_sizes.append(removal_rate)
            
            self.results_df = pd.concat([self.results_df, pd.DataFrame([row])], ignore_index=True)
            if mean_score > best_score:
                best_score = mean_score
                self.selected_features = features
                self.target_class = tc
                self.subgroup_class = sgc
        
    # =====================================================
    # Step 4: Feature Engineering
    # =====================================================                       
    def scale_features(self, X, scaler_type='standard'):
        
        X = X.copy()
        clmns = X.columns.values
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=clmns, index=X.index)
        return X
    
    def feature_importance_shap_cv(self, model, model_name, n_splits=5, test_size=0.2):
        
        feature_importances = np.zeros(self.X.shape[1])
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size = 0.2, random_state=self.random_state)
        model = model.fit(X_train, y_train)
                    
        if model_name in ["SVM", "MLP"]:
            result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=self.random_state, n_jobs=-1)
            feature_importance = result.importances_mean  
        else:
            if "LR" in model_name:
                masker = shap.maskers.Independent(X_train)
                explainer = shap.LinearExplainer(model, masker)
            else:
                explainer = shap.TreeExplainer(model)
            explainer.check_additivity = False
            shap_values = explainer(X_val)
            if len(shap_values.shape)>2:
                feature_importance = np.abs(shap_values[..., 1].values).mean(axis=0)                 
            else:
                feature_importance = np.abs(shap_values.values).mean(axis=0)

        feature_importances = feature_importance
        importance_shap = pd.DataFrame({'Feature': self.X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
        return importance_shap
    
    def feature_importance_rf_cv(self, model, model_name, n_splits=5, test_size=0.2):
        
        feature_importances = np.zeros(self.X.shape[1])
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size = 0.2, random_state=self.random_state)
        model = model.fit(X_train, y_train)
                    
        if model_name in ['SVM', ' LR']:
            feature_importance = np.abs(model.coef_[0])
        elif 'MLP' in model_name:
            result = permutation_importance(model, X_val, y_val, n_repeats=1, random_state=self.random_state, n_jobs=-1)
            feature_importance = result.importances_mean
        else:
            feature_importance = model.feature_importances_

        feature_importances = feature_importance
        importance_df = pd.DataFrame({'Feature': self.X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
        return importance_df

    def feature_importance_rfe(self, n_features_to_select):
        
        model = self.best_model
        rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=1).fit(self.X, self.y)
        selected_features = self.X.columns[rfe.support_]
        return selected_features

    def features_selected(self, name='rfi', n_jobs=-1):
        
        num_f = self.X.shape[1]
        metrics = self.evaluate_model_features_selection(self.best_model)
        score = np.mean([float(metrics['f1'].split("±")[0]), float(metrics['accuracy'].split("±")[0])])
        
        if score > self.best_sc:
            self.best_sc = score
            self.best_features = self.X.columns
            best_metrics = metrics
            self.results.append((self.best_sc, self.best_features, best_metrics, self.best_name))

        for model_name, model in self.models.items():
            if 'KNN' in model_name: continue
            if "rfi" in name:
                importance = self.feature_importance_rf_cv(model=model, model_name=model_name)
            elif "shap" in name:
                importance = self.feature_importance_shap_cv(model=model, model_name=model_name)

            for i in range(num_f - 10):
                if "rfe" in name:
                    n_features_to_select = num_f - i - 1
                    top_features = self.feature_importance_rfe(n_features_to_select)
                else:
                    top_features = importance['Feature'][:num_f - i].tolist()

                new_metrics = self.evaluate_model_features_selection(self.best_model, selected_features=top_features)
                new_score = np.mean([float(new_metrics['f1'].split("±")[0]), float(new_metrics['accuracy'].split("±")[0])])

                if new_score > self.best_sc:
                    self.best_sc = new_score
                    self.best_features = top_features
                    self.results.append((self.best_sc, self.best_features, new_metrics, model_name))
        
        g_best_sc, g_best_features, g_best_metrics, g_best_name = max(self.results, key=lambda x: x[0])
        
        return self.best_features

    # =====================================================
    # Step 6: Model Training
    # =====================================================
    def split_data(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        
    def models_bulding(self):
        self.models = {
            ' LR': LogisticRegression(random_state=self.random_state),
            'SVM': SVC(kernel='linear', random_state=self.random_state, probability=True),
            ' RF': RandomForestClassifier(random_state=self.random_state),
            'XGB': XGBClassifier(random_state=self.random_state),
            'MLP': MLPClassifier(random_state=self.random_state, early_stopping=True, max_iter=100)
            }
        
    def train_and_evaluate_all_models(self, evaluation_method='holdout', n_splits=5):
        self.models_bulding()
        if evaluation_method not in ['holdout', 'kfold']: raise ValueError("Invalid evaluation_method")
        
        for name, model in self.models.items():
            if evaluation_method == 'holdout':
                metrics, y_propa = self.evaluate_model(model, show_cm=False)
                self.metrics_results[name] = metrics
                self.y_pred_results[name] = y_propa
                print(f"{name} Metrics: {metrics}")
                
            elif evaluation_method == 'kfold':
                metrics = self.evaluate_model_kfold(model)
                self.metrics_results[name] = metrics
                print(f"{name} Metrics: {metrics}")
                
            
        best_model_name = max(self.metrics_results, key=lambda k: self.metrics_results[k]['f1'].split("±")[0])
        self.best_model = self.models[best_model_name]
        self.best_name = best_model_name
        return best_model_name, self.best_model, self.metrics_results

    # =====================================================
    # Step 7: Model Evaluation
    # =====================================================
    def evaluate_model(self, model, show_cm=False):
        model.fit(self.X_train, self.y_train)
        y_prob = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)
        return {
            'accuracy': f"{accuracy_score(self.y_test, y_pred)*100:.2f}",
            'precision': f"{precision_score(self.y_test, y_pred, average='weighted')*100:.2f}",
            'recall': f"{recall_score(self.y_test, y_pred, average='weighted')*100:.2f}",
            'f1': f"{f1_score(self.y_test, y_pred, average='weighted')*100:.2f}",
            "roc_auc": f"{roc_auc_score(self.y_test, y_prob)*100:.2f}" 
        }, y_prob 
        
    def evaluate_model_features_selection(self, model, selected_features = None):
        X = self.X if selected_features is None else self.X[selected_features]
        scores_kfold_holdout = {key: [] for key in ["accuracy", "f1"]}
        metrics = {}
        y = self.y
        
        for train_idx, val_idx in self.cv.split(X, y):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_fold_train, y_fold_train)
            y_val_pred = model.predict(X_fold_val)
            scores_kfold_holdout['accuracy'].append(accuracy_score(y_fold_val, y_val_pred)) 
            scores_kfold_holdout['f1'].append(f1_score(y_fold_val, y_val_pred, average='weighted')) 
       
        for metric in ["accuracy", "f1"]:
            metrics[metric] = f"{np.mean(scores_kfold_holdout[metric])*100:.2f} ± {np.std(scores_kfold_holdout[metric])*100:.1f} %"        
        return metrics        
    
    def evaluate_model_kfold(self, model, selected_features = None, test_size=0.2):
        X = self.X if selected_features is None else self.X[selected_features]
        scores_kfold_holdout = {key: [] for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
        metrics = {}
        y = self.y
        
        for train_idx, val_idx in self.cv.split(X, y):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_fold_train, y_fold_train)
            y_val_proba = model.predict_proba(X_fold_val)[:, 1]
            y_val_pred = model.predict(X_fold_val)
            
            scores_kfold_holdout['accuracy'].append(accuracy_score(y_fold_val, y_val_pred)) 
            scores_kfold_holdout['precision'].append(precision_score(y_fold_val, y_val_pred, average='weighted')) 
            scores_kfold_holdout['recall'].append(recall_score(y_fold_val, y_val_pred, average='weighted')) 
            scores_kfold_holdout['f1'].append(f1_score(y_fold_val, y_val_pred, average='weighted')) 
            scores_kfold_holdout['roc_auc'].append(roc_auc_score(y_fold_val, y_val_proba)) 
       
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            metrics[metric] = f"{np.mean(scores_kfold_holdout[metric])*100:.2f} ± {np.std(scores_kfold_holdout[metric])*100:.1f} %"        
        return metrics
            
    # =====================================================
    # Final Step: Reporting
    # =====================================================
    def compare_models(self, metrics_results1, metrics_results2):
        rows = []
        for model_name in metrics_results1:
            if model_name in metrics_results2:
                metrics1 = metrics_results1[model_name]
                metrics2 = metrics_results2[model_name]
                rows.append({'Model': model_name, 'DataName': 'Por_Dataset', 'Accuracy': metrics1['accuracy'], 'F1 Score': metrics1['f1'], 'Recall': metrics1['recall'], 'AUC': metrics1['roc_auc']})
                rows.append({'Model': model_name, 'DataName': 'Sol_Dataset', 'Accuracy': metrics2['accuracy'], 'F1 Score': metrics2['f1'], 'Recall': metrics2['recall'], 'AUC': metrics2['roc_auc']})
        comparison = pd.DataFrame(rows)
        return comparison