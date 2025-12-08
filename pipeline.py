import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)


# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def load_data(filepath):
    """
    Load the hotel bookings dataset from CSV.
    
    Parameters:
        filepath (str): Path to hotel_bookings.csv
    
    Returns:
        pd.DataFrame: Hotel bookings dataset with 119,390 rows and 32 columns
    """
    return pd.read_csv(filepath)


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Create features that enhance predictive power
    
    Parameters:
        df (pd.DataFrame): Hptel bookings dataset
    
    Returns:
        df (pd.DataFrame): Hotel bookings dataset with new engineered features
    
    """
    df = df.copy()
    
    # Handle missing values
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    
    # Create is_family
    df['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)
    
    # Create duration_of_stay
    df['duration_of_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    
    # Create season_of_booking
    season_mapping = {
        'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
        'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
        'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
        'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn'
    }
    df['season_of_booking'] = df['arrival_date_month'].map(season_mapping)
    
    # Create room_type_changed
    df['room_type_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)
    
    # Create had_previous_cancellation
    df['had_previous_cancellation'] = (df['previous_cancellations'] > 0).astype(int)
    
    return df


# =============================================================================
# STEP 3: FEATURE SELECTION
# =============================================================================

def select_features(df):
    """
    Remove columns that shouldn't be used for prediction.
    (Select features for prediction)
    
    Parameters:
        df (pd.DataFrame): Hotel bookings dataset with engineered features
    
    Returns:
        df (pd.DataFrame): Hotel bookings dataset with selected columns
    """
    columns_to_drop = [
        'reservation_status', 'reservation_status_date',  # Leakage
        'arrival_date_year', 'arrival_date_month',        # Redundant
        'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights',
        'children', 'babies',
        'company', 'country', 'agent',                    # High cardinality
        'reserved_room_type', 'assigned_room_type'        # Redundant
    ]
    return df.drop(columns=columns_to_drop, errors='ignore')


# =============================================================================
# STEP 4: PREPARE DATA FOR MODELING
# =============================================================================

def prepare_data(df):
    """
    Separate features (X) from target (y) and identify column types.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataset
    
    Returns:
        X (pd.DataFrame): Feature matrix (all columns except is_canceled)
        y (pd.Series): Target vector (is_canceled: 0 or 1)
        numeric (list): Names of numeric columns for StandardScaler
        categorical (list): Names of categorical columns for OneHotEncoder
    """
    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    
    # Identify column types for preprocessing
    numeric = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return X, y, numeric, categorical


# =============================================================================
# STEP 5: CREATE PREPROCESSING PIPELINE
# =============================================================================

def create_preprocessor(numeric, categorical):
    """
    Build sklearn ColumnTransformer for preprocessing.
    
    Parameters:
        numeric (list): Numeric column names
        categorical (list): Categorical column names
    
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    preprocess = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])
    return preprocess


# =============================================================================
# STEP 6: BUILD MODELS
# =============================================================================

def build_logistic_regression(preprocessor):
    """
    Build Logistic Regression pipeline (Baseline Model).
    
    Parameters:
        preprocessor: ColumnTransformer from create_preprocessor()
    
    Returns:
        Pipeline: Complete preprocessing + Logistic Regression model
    """
    return Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])


def build_random_forest(preprocessor):
    """
    Build Random Forest pipeline (Strong Baseline Model).
    
    Parameters:
        preprocessor: ColumnTransformer from create_preprocessor()
    
    Returns:
        Pipeline: Complete preprocessing + Random Forest model
    """
    return Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42
        ))
    ])


# =============================================================================
# STEP 7: MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model and return performance metrics.
    
    Parameters:
        model: Trained sklearn Pipeline
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (canceled)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba
    }

# =============================================================================
# STEP 8: FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(model, numeric, categorical):
    """
    Extract feature importances from Random Forest.
    
    Parameters:
        model: Trained Random Forest Pipeline
        numeric: List of numeric feature names
        categorical: List of categorical feature names
    
    Returns:
        pd.DataFrame: Features ranked by importance
    
    How Random Forest Calculates Importance:
    ----------------------------------------
    - Based on "mean decrease in impurity" (Gini importance)
    - For each feature, measures how much it reduces uncertainty when used for splits
    - Higher importance = feature is more useful for predictions
    
    Why This Matters:
    -----------------
    - Identifies key drivers of cancellations
    - Guides business recommendations
    - Helps with feature selection for future models
    """
    # Get the trained Random Forest model from the pipeline
    rf_model = model.named_steps['model']
    
    # Get OneHotEncoder to retrieve encoded feature names
    ohe = model.named_steps['prep'].named_transformers_['cat']
    cat_features = ohe.get_feature_names_out(categorical)
    
    # Combine numeric and encoded categorical feature names
    all_features = list(numeric) + list(cat_features)
    
    # Get importance scores
    importances = rf_model.feature_importances_
    
    # Create sorted DataFrame
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df


# =============================================================================
# STEP 9: VISUALIZATIONS
# =============================================================================

def plot_confusion_matrices(y_test, results_lr, results_rf, save_path=None):
    """
    Plot side-by-side confusion matrices for both models.
    
    Confusion Matrix Interpretation:
    --------------------------------
    
                    Predicted
                    Not Cancel | Cancel
    Actual  Not Cancel    TN   |   FP
            Cancel        FN   |   TP
    
    - TN (True Negative): Correctly predicted NOT canceled
    - TP (True Positive): Correctly predicted canceled
    - FP (False Positive): Predicted cancel but didn't (false alarm)
    - FN (False Negative): Predicted not cancel but did (missed)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = ['Not Canceled', 'Canceled']
    
    for ax, results, title, cmap in zip(
        axes, [results_lr, results_rf],
        ['Logistic Regression', 'Random Forest'],
        ['Blues', 'Greens']
    ):
        cm = confusion_matrix(y_test, results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(f'{title}\nConfusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_test, results_lr, results_rf, save_path=None):
    """
    Plot ROC curves for model comparison.
    
    ROC Curve Interpretation:
    -------------------------
    - X-axis: False Positive Rate (1 - Specificity)
    - Y-axis: True Positive Rate (Sensitivity/Recall)
    - Each point represents a different probability threshold
    - Diagonal line = random guessing
    - Curve closer to top-left = better model
    - AUC summarizes overall performance (higher is better)
    """
    plt.figure(figsize=(10, 7))
    
    for results, label, color in zip(
        [results_lr, results_rf],
        ['Logistic Regression', 'Random Forest'],
        ['blue', 'green']
    ):
        fpr, tpr, _ = roc_curve(y_test, results['y_proba'])
        plt.plot(fpr, tpr, label=f'{label} (AUC = {results["roc_auc"]:.4f})', 
                 linewidth=2, color=color)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(importance_df, top_n=15, save_path=None):
    """Plot horizontal bar chart of top feature importances."""
    plt.figure(figsize=(12, 8))
    
    top_features = importance_df.head(top_n)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    plt.barh(range(len(top_features)), top_features['Importance'].values, color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def run_pipeline(filepath, test_size=0.2, random_state=42, save_plots=False, output_dir=''):
    """
    Execute the complete ML pipeline.
    
    Parameters:
        filepath (str): Path to hotel_bookings.csv
        test_size (float): Proportion for test set (default 0.2 = 20%)
        random_state (int): Random seed for reproducibility
        save_plots (bool): Whether to save visualization plots
        output_dir (str): Directory for saved outputs
    
    Returns:
        dict: Contains trained models, evaluation results, and feature importance
    """
    
    # -------------------------------------------------------------------------
    # DATA PREPARATION
    # -------------------------------------------------------------------------
    df = load_data(filepath)
    df = engineer_features(df)
    df = select_features(df)
    X, y, numeric, categorical = prepare_data(df)
    
    # -------------------------------------------------------------------------
    # TRAIN-TEST SPLIT
    # stratify=y ensures both sets have same class distribution (37% canceled)
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # -------------------------------------------------------------------------
    # BUILD AND TRAIN MODELS
    # -------------------------------------------------------------------------
    preprocessor = create_preprocessor(numeric, categorical)
    
    lr_model = build_logistic_regression(preprocessor)
    rf_model = build_random_forest(preprocessor)
    
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # EVALUATE MODELS
    # -------------------------------------------------------------------------
    results_lr = evaluate_model(lr_model, X_test, y_test)
    results_rf = evaluate_model(rf_model, X_test, y_test)
    
    # Cross-validation for more robust estimates
    lr_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='roc_auc')
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
    lr_mean = lr_scores.mean()
    rf_mean = rf_scores.mean()
    
    results_lr['cv_roc_auc'] = lr_mean
    results_rf['cv_roc_auc'] = rf_mean
    
    # -------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    importance_df = get_feature_importance(rf_model, numeric, categorical)
    
    # -------------------------------------------------------------------------
    # VISUALIZATIONS
    # -------------------------------------------------------------------------
    plot_paths = {
        'confusion': f'{output_dir}/confusion_matrices.png' if save_plots else None,
        'roc': f'{output_dir}/roc_curves.png' if save_plots else None,
        'importance': f'{output_dir}/feature_importance.png' if save_plots else None
    }
    
    plot_confusion_matrices(y_test, results_lr, results_rf, plot_paths['confusion'])
    plot_roc_curves(y_test, results_lr, results_rf, plot_paths['roc'])
    plot_feature_importance(importance_df, save_path=plot_paths['importance'])
    
    # -------------------------------------------------------------------------
    # CREATE COMPARISON SUMMARY
    # -------------------------------------------------------------------------
    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [results_lr['accuracy'], results_rf['accuracy']],
        'Precision': [results_lr['precision'], results_rf['precision']],
        'Recall': [results_lr['recall'], results_rf['recall']],
        'F1-Score': [results_lr['f1'], results_rf['f1']],
        'ROC-AUC': [results_lr['roc_auc'], results_rf['roc_auc']],
        'CV ROC-AUC': [results_lr['cv_roc_auc'], results_rf['cv_roc_auc']]
    })
    
    return {
        'models': {'logistic_regression': lr_model, 'random_forest': rf_model},
        'results': {'logistic_regression': results_lr, 'random_forest': results_rf},
        'comparison': comparison_df,
        'feature_importance': importance_df,
        'data': {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Run the complete pipeline
    results = run_pipeline(
        filepath='hotel_bookings.csv',
        save_plots=True,
        output_dir='/Users/abhi/Graduate Courses/DSE 501/HotelBookingsDatasetAnalysis'
    )
    
    # Print summary
    print()
    print("MODEL COMPARISON: ")
    print()
    print(results['comparison'].to_string(index=False))
    
    print()
    print("TOP 10 FEATURES: ")
    print()
    print(results['feature_importance'].head(10).to_string(index=False))
    
    print()
    print("CLASSIFICATION REPORTS: ")
    print("="*60)
    print("\nLogistic Regression:")
    print(classification_report(
        results['data']['y_test'], 
        results['results']['logistic_regression']['y_pred'],
        target_names=['Not Canceled', 'Canceled']
    ))
    
    print("\nRandom Forest:")
    print(classification_report(
        results['data']['y_test'], 
        results['results']['random_forest']['y_pred'],
        target_names=['Not Canceled', 'Canceled']
    ))