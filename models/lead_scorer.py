"""
Lead Scoring Model Module
Handles model training, prediction, and evaluation
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

class LeadScoringModel:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_metrics = {}
        self.is_trained = False
        
        # Define categorical and numerical features
        self.categorical_features = ['lead_source', 'industry', 'company_size']
        self.numerical_features = ['age', 'page_views', 'time_on_site', 'email_opens', 
                                 'email_clicks', 'form_submissions', 'content_downloads', 
                                 'days_since_contact']
    
    def _create_preprocessor(self):
        """Create preprocessing pipeline"""
        # Preprocessing for numerical data
        numerical_transformer = StandardScaler()
        
        # Preprocessing for categorical data
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        # Bundle preprocessing for numerical and categorical data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
    
    def _create_model(self):
        """Create the machine learning model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_features(self, df):
        """Prepare features for training/prediction"""
        # Select only the features we need
        feature_columns = self.categorical_features + self.numerical_features
        return df[feature_columns].copy()
    
    def train(self, df, target_column='converted', test_size=0.2, perform_cv=True):
        """Train the lead scoring model"""
        print(f"Training {self.model_type} model...")
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df[target_column]
        
        # Create preprocessor and model
        self._create_preprocessor()
        base_model = self._create_model()
        
        # Create pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', base_model)
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.model_metrics = {
            'accuracy': self.model.score(X_test, y_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Perform cross-validation if requested
        if perform_cv:
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
            self.model_metrics['cv_mean'] = cv_scores.mean()
            self.model_metrics['cv_std'] = cv_scores.std()
        
        self.is_trained = True
        print(f"Model trained successfully!")
        print(f"Accuracy: {self.model_metrics['accuracy']:.3f}")
        print(f"ROC AUC: {self.model_metrics['roc_auc']:.3f}")
        
        if perform_cv:
            print(f"Cross-validation AUC: {self.model_metrics['cv_mean']:.3f} (+/- {self.model_metrics['cv_std']*2:.3f})")
        
        return self.model_metrics
    
    def predict_lead_score(self, lead_data):
        """Predict lead score for a single lead or multiple leads"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle single lead (dict) or multiple leads (DataFrame)
        if isinstance(lead_data, dict):
            df = pd.DataFrame([lead_data])
        else:
            df = lead_data.copy()
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Convert to lead scores (0-100)
        lead_scores = (probabilities * 100).astype(int)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        results = []
        for i, (score, pred, prob) in enumerate(zip(lead_scores, predictions, probabilities)):
            result = {
                'lead_score': int(score),
                'conversion_probability': float(prob),
                'predicted_conversion': bool(pred),
                'score_category': self._categorize_score(score)
            }
            results.append(result)
        
        return results[0] if isinstance(lead_data, dict) else results
    
    def _categorize_score(self, score):
        """Categorize lead score into buckets"""
        if score >= 80:
            return 'Hot'
        elif score >= 60:
            return 'Warm'
        elif score >= 40:
            return 'Cold'
        else:
            return 'Ice Cold'
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get the trained classifier from the pipeline
        classifier = self.model.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            # Get feature names after preprocessing
            preprocessor = self.model.named_steps['preprocessor']
            
            # Get feature names from the preprocessor
            feature_names = []
            
            # Numerical features
            feature_names.extend(self.numerical_features)
            
            # Categorical features (one-hot encoded)
            cat_transformer = preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'get_feature_names_out'):
                cat_features = cat_transformer.get_feature_names_out(self.categorical_features)
                feature_names.extend(cat_features)
            
            # Get importance scores
            importances = classifier.feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        else:
            return None
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data.get('feature_names')
        self.model_metrics = model_data.get('model_metrics', {})
        self.categorical_features = model_data.get('categorical_features', self.categorical_features)
        self.numerical_features = model_data.get('numerical_features', self.numerical_features)
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        return self.model_metrics
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        if 'confusion_matrix' not in self.model_metrics:
            raise ValueError("No confusion matrix available. Train the model first.")
        
        cm = np.array(self.model_metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Converted', 'Converted'], 
                   yticklabels=['Not Converted', 'Converted'])
        plt.title(f'Confusion Matrix - {self.model_type.replace("_", " ").title()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def plot_feature_importance(self, top_n=15, save_path=None):
        """Plot feature importance"""
        feature_importance = self.get_feature_importance()
        
        if feature_importance is None:
            print("Feature importance not available for this model type")
            return None
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {self.model_type.replace("_", " ").title()}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt

if __name__ == "__main__":
    # Test the model with mock data
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.mock_database import MockLeadDatabase
    
    # Create mock database
    db = MockLeadDatabase()
    data = db.get_training_data()
    
    # Train model
    model = LeadScoringModel(model_type='random_forest')
    metrics = model.train(data)
    
    # Test prediction
    sample_lead = {
        'lead_source': 'Email',
        'industry': 'Technology',
        'company_size': 'Medium (51-200)',
        'age': 35,
        'page_views': 10,
        'time_on_site': 800,
        'email_opens': 5,
        'email_clicks': 2,
        'form_submissions': 1,
        'content_downloads': 2,
        'days_since_contact': 7
    }
    
    result = model.predict_lead_score(sample_lead)
    print(f"\nSample prediction: {result}")
