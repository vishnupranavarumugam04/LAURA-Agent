# File: utils/real_ml_models.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime, timedelta

class RealProductivityPredictor:
    """ACTUAL ML model for productivity prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'sleep_quality', 'stress_level', 'focus_score', 
            'time_of_day', 'task_complexity', 'break_frequency',
            'previous_day_productivity', 'day_of_week', 
            'tasks_completed_yesterday', 'sleep_hours'
        ]
        
    def generate_realistic_training_data(self, n_samples=5000):
        """Generate realistic synthetic dataset"""
        np.random.seed(42)
        
        data = {
            'sleep_quality': np.random.normal(75, 12, n_samples),
            'stress_level': np.random.gamma(2, 15, n_samples),
            'focus_score': np.random.beta(8, 2, n_samples) * 100,
            'time_of_day': np.random.choice(range(6, 23), n_samples),
            'task_complexity': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.5, 0.2]),
            'break_frequency': np.random.normal(60, 20, n_samples),
            'previous_day_productivity': np.random.normal(75, 15, n_samples),
            'day_of_week': np.random.choice(range(7), n_samples),
            'tasks_completed_yesterday': np.random.poisson(5, n_samples),
            'sleep_hours': np.random.normal(7, 1.5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic productivity with non-linear relationships
        productivity = (
            0.25 * df['sleep_quality'] +
            -0.20 * df['stress_level'] +
            0.20 * df['focus_score'] +
            10 * np.sin((df['time_of_day'] - 9) * np.pi / 12) +  # Peak at 9-11 AM
            -8 * df['task_complexity'] +
            0.05 * df['break_frequency'] +
            0.15 * df['previous_day_productivity'] +
            -3 * (df['day_of_week'] == 0).astype(int) +  # Monday penalty
            2 * df['tasks_completed_yesterday'] +
            5 * df['sleep_hours'] +
            np.random.normal(0, 8, n_samples)  # Realistic noise
        )
        
        # Clip to 0-100 range
        df['productivity'] = np.clip(productivity, 0, 100)
        
        return df
    
    def train(self, data=None):
        """Train the model with hyperparameter tuning"""
        if data is None:
            data = self.generate_realistic_training_data()
        
        X = data[self.feature_names]
        y = data['productivity']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        print("Training with hyperparameter tuning...")
        grid_search.fit(X_train_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'best_params': grid_search.best_params_,
            'cv_scores': cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='r2'
            ).tolist()
        }
        
        print(f"Training MAE: {metrics['train_mae']:.2f}")
        print(f"Test MAE: {metrics['test_mae']:.2f}")
        print(f"Test R²: {metrics['test_r2']:.3f}")
        
        # Save model
        self.save_model()
        
        return metrics
    
    def predict(self, features: dict):
        """Make prediction with confidence intervals"""
        if self.model is None:
            self.load_model()
        
        # Convert dict to DataFrame
        feature_array = [[features.get(f, 0) for f in self.feature_names]]
        feature_scaled = self.scaler.transform(feature_array)
        
        # Get prediction from all trees for confidence interval
        predictions = np.array([
            tree.predict(feature_scaled)[0] 
            for tree in self.model.estimators_
        ])
        
        mean_pred = predictions.mean()
        std_pred = predictions.std()
        
        return {
            'predicted_productivity': round(mean_pred, 1),
            'confidence_interval': [
                round(mean_pred - 1.96 * std_pred, 1),
                round(mean_pred + 1.96 * std_pred, 1)
            ],
            'confidence_score': min(1.0, 1 / (1 + std_pred / 10))
        }
    
    def get_feature_importance(self):
        """Get real feature importance from trained model"""
        if self.model is None:
            self.load_model()
        
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        return dict(sorted(
            importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
    
    def save_model(self):
        """Save trained model"""
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/productivity_model.pkl')
        joblib.dump(self.scaler, 'models/productivity_scaler.pkl')
        
        # Save metadata
        metadata = {
            'trained_date': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'model_type': 'RandomForestRegressor'
        }
        with open('models/productivity_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load('models/productivity_model.pkl')
            self.scaler = joblib.load('models/productivity_scaler.pkl')
        except FileNotFoundError:
            print("Model not found. Training new model...")
            self.train()


class RealTaskSuccessClassifier:
    """ACTUAL classification model for task success"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'complexity_score', 'time_of_day_score', 'energy_level',
            'similar_tasks_success_rate', 'estimated_duration',
            'current_stress', 'available_time', 'task_priority'
        ]
    
    def generate_training_data(self, n_samples=3000):
        """Generate realistic task outcome data"""
        np.random.seed(42)
        
        data = {
            'complexity_score': np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.4, 0.2]),
            'time_of_day_score': np.random.randint(1, 10, n_samples),
            'energy_level': np.random.normal(70, 20, n_samples),
            'similar_tasks_success_rate': np.random.beta(7, 3, n_samples) * 100,
            'estimated_duration': np.random.lognormal(3.5, 0.5, n_samples),
            'current_stress': np.random.gamma(2, 15, n_samples),
            'available_time': np.random.lognormal(4, 0.6, n_samples),
            'task_priority': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3])
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic success probability
        success_logit = (
            -2 +  # Intercept
            -0.8 * df['complexity_score'] +
            0.3 * df['time_of_day_score'] +
            0.02 * df['energy_level'] +
            0.03 * df['similar_tasks_success_rate'] +
            -0.01 * df['estimated_duration'] +
            -0.02 * df['current_stress'] +
            0.01 * df['available_time'] +
            0.2 * df['task_priority']
        )
        
        success_prob = 1 / (1 + np.exp(-success_logit))
        df['success'] = (np.random.random(n_samples) < success_prob).astype(int)
        
        return df
    
    def train(self, data=None):
        """Train classification model"""
        if data is None:
            data = self.generate_training_data()
        
        X = data[self.feature_names]
        y = data['success']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting Classifier
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        gbc = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gbc, param_grid, cv=5,
            scoring='f1', n_jobs=-1
        )
        
        print("Training task success classifier...")
        grid_search.fit(X_train_scaled, y_train)
        
        self.model = grid_search.best_estimator_
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_f1': f1_score(y_test, test_pred),
            'best_params': grid_search.best_params_
        }
        
        print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"Test F1: {metrics['test_f1']:.3f}")
        
        # Save
        self.save_model()
        
        return metrics
    
    def predict_probability(self, features: dict):
        """Predict success probability"""
        if self.model is None:
            self.load_model()
        
        feature_array = [[features.get(f, 0) for f in self.feature_names]]
        feature_scaled = self.scaler.transform(feature_array)
        
        prob = self.model.predict_proba(feature_scaled)[0][1]
        
        return {
            'success_probability': round(prob * 100, 1),
            'risk_level': 'low' if prob > 0.7 else 'medium' if prob > 0.4 else 'high',
            'confidence': 0.85
        }
    
    def save_model(self):
        """Save classifier"""
        import os
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/task_success_model.pkl')
        joblib.dump(self.scaler, 'models/task_success_scaler.pkl')
    
    def load_model(self):
        """Load classifier"""
        try:
            self.model = joblib.load('models/task_success_model.pkl')
            self.scaler = joblib.load('models/task_success_scaler.pkl')
        except FileNotFoundError:
            print("Model not found. Training new model...")
            self.train()


# Training script
if __name__ == "__main__":
    print("="*50)
    print("Training Productivity Predictor...")
    print("="*50)
    
    prod_model = RealProductivityPredictor()
    prod_metrics = prod_model.train()
    
    print("\n" + "="*50)
    print("Training Task Success Classifier...")
    print("="*50)
    
    task_model = RealTaskSuccessClassifier()
    task_metrics = task_model.train()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("\nModels saved to 'models/' directory")
    print("\nProductivity Model Metrics:")
    for k, v in prod_metrics.items():
        print(f"  {k}: {v}")
    
    print("\nTask Success Model Metrics:")
    for k, v in task_metrics.items():
        print(f"  {k}: {v}")