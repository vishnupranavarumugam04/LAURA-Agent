from utils.real_ml_models import RealProductivityPredictor, RealTaskSuccessClassifier

def main():
    print("="*60)
    print("TRAINING ALL ML MODELS")
    print("="*60)
    
    # Train productivity predictor
    print("\n[1/2] Training Productivity Predictor...")
    prod = RealProductivityPredictor()
    prod_metrics = prod.train()
    
    print(f"\n✅ Productivity Model Complete!")
    print(f"   - Test MAE: {prod_metrics['test_mae']:.2f}")
    print(f"   - Test R²: {prod_metrics['test_r2']:.3f}")
    
    # Train task success classifier
    print("\n[2/2] Training Task Success Classifier...")
    task = RealTaskSuccessClassifier()
    task_metrics = task.train()
    
    print(f"\n✅ Task Success Model Complete!")
    print(f"   - Test Accuracy: {task_metrics['test_accuracy']:.3f}")
    print(f"   - Test F1 Score: {task_metrics['test_f1']:.3f}")
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED AND SAVED!")
    print("="*60)
    print("\nModels saved to: aloc/models/")
    print("You can now run the app with: streamlit run app.py")

if __name__ == "__main__":
    main()