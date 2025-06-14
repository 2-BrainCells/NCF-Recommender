"""
Main entry point for Dyslexia Recommendation System
Usage examples and demo
"""
import warnings
warnings.filterwarnings('ignore')

from recommendation_system import DyslexiaRecommendationSystem


def main():
    """Main function demonstrating the recommendation system"""
    
    print("🧠 Dyslexia Recommendation System")
    print("=" * 50)
    
    try:
        # Initialize the recommendation system
        rec_system = DyslexiaRecommendationSystem()
        
        # Load and preprocess data
        print("📊 Loading data...")
        rec_system.load_data(
            demographic_file='cleaned_data_priv.xlsx - Risposte del modulo 1.csv',
            ratings_file='values.csv'
        )
        
        # Print data statistics
        print(f"✅ Data loaded:")
        print(f"   Users: {rec_system.data_processor.num_users}")
        print(f"   Items: {rec_system.data_processor.num_items}")
        print(f"   Ratings: {len(rec_system.data_processor.ratings_data)}")
        
        # Optional: Run hyperparameter optimization
        run_hpo = input("🔧 Run hyperparameter optimization? (y/n): ").lower() == 'y'
        
        if run_hpo:
            print("🚀 Starting hyperparameter optimization...")
            hpo_results = rec_system.run_hyperparameter_optimization(
                n_trials=50,  # Adjust based on your time constraints
                timeout=1800,  # 30 minutes
                save_study=True
            )
            
            print("✅ HPO Results:")
            print(f"   Best value: {hpo_results['best_value']:.6f}")
            print(f"   Best params: {hpo_results['best_params']}")
            print(f"   Trials completed: {hpo_results['n_trials']}")
        
        # Train the model
        print("🎯 Training model...")
        training_results = rec_system.train_model(
            epochs=20, 
            use_best_params=run_hpo  # Use HPO results if available
        )
        
        print("✅ Training completed!")
        print(f"   Test RMSE: {training_results['test_metrics']['rmse']:.4f}")
        print(f"   Test R²: {training_results['test_metrics']['r2']:.4f}")
        print(f"   Precision @10: {training_results['test_metrics']['precision_10']:.4f}")
        print(f"   Recall @10: {training_results['test_metrics']['recall_10']:.4f}")
        
        # Save the trained model
        model_path = 'dyslexia_recommendation_model_optimized.pth' if run_hpo else 'dyslexia_recommendation_model.pth'
        rec_system.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Demo recommendations
        print("\n" + "=" * 50)
        print("🎯 RECOMMENDATION DEMOS")
        print("=" * 50)
        
        # Example 1: New user with preferences
        print("\n📝 Example 1: New user with preferences")
        new_user_profile = {
            'id': 7000,
            'age': 20,
            'gender': 'M',
            'diagnosis_timing': 'Medie',
            'has_other_difficulties': 'No, solo dislessia',
            'other_difficulties_details': 'Nessuno',
            'family_history': 'Si',
            'preferences': ['Self-made Study Aids', 'Online Lessons']
        }
        
        recommendations_json = rec_system.get_recommendations(new_user_profile, top_k=5)
        print("Recommendations:")
        print(recommendations_json)
        
        # Example 2: Existing user
        print("\n📝 Example 2: Existing user")
        max_user_id = rec_system.data_processor.num_users - 1
        existing_user_id = min(400, max_user_id)  # Ensure user exists
        existing_user_profile = {'id': existing_user_id}
        recommendations_json = rec_system.get_recommendations(existing_user_profile, top_k=5)
        print("Recommendations:")
        print(recommendations_json)
        
        # Example 3: New user without preferences (popular items)
        print("\n📝 Example 3: New user without preferences")
        new_user_no_prefs = {
            'id': 8000,
            'age': 25,
            'gender': 'F'
        }
        recommendations_json = rec_system.get_recommendations(new_user_no_prefs, top_k=5)
        print("Recommendations:")
        print(recommendations_json)
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()