import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd
import numpy as np
from datetime import datetime
from recommendation_system import DyslexiaRecommendationSystem

def load_and_train_system(demographic_file: str, ratings_file: str, run_hpo: bool = False, n_trials: int = 50, timeout: int = 1800, epochs: int = 20):
    """Initialize and train the complete dyslexia recommendation system with optional hyperparameter optimization."""
    rec_system = DyslexiaRecommendationSystem()

    try:
        rec_system.load_data(demographic_file, ratings_file)
        print(f"Data loaded: {rec_system.data_processor.num_users} users, {rec_system.data_processor.num_items} items, {len(rec_system.data_processor.ratings_data)} ratings")

        hpo_results = None
        if run_hpo:
            print("Running hyperparameter optimization...")
            hpo_results = rec_system.run_hyperparameter_optimization(
                n_trials=n_trials,
                timeout=timeout,
                save_study=True
            )
            print(f"HPO completed. Best value: {hpo_results['best_value']:.6f}")

        print("Training model...")
        training_results = rec_system.train_model(
            epochs=epochs,
            use_best_params=run_hpo
        )

        print(f"Training completed:")
        print(f" Test RMSE: {training_results['test_metrics']['rmse']:.4f}")
        print(f" Test R²: {training_results['test_metrics']['r2']:.4f}")
        print(f" Precision@10: {training_results['test_metrics']['precision_10']:.4f}")
        print(f" Recall@10: {training_results['test_metrics']['recall_10']:.4f}")

        model_path = 'dyslexia_model_optimized.pth' if run_hpo else 'dyslexia_model.pth'
        rec_system.save_model(model_path)
        print(f"Model saved to {model_path}")

        return rec_system, training_results, hpo_results

    except Exception as e:
        print(f"Error in load_and_train_system: {str(e)}")
        raise

def demo_new_user_with_preferences(rec_system: DyslexiaRecommendationSystem, user_id: int = 7000, top_k: int = 10):
    """Demonstrate recommendation generation for new user with explicitly stated learning preferences."""
    user_profile = {
        'id': user_id,
        'age': 22,
        'gender': 'M',
        'diagnosis_timing': 'Medie',
        'has_other_difficulties': 'No, solo dislessia',
        'other_difficulties_details': 'Nessuno',
        'family_history': 'Sì',
        'preferences': ['Self-made Study Aids', 'Online Lessons', 'Digital Books']
    }

    try:
        recommendations_json = rec_system.get_recommendations(user_profile, top_k=top_k)
        recommendations_data = json.loads(recommendations_json)

        print(f"\nNew User with Preferences (ID: {user_id}):")
        print(f"User Type: {recommendations_data.get('user_type', 'N/A')}")
        print(f"Preferences: {user_profile['preferences']}")

        if 'recommendations' in recommendations_data:
            print("Top Recommendations:")
            for i, rec in enumerate(recommendations_data['recommendations'][:5], 1):
                print(f" {i}. {rec.get('item_code', 'Unknown')} ({rec.get('category', 'Other')}) - {rec.get('predicted_rating', 'N/A')}/5.0")

        return recommendations_data

    except Exception as e:
        print(f"Error in demo_new_user_with_preferences: {str(e)}")
        return None

def demo_new_user_without_preferences(rec_system: DyslexiaRecommendationSystem, user_id: int = 8000, top_k: int = 10):
    """Demonstrate cold start recommendation scenario for new user without stated preferences using demographic-based suggestions."""
    user_profile = {
        'id': user_id,
        'age': 25,
        'gender': 'F',
        'diagnosis_timing': 'Superiori',
        'has_other_difficulties': 'Sì',
        'other_difficulties_details': 'Disgrafia',
        'family_history': 'No'
    }

    try:
        recommendations_json = rec_system.get_recommendations(user_profile, top_k=top_k)
        recommendations_data = json.loads(recommendations_json)

        print(f"\nNew User without Preferences (ID: {user_id}):")
        print(f"User Type: {recommendations_data.get('user_type', 'N/A')}")
        print(f"Demographics: {user_profile['age']} years, {user_profile['gender']}, diagnosed at {user_profile['diagnosis_timing']}")

        if 'recommendations' in recommendations_data:
            print("Top Recommendations (Popular Items):")
            for i, rec in enumerate(recommendations_data['recommendations'][:5], 1):
                print(f" {i}. {rec.get('item_code', 'Unknown')} ({rec.get('category', 'Other')}) - {rec.get('predicted_rating', 'N/A')}/5.0")

        return recommendations_data

    except Exception as e:
        print(f"Error in demo_new_user_without_preferences: {str(e)}")
        return None

def demo_existing_user(rec_system: DyslexiaRecommendationSystem, user_id: int = 400, top_k: int = 10):
    """Demonstrate personalized recommendations for existing user based on historical rating patterns and preferences."""
    try:
        max_user_id = rec_system.data_processor.num_users - 1
        if user_id > max_user_id:
            user_id = min(400, max_user_id)

        user_info = rec_system.get_user_information(user_id)
        
        if 'error' in user_info:
            print(f"\nExisting User Error: {user_info['error']}")
            return None

        print(f"\nExisting User (ID: {user_id}):")
        print(f"Total Ratings: {user_info['statistics']['total_ratings']}")
        print(f"Average Rating: {user_info['statistics']['average_rating']}/5.0")
        print(f"Preference Quality: {user_info['preference_quality']}")

        if user_info['highly_rated_items']:
            print("Highly Rated Items:")
            for item in user_info['highly_rated_items'][:10]:
                print(f" - {item['item_code']} ({item['category']}): {item['rating']}/5.0")

        user_profile = {'id': user_id}
        recommendations_json = rec_system.get_recommendations(user_profile, top_k=top_k)
        recommendations_data = json.loads(recommendations_json)

        return recommendations_data, user_info

    except Exception as e:
        print(f"Error in demo_existing_user: {str(e)}")
        return None, None

def run_user_comparison(rec_system: DyslexiaRecommendationSystem, user_ids: list = [100, 200, 300, 400]):
    """Analyze and compare multiple users' rating patterns and preference profiles for system evaluation."""
    print(f"\nUser Comparison Analysis:")
    print("=" * 50)

    for user_id in user_ids:
        try:
            max_user_id = rec_system.data_processor.num_users - 1
            if user_id > max_user_id:
                continue

            user_info = rec_system.get_user_information(user_id)
            if 'error' not in user_info:
                print(f"\nUser {user_id}:")
                print(f" Ratings: {user_info['statistics']['total_ratings']}")
                print(f" Avg Rating: {user_info['statistics']['average_rating']}/5.0")
                print(f" Quality: {user_info['preference_quality']}")

                if user_info['category_preferences']:
                    top_category = list(user_info['category_preferences'].keys())[0]
                    top_rating = user_info['category_preferences'][top_category]['average_rating']
                    print(f" Top Category: {top_category} ({top_rating}/5.0)")

        except Exception as e:
            print(f" Error for user {user_id}: {str(e)}")

def analyze_system_performance(training_results: dict, hpo_results: dict = None):
    """Comprehensive analysis of recommendation system performance metrics and optimization results."""
    print(f"\nSystem Performance Analysis:")
    print("=" * 50)

    if training_results:
        metrics = training_results['test_metrics']
        print(f"Model Performance:")
        print(f" RMSE: {metrics['rmse']:.4f}")
        print(f" R² Score: {metrics['r2']:.4f}")
        print(f" MAE: {metrics['mae']:.4f}")
        print(f" Precision@10: {metrics['precision_10']:.4f}")
        print(f" Recall@10: {metrics['recall_10']:.4f}")

        quality_score = (metrics['r2'] + metrics['precision_10'] + metrics['recall_10']) / 3
        print(f" Overall Quality Score: {quality_score:.4f}")

    if hpo_results:
        print(f"\nHyperparameter Optimization:")
        print(f" Best Value: {hpo_results['best_value']:.6f}")
        print(f" Trials Completed: {hpo_results['n_trials']}")
        print(f" Best Parameters: {hpo_results['best_params']}")

# def generate_batch_recommendations(rec_system: DyslexiaRecommendationSystem, output_file: str = 'batch_recommendations.csv'):
#     """Generate recommendations for multiple users simultaneously and export results to CSV for analysis."""
#     print(f"\nGenerating batch recommendations...")

#     try:
#         batch_results = []
#         sample_users = [100, 200, 300, 400, 500]
#         max_user_id = rec_system.data_processor.num_users - 1

#         for user_id in sample_users:
#             if user_id <= max_user_id:
#                 try:
#                     user_profile = {'id': user_id}
#                     recommendations_json = rec_system.get_recommendations(user_profile, top_k=5)
#                     recommendations_data = json.loads(recommendations_json)

#                     if 'recommendations' in recommendations_data:
#                         for i, rec in enumerate(recommendations_data['recommendations']):
#                             batch_results.append({
#                                 'user_id': user_id,
#                                 'rank': i + 1,
#                                 'item_code': rec.get('item_code', 'Unknown'),
#                                 'category': rec.get('category', 'Other'),
#                                 'predicted_rating': rec.get('predicted_rating', 0),
#                                 'recommendation_type': rec.get('recommendation_type', 'unknown'),
#                                 'timestamp': datetime.now().isoformat()
#                             })

#                 except Exception as e:
#                     print(f" Error for user {user_id}: {str(e)}")

#         if batch_results:
#             df = pd.DataFrame(batch_results)
#             df.to_csv(output_file, index=False)
#             print(f"Batch recommendations saved to {output_file}")
#             print(f"Total recommendations generated: {len(batch_results)}")

#         return batch_results

#     except Exception as e:
#         print(f"Error in generate_batch_recommendations: {str(e)}")
#         return []

def interactive_demo(rec_system: DyslexiaRecommendationSystem):
    """Provide interactive command-line interface for testing various recommendation scenarios and system capabilities."""
    print(f"\nInteractive Demo Mode")
    print("=" * 30)

    while True:
        print(f"\nOptions:")
        print(f"1. Test existing user")
        print(f"2. Test new user with preferences")
        print(f"3. Test new user without preferences")
        print(f"4. Compare multiple users")
        # print(f"5. Generate batch recommendations")
        print(f"5. Exit")

        try:
            choice = input(f"\nEnter your choice (1-6): ").strip()

            if choice == '1':
                user_id = int(input(f"Enter user ID (0-{rec_system.data_processor.num_users-1}): "))
                demo_existing_user(rec_system, user_id)

            elif choice == '2':
                user_id = int(input(f"Enter new user ID (7000-9999): "))
                demo_new_user_with_preferences(rec_system, user_id)

            elif choice == '3':
                user_id = int(input(f"Enter new user ID (7000-9999): "))
                demo_new_user_without_preferences(rec_system, user_id)

            elif choice == '4':
                run_user_comparison(rec_system)

            # elif choice == '5':
            #     output_file = input(f"Enter output filename (default: batch_recommendations.csv): ").strip()
            #     if not output_file:
            #         output_file = 'batch_recommendations.csv'
            #     generate_batch_recommendations(rec_system, output_file)

            elif choice == '5':
                print(f"Exiting interactive demo...")
                break

            else:
                print(f"Invalid choice. Please enter 1-6.")

        except ValueError:
            print(f"Please enter a valid number.")
        except KeyboardInterrupt:
            print(f"\nExiting interactive demo...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Main application entry point orchestrating system initialization, training, and demonstration workflows."""
    print("🧠 Dyslexia Recommendation System")
    print("=" * 50)

    demographic_file = 'cleaned_data_priv.xlsx - Risposte del modulo 1.csv'
    ratings_file = 'values.csv'

    try:
        run_hpo = input("Run hyperparameter optimization? (y/n): ").lower() == 'y'
        
        rec_system, training_results, hpo_results = load_and_train_system(
            demographic_file=demographic_file,
            ratings_file=ratings_file,
            run_hpo=run_hpo,
            n_trials=50 if run_hpo else 0,
            timeout=1800,
            epochs=20
        )

        analyze_system_performance(training_results, hpo_results)

        print(f"\n" + "=" * 50)
        print(f"RECOMMENDATION DEMOS")
        print(f"=" * 50)

        demo_new_user_with_preferences(rec_system)
        demo_new_user_without_preferences(rec_system)
        demo_existing_user(rec_system)
        run_user_comparison(rec_system)

        interactive_mode = input(f"\nRun interactive demo? (y/n): ").lower() == 'y'
        if interactive_mode:
            interactive_demo(rec_system)

        # generate_batch = input(f"\nGenerate batch recommendations? (y/n): ").lower() == 'y'
        # if generate_batch:
        #     generate_batch_recommendations(rec_system)

        print(f"\n🎉 Demo completed successfully!")

    except FileNotFoundError as e:
        print(f"❌ Data files not found: {str(e)}")
        print(f"Please ensure data files are in the correct location")
    except KeyboardInterrupt:
        print(f"\n⏹️ Process interrupted by user")
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print(f"Please ensure all required packages are installed")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
