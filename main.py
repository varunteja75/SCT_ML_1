import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from house_price_predictor import HousePricePredictor
from data_generator import generate_house_data, save_sample_data
import pandas as pd

def main():
    print("🏠 House Price Prediction with Linear Regression")
    print("=" * 50)
    
    # Create predictor instance
    predictor = HousePricePredictor()
    
    # Generate sample data (or load your own CSV)
    print("📊 Generating sample data...")
    data = generate_house_data(1000)
    
    # Save data for future use
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/house_data.csv', index=False)
    print("Data saved to data/house_data.csv")
    
    # Display data info
    print(f"\nDataset shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
    print(f"\nData statistics:")
    print(data.describe())
    
    # Prepare features
    X, y = predictor.prepare_features(data)
    
    # Train model
    print("\n🎯 Training model...")
    X_train, X_test, y_train, y_test = predictor.train_model(X, y)
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    metrics, y_pred = predictor.evaluate_model()
    
    print(f"Test R² Score: {metrics['test_r2']:.4f}")
    print(f"Test RMSE: ${metrics['test_rmse']:,.2f}")
    print(f"Test MAE: ${metrics['test_mae']:,.2f}")
    
    # Feature importance
    print("\n🔍 Feature importance:")
    coef_dict, intercept = predictor.get_feature_importance()
    print(f"Intercept: ${intercept:,.2f}")
    for feature, coef in coef_dict.items():
        print(f"{feature}: ${coef:,.2f}")
    
    # Sample predictions
    print("\n🏡 Sample predictions:")
    examples = [
        (1500, 3, 2, "Small family home"),
        (2500, 4, 3, "Large family home"), 
        (1200, 2, 1, "Starter home"),
        (3500, 5, 4, "Luxury home")
    ]
    
    for sqft, beds, baths, description in examples:
        price = predictor.predict_price(sqft, beds, baths)
        print(f"{description}: {sqft} sqft, {beds} bed, {baths} bath → ${price:,.2f}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    predictor.visualize_results(y_pred)
    
    # Interactive prediction
    print("\n🎮 Interactive prediction mode:")
    print("Enter house details to get price prediction (or 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nEnter 'sqft bedrooms bathrooms' (e.g., '2000 3 2'): ")
            if user_input.lower() == 'quit':
                break
                
            sqft, beds, baths = map(float, user_input.split())
            price = predictor.predict_price(sqft, beds, baths)
            print(f"Predicted price: ${price:,.2f}")
            
        except (ValueError, IndexError):
            print("Invalid input. Please enter three numbers separated by spaces.")
        except KeyboardInterrupt:
            break
    
    print("\nThanks for using the House Price Predictor!")

if __name__ == "__main__":
    main()