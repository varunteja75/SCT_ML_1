import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def prepare_features(self, data: pd.DataFrame):
        # Assuming data has columns: sqft, bedrooms, bathrooms, price
        X = data[['sqft', 'bedrooms', 'bathrooms']]
        y = data['price']
        return X, y

    def train_model(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def evaluate_model(self):
        test_r2 = r2_score(self.y_test, self.y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        test_mae = mean_absolute_error(self.y_test, self.y_pred)
        metrics = {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        return metrics, self.y_pred

    def get_feature_importance(self):
        coef_dict = dict(zip(['sqft', 'bedrooms', 'bathrooms'], self.model.coef_))
        intercept = self.model.intercept_
        return coef_dict, intercept

    def predict_price(self, sqft, bedrooms, bathrooms):
        features = np.array([[sqft, bedrooms, bathrooms]])
        price = self.model.predict(features)[0]
        return price

    def visualize_results(self, y_pred):
        plt.figure(figsize=(10,6))
        plt.scatter(self.y_test, y_pred, alpha=0.7)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted House Prices')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.show()
