import pandas as pd
import numpy as np

def generate_house_data(num_samples=1000):
    np.random.seed(42)
    sqft = np.random.normal(2000, 500, num_samples).astype(int)
    bedrooms = np.random.randint(1, 6, num_samples)
    bathrooms = np.random.randint(1, 4, num_samples)
    price = sqft * 150 + bedrooms * 10000 + bathrooms * 5000 + np.random.normal(0, 20000, num_samples)
    price = price.astype(int)
    data = pd.DataFrame({
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'price': price
    })
    return data

def save_sample_data(data, filename='data/house_data.csv'):
    data.to_csv(filename, index=False)
