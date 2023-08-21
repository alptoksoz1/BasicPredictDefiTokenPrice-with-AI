# Required Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# Load Data
data = {}

tokens_data_files = {
    'aave': {
        'tvl': 'aaveHistoricalTVL.json',
        'price': 'Aave_20.06.2023-21.08.2023_historical_data_coinmarketcap.csv'
    },
    'dydx': {
        'tvl': 'dydxHistoricalTVL.json',
        'price': 'dYdX_20.06.2023-21.08.2023_historical_data_coinmarketcap.csv'
    },
    'uniswap': {
        'tvl': 'uniswapHistoricalTVL.json',
        'price': 'Uniswap_20.06.2023-21.08.2023_historical_data_coinmarketcap.csv'
    },
    'sushiswap': {
        'tvl': 'sushiSwapHistoricalTVL.json',
        'price': 'SushiSwap_20.06.2023-21.08.2023_historical_data_coinmarketcap.csv'
    },
    'pancakeswap': {
        'tvl': 'pancakeSwapHistoricalTVL.json',
        'price': 'PancakeSwap_20.06.2023-21.08.2023_historical_data_coinmarketcap.csv'
    }
}

for token, files in tokens_data_files.items():
    with open(files["tvl"], 'r') as file:
        tvl_data = json.load(file)['tvl']
    data[token] = {
        'tvl': pd.DataFrame(tvl_data),
        'price': pd.read_csv(files["price"], delimiter=';')
    }

# Function to preprocess and train model for each token
def process_and_train(token):
    tvl_df = data[token]['tvl']
    tvl_df['date'] = pd.to_datetime(tvl_df['date'], unit='s').dt.date
    price = data[token]['price']
    price['timestamp'] = pd.to_datetime(price['timestamp']).dt.date
    merged = pd.merge(tvl_df, price[['timestamp', 'close']], left_on='date', right_on='timestamp', how='inner')
    X = merged[['totalLiquidityUSD']]
    y = merged['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{token.capitalize()} Token - Mean Squared Error: {mse}")
    
    # Plot Actual vs Predicted Prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Prices', marker='o', linestyle='-', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Prices', marker='x', linestyle='--', color='red')
    plt.title(f'Actual vs Predicted {token.capitalize()} Token Prices')
    plt.xlabel('Index')
    plt.ylabel('Price (in USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{token}_actual_vs_predicted.png")
    plt.show()

# Process and train model for each token
for token in tokens_data_files.keys():
    process_and_train(token)
