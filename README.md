# Token Price Prediction Project

## Project Description
This project aims to predict the prices of crypto tokens in the DeFi ecosystem using the Total Value Locked (TVL) as a feature. Using historical data for several tokens, the project trained prediction models to evaluate the relationship between TVL and token prices.

## Setup

### Required Libraries
- pandas
- numpy
- sklearn
- matplotlib

To install the libraries:
```bash
pip install pandas numpy sklearn matplotlib
```




### Data Sets
The data sets used in this project include:
- Aave
- dYdX
- Uniswap
- SushiSwap
- PancakeSwap

For each token, historical data was collected containing daily prices and TVL.

## Modeling
In this project, the RandomForestRegressor model was used to predict token prices using TVL as a feature. The model was trained using the training set and its performance was evaluated by comparing with the actual prices in the test set.

## Usage
To run the project code:
```bash
python __main__.py
```



## Results
The predictions of the model, when compared to the actual prices, indicated a significant influence of TVL on the crypto token prices. However, it's essential to remember that the volatility of the crypto market and external factors can significantly impact prices.

