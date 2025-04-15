import yfinance as yf
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np

def runPrediction(stock):
    stock = stock.upper()
    DATA_PATH = stock + "_data.json"

    if os.path.exists(DATA_PATH):
        # Load data from file if it already exists
        with open(DATA_PATH) as file:
            stocks_hist = pd.read_json(DATA_PATH)
    else:
        # Fetch data from Yahoo Finance if file does not exist
        stocks_hist = yf.download(stock, period="6mo")
        
        # Saving data to file
        stocks_hist.to_json(DATA_PATH)

    data = stocks_hist[["Close"]]
    data = data.rename(columns={"Close": "Actual_Close"})

    # Adding a column to the data that indicates whether the price went up or down
    # if the price went up, the value is 1, otherwise it is 0
    data["Target"] = stocks_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
    stocks_prev = stocks_hist.copy().shift(1)

    # Setting up predictors
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    data = data.join(stocks_prev[predictors]).iloc[1:]
    weekly_mean = data.rolling(7).mean()
    quarterly_mean = data.rolling(90).mean()
    annual_mean = data.rolling(365).mean()
    weekly_trend = data.shift(1).rolling(7).mean()["Target"]

    data["weekly_mean"] = weekly_mean["Close"] / data["Close"]
    data["quarterly_mean"] = quarterly_mean["Close"] / data["Close"]
    data["annual_mean"] = annual_mean["Close"] / data["Close"]

    data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
    data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
    data["weekly_trend"] = weekly_trend

    data["open_close_ratio"] = data["Open"] / data["Close"]
    data["high_close_ratio"] = data["High"] / data["Close"]
    data["low_close_ratio"] = data["Low"] / data["Close"]

    full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]

    # Setting up the model
    model = RandomForestClassifier(n_estimators=500, min_samples_split=200, random_state=1)

    # Backtesting engine
    def backtest(data, model, predictors, start=100, step=5):
        predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:i+step].copy()
            
            model.fit(train[predictors], train["Target"])
            
            preds = model.predict_proba(test[predictors])[:,1]
            preds = pd.Series(preds, index=test.index)
            preds[preds > 0.55] = 1
            preds[preds <= 0.55] = 0
            combined = pd.concat({"Target": test["Target"], "Predictions": preds}, axis=1)
            predictions.append(combined)
            
        return pd.concat(predictions)

    def simulate_trading(data, predictions, initial_capital=10000):
        capital = initial_capital
        stock_held = 0
        stock_price = data["Close"]
        
        for i in range(len(predictions)):
            if predictions["Predictions"].iloc[i] == 1 and capital > 0:
                # If prediction is to buy (price will go up), buy stock
                # Invest all the capital
                stock_to_buy = capital / stock_price.iloc[i]
                stock_held += stock_to_buy
                capital = 0
                print(f"Buying {stock_to_buy} stock at {stock_price.iloc[i]}, Day {i + 1}")
                
            elif predictions["Predictions"].iloc[i] == 0 and stock_held > 0:
                # If prediction is to sell (price will go down), sell stock
                print(f"Selling {stock_held} stock at {stock_price.iloc[i]}, Day {i + 1}")
                capital += stock_held * stock_price.iloc[i]
                stock_held = 0  # Sold all the stock

        # Final value = remaining capital + value of held stock
        final_value = capital + stock_held * stock_price.iloc[-1]
        return final_value

    # Running the backtest
    predictions = backtest(data, model, full_predictors)
    print(predictions)
    end_capital = simulate_trading(data, predictions)
    print(f"Final value: {end_capital}")
    return end_capital

def test(stocks):
        gainer = 0
        loser = 0
        for stock in stocks:
            if runPrediction(stock) > 10000:
                gainer += 1
        print(gainer / len(stocks) * 100)
        print(loser / len(stocks) * 100)
        
test(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"])