import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

def run_arima_model(csv_file_path):
    # Read data and select the open and closing column from the data
    data = pd.read_csv(csv_file_path)

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Extracting the target variable 'Close' (dependent variable) from the DataFrame
    y = data['Close']

    # Train ARIMA model
    arima_model = ARIMA(y, order=(5, 1, 0))  # You may need to tune the order parameter
    arima_fit = arima_model.fit()

    # Print out the last closing price
    last_closing_price = data['Close'].iloc[-1]
    print("\nLast Closing Price:", last_closing_price)

    # Forecast for the next day using ARIMA
    next_day_forecast_arima = arima_fit.forecast(steps=1).values
    print("ARIMA Forecast for the next day:", next_day_forecast_arima[0])

    # Forecast for the 7th day using ARIMA
    seventh_day_forecast_arima = arima_fit.forecast(steps=7).values
    print("ARIMA Forecast for the 7th day:", seventh_day_forecast_arima[len(seventh_day_forecast_arima)-1])
    
    # Printing out predicted data of the week if needed
    #print("ARIMA Forecast for the 7th day:", seventh_day_forecast_arima)

    # Plotting the results
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], label='Actual Data')

    # Scatter plot for ARIMA forecast
    ax.scatter(data['Date'].iloc[-1], last_closing_price, color='blue', label='Last Closing Price')
    ax.scatter(data['Date'].iloc[-1] + pd.DateOffset(days=1), next_day_forecast_arima[0], color='red', label='ARIMA Next Day Forecast')
    ax.scatter(data['Date'].iloc[-1] + pd.DateOffset(days=7), seventh_day_forecast_arima[-1], color='orange', label='ARIMA 7th Day Forecast')

    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Example usage
run_arima_model('/Users/anatol/Desktop/FYP/AAPL.csv')
