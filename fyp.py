import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def run_mlp_model(csv_file_path):
    # Reads data and selects the open and closing column from the data
    data = pd.read_csv(csv_file_path)

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Extracting the target variable 'Close' (dependent variable) from the DataFrame
    y = data['Close']

    # Extracting the feature 'Open' (independent variable) from the DataFrame
    X = data['Open']

    # Reshape the data as MLPRegressor expects a 2D array
    X = X.values.reshape(-1, 1)
    y = y.values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display the shapes of the resulting sets
    '''
    print("\nX_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    '''
     # Display the shapes of the resulting sets
    print("\nTrain Set:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("\nTest Set:")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, ),
        max_iter=1000,
        random_state=42
    )

    mlp_model.fit(X_train, y_train)

    # Sets the last closing price
    last_closing_price = data['Close'].iloc[-1]
    
    # Forecast for the next day
    # Assuming the last row of the dataset represents the latest data
    last_day_open = data['Open'].iloc[-1].reshape(-1, 1)
    next_day_forecast = mlp_model.predict(last_day_open)
    
    # Calculate error metrics for the next day forecast
    #mse_next_day = mean_squared_error(y_test[:1], next_day_forecast)
    #r2_next_day = r2_score(y_test[:1], next_day_forecast)

    print("\nError metrics for the next day forecast:")
    #print(f'Mean Squared Error: {mse_next_day}')
    #print(f'R-squared: {r2_next_day}')

    # Forecast for the 7th day from the current day
    # Assuming the last row of the dataset represents the latest data
    seventh_day_open = data['Open'].iloc[-1].reshape(-1, 1)
    seventh_day_forecasts = []
    for _ in range(7):
        seventh_day_forecast = mlp_model.predict(seventh_day_open)
        seventh_day_open = seventh_day_forecast.reshape(-1, 1)
        seventh_day_forecasts.append(seventh_day_forecast[0])
        
    # Prints out the closing price predictions with the last given value
    print("\nLast Closing Price:", last_closing_price)
    print("Forecast for the next day:", next_day_forecast[0])
    print("Forecasts for the 7th day from the current day:", seventh_day_forecasts[6])
    
    # Calculate % difference from the last day to next day forecast
    percentage_diff_next_day = ((next_day_forecast[0] - last_day_open[-1][0]) / last_day_open[-1][0]) * 100


    # Calculate % difference from the last day to 7th day forecast
    percentage_diff_7th_day = ((seventh_day_forecasts[-1] - last_day_open[-1][0]) / last_day_open[-1][0]) * 100

    # Print the results
    print("\nPercentage difference from the last day to next day forecast:", percentage_diff_next_day, "%")
    print("Percentage difference from the last day to 7th day forecast:", percentage_diff_7th_day, "%")

    # Calculate error metrics for the 7th day forecast
    mse_seventh_day = mean_squared_error(y_test[:7], seventh_day_forecasts)
    r2_seventh_day = r2_score(y_test[:7], seventh_day_forecasts)

    print("\nError metrics for the 7th day forecast:")
    print(f'Mean Squared Error: {mse_seventh_day}')
    print(f'R-squared: {r2_seventh_day}')

    # Plotting the results
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], label='Actual Data')
    ax.scatter(data['Date'].iloc[-1], next_day_forecast[0], color='red', label='Next Day Forecast')
    ax.scatter(data['Date'].iloc[-1] + pd.DateOffset(days=6), seventh_day_forecasts[6], color='orange', label='7th Day Forecast')

    # Plotting the predicted closing prices made by the model
    prediction_dates = pd.date_range(start=data['Date'].iloc[-1], periods=7, freq='D')
    model_predictions = mlp_model.predict(data['Open'].values.reshape(-1, 1)[-7:])
    ax.plot(prediction_dates, model_predictions, color='green', label='Model Predictions')

    # Formatting x-axis to show years
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Set x-axis limits to include the entire range of dates plus one additional year for better visualization
    ax.set_xlim(data['Date'].iloc[0], prediction_dates[-1] + pd.DateOffset(days=365))

    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
    
# Example usage
run_mlp_model(r"C:\Users\ASatl\OneDrive\Desktop\let\AAPL.csv")
