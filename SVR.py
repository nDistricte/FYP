import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def run_svr_model(csv_file_path):
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

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Create and train the SVR model
    svr_model = SVR(kernel="rbf")
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    svr_model.fit(X_train_scaled, y_train_scaled)

    # Make predictions on the test set
    X_test_scaled = scaler_x.transform(X_test)
    y_pred_scaled = svr_model.predict(X_test_scaled)

    # Inverse transform the predictions to get them back to the original scale
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)  # Reshape to 2D array
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # For training data
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # For testing data
    X_test_scaled = scaler_x.transform(X_test)
    y_pred_scaled = svr_model.predict(X_test_scaled)
    
    # Forecast for the next day
    # Assuming the last row of the dataset represents the latest data
    last_day_open = data['Open'].iloc[-1].reshape(-1, 1)
    next_day_forecast_scaled = svr_model.predict(scaler_x.transform(last_day_open))
    next_day_forecast_unscaled = scaler_y.inverse_transform(next_day_forecast_scaled.reshape(-1, 1))

    print("\nThe next day prediction is: ", next_day_forecast_unscaled[0, 0])

    # Initialize an array to store forecasts for the next 6 days
    week_forecasts = []

    # Forecast for the next 6 days
    for i in range(1, 7):
        # Use the predicted value of the previous day as the 'Open' value for forecasting the next day
        next_day_open = next_day_forecast_unscaled.reshape(-1, 1)
        next_day_forecast_scaled = svr_model.predict(scaler_x.transform(next_day_open))
        next_day_forecast_unscaled = scaler_y.inverse_transform(next_day_forecast_scaled.reshape(-1, 1))

        # Append the forecast to the array
        week_forecasts.append(next_day_forecast_unscaled[0, 0])

    # The 'week_forecasts' array now contains the forecasts for the entire week
    print("7th Day Forecasts:", week_forecasts[len(week_forecasts)-1], end="\n\n")
    
    # Printing out predicted data of the week if needed
    #print("7th Day Forecasts:", week_forecasts, end="\n\n")

    # Calculate % difference from the last day to next day forecast
    percentage_diff_next_day = ((next_day_forecast_unscaled[0, 0] - last_day_open[-1][0]) / last_day_open[-1][0]) * 100

    # Calculate % difference from the last day to 7th day forecast
    percentage_diff_7th_day = ((week_forecasts[-1] - last_day_open[-1][0]) / last_day_open[-1][0]) * 100

    # Print the results
    print("Percentage difference from the last day to next day forecast:", percentage_diff_next_day, "%")
    print("Percentage difference from the last day to 7th day forecast:", percentage_diff_7th_day, "%")

    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Plotting the results
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], label='Actual Data')

    # Scatter plot for the last given value, first forecast, and last forecast
    ax.scatter(data['Date'].iloc[-1], last_day_open[-1], color='blue', label='Last Given Value')
    ax.scatter(data['Date'].iloc[-1] + pd.DateOffset(days=1), next_day_forecast_unscaled[0, 0], color='red', label='Next Day Forecast')
    ax.scatter(data['Date'].iloc[-1] + pd.DateOffset(days=6), week_forecasts[-1], color='green', label='7th Day Forecast')

    # Connect the three points with a line
    ax.plot([data['Date'].iloc[-1], data['Date'].iloc[-1] + pd.DateOffset(days=1), data['Date'].iloc[-1] + pd.DateOffset(days=6)],
            [last_day_open[-1][0], next_day_forecast_unscaled[0, 0], week_forecasts[-1]],
            linestyle='dashed', color='gray', label='Connection Line')

    # Formatting x-axis to show years
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Set x-axis limits to include the entire range of dates plus one additional year for better visualization
    start_date = data['Date'].iloc[0]
    end_date = data['Date'].iloc[-1] + pd.DateOffset(days=365)  # Adding one year to the last date
    ax.set_xlim(start_date, end_date)

    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()



# Example usage
run_svr_model(r"C:\Users\ASatl\OneDrive\Desktop\let\AAPL.csv")

