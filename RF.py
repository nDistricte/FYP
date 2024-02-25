import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
    
def run_rf_model(csv_file_path):
    
    data = pd.data.read(csv_file_path)
    
    date['Date'] = pd.to_datetime(data['Date'])
    
    y = data['Close']
    
    X = data['Open']
    
    # Reshape the data as MLPRegressor expects a 2D array
    X = X.values.reshape(-1, 1)
    y = y.values
    
    x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
    regr = RandomForestRegressor(max_depth=2, random_state=0)