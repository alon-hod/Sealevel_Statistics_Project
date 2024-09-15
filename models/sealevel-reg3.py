import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define the file paths
co2_path = 'raw_data/co2.csv'
sealevel_path = 'raw_data/sealevel.csv'
temp_path = 'raw_data/temp.csv'

# Read the CSV files
co2_df = pd.read_csv(co2_path)
sealevel_df = pd.read_csv(sealevel_path)
temp_df = pd.read_csv(temp_path, index_col=None)

# Apply nanmedian to fill missing values in 'TOPEX/Poseidon', 'Jason-1', 'Jason-2', 'Jason-3', 'Sentinel-6MF'
sealevel_df['SeaLevel'] = sealevel_df[['TOPEX/Poseidon', 'Jason-1', 'Jason-2', 'Jason-3', 'Sentinel-6MF']].apply(lambda row: np.nanmedian(row), axis=1)
sealevel_df=sealevel_df.drop(columns=['TOPEX/Poseidon', 'Jason-1', 'Jason-2', 'Jason-3', 'Sentinel-6MF'])
# Filter CO2 data to include only observations from the first sea level observation year onward
first_sealevel_year = sealevel_df['year'].min()
co2_df = co2_df[co2_df['year'] >= first_sealevel_year]

# Interpolate CO2 values to match the SeaLevel data
sealevel_years = sealevel_df['year']
co2_interpolated = np.interp(sealevel_years, co2_df['decimal'], co2_df['average'])  # 'average' is the CO2 column

# Add the interpolated CO2 to the SeaLevel dataframe
sealevel_df['average'] = co2_interpolated

# Add 'moon_phase_sin' and 'moon_phase_cos' columns to represent the moon's position (simplified)
lunar_cycle_days = 29.53
sealevel_df['moon_phase_sin'] = np.sin(2 * np.pi * (sealevel_df['year'] % lunar_cycle_days) / lunar_cycle_days)
sealevel_df['moon_phase_cos'] = np.cos(2 * np.pi * (sealevel_df['year'] % lunar_cycle_days) / lunar_cycle_days)

# Extract the integer part of the year as the actual year and the fractional part to calculate the month
sealevel_df['actual_year'] = sealevel_df['year'].astype(int)
sealevel_df['month'] = ((sealevel_df['year'] - sealevel_df['actual_year']) * 12 + 1).astype(int)

sealevel_df['month_sin'] = np.sin(2 * np.pi * sealevel_df['month']/12)
sealevel_df['month_cos'] = np.cos(2 * np.pi * sealevel_df['month']/12)

# Reshape temperature data and merge with sealevel_df
temp_df_long = temp_df.melt(id_vars=['Year'], value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                            var_name='Month', value_name='Temperature')

# Map month names to numbers
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
temp_df_long['Month'] = temp_df_long['Month'].map(month_map)
temp_df_long.rename(columns={'Year': 'year'}, inplace=True)

# Merge temperature data with sealevel_df
sealevel_df = pd.merge(sealevel_df, temp_df_long, how='left', left_on=['actual_year', 'month'], right_on=['year', 'Month'])
sealevel_df['Temperature'] = sealevel_df['Temperature'].astype(float)

# Drop the 'Month' and 'actual_year' columns as they're redundant after merging
sealevel_df.drop(columns=['Month', 'year_y'], inplace=True)
sealevel_df.rename(columns={'year_x': 'year'}, inplace=True)

# Ensure there are no missing or infinite values in the data
sealevel_df.replace([np.inf, -np.inf], np.nan, inplace=True)
sealevel_df.dropna(inplace=True)

# Check for any missing values after all processing steps
print("Missing values after processing:\n", sealevel_df.isnull().sum())

# Define the maximum year for training
min_train_year = 0
max_train_year = 2014

# Split the data into training and test sets
train_df = sealevel_df[sealevel_df['year'] >= min_train_year]
train_df = train_df[train_df['year'] <= max_train_year]
test_df = sealevel_df[sealevel_df['year'] > max_train_year]
# train_df.columns
# future_df.columns
# Check if month column is empty
print("Training set month column value counts:\n", train_df['month'].value_counts())

# Add rows for 2100 with 'average' values of 400 and 1200 ppm, one row for each month
future_year = 2100
future_months = list(range(1, 13))
future_data = []

for co2_level in [400, 1200]:
    for month in future_months:
        future_data.append({
            'year': future_year + (month - 1) / 12,
            'month': month,
            'decimal': future_year + (month - 1) / 12,
            'average': co2_level,
            'Temperature': 10,
            'moon_phase_sin': np.sin(2 * np.pi * ((future_year + (month - 1) / 12) % lunar_cycle_days) / lunar_cycle_days),
            'moon_phase_cos': np.cos(2 * np.pi * ((future_year + (month - 1) / 12) % lunar_cycle_days) / lunar_cycle_days),
            'month_sin': np.sin(2 * np.pi * month/12),
            'month_cos': np.cos(2 * np.pi * month/12),
        })

future_df = pd.DataFrame(future_data)

future_df['Temperature'][0] = future_df['Temperature'][0] + 0.001

# Function to create and evaluate models
def create_and_evaluate_model(remove_columns=None):
    features = ['average', 'Temperature', 'month_cos','month_sin', 'moon_phase_sin', 'moon_phase_cos']
    print(features)
    # Remove specified columns from features if any
    if remove_columns is not None:
        features = [feature for feature in features if feature not in remove_columns]
    
    # Prepare the features and target variable for training
    X_train = train_df[features]
    y_train = train_df['SeaLevel']
    
    # Ensure there are no missing or infinite values in the features
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.dropna(inplace=True)
    y_train = y_train[X_train.index]  # Drop corresponding y_train values
    
    # Check if X_train contains any missing or infinite values
    print("Missing values in X_train before encoding:\n", X_train.isnull().sum())
    
    # One-hot encode the month column for training if it is included in features
    X_encoded_train = X_train
    columns = list(X_encoded_train.columns)

    # Add a constant term for the intercept
    X_encoded_train = sm.add_constant(X_encoded_train)
    columns = ['const'] + columns
    
    # Create a DataFrame for X_encoded_train with meaningful column names
    X_encoded_train_df = pd.DataFrame(X_encoded_train, columns=columns, index=X_train.index)
    # X_encoded_train_df=X_encoded_train_df.drop(columns=['month'])

    # Check for missing or infinite values in X_encoded_train_df
    print("Missing values in X_encoded_train_df:\n", X_encoded_train_df.isnull().sum())
    
    # Create and fit the OLS model
    ols_model = sm.OLS(y_train, X_encoded_train_df).fit()
    
    # Print the summary of the regression
    print(ols_model.summary())
    
    # Prepare the features for testing
    X_test = test_df[features]
    y_test = test_df['SeaLevel']
    
    # Ensure there are no missing or infinite values in the features
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.dropna(inplace=True)
    y_test = y_test[X_test.index]  # Drop corresponding y_test values
    
    # Check if X_test contains any missing or infinite values
    print("Missing values in X_test before encoding:\n", X_test.isnull().sum())
    
    # One-hot encode the month column for testing if it is included in features
    X_encoded_test = X_test
    
    # Add a constant term for the intercept
    X_encoded_test = sm.add_constant(X_encoded_test)
    
    # Create a DataFrame for X_encoded_test with meaningful column names
    X_encoded_test_df = pd.DataFrame(X_encoded_test, columns=columns, index=X_test.index)
    
    # Check for missing or infinite values in X_encoded_test_df
    print("Missing values in X_encoded_test_df:\n", X_encoded_test_df.isnull().sum())
    
    # Make predictions on the test set with OLS model
    train_df['SeaLevel_Predicted_OLS'] = ols_model.predict(X_encoded_train_df)
    test_df['SeaLevel_Predicted_OLS'] = ols_model.predict(X_encoded_test_df)
    
    # Calculate performance metrics for the OLS model
    rmse_ols = np.sqrt(mean_squared_error(y_test, test_df['SeaLevel_Predicted_OLS']))
    mae_ols = mean_absolute_error(y_test, test_df['SeaLevel_Predicted_OLS'])
    r2_ols = r2_score(y_test, test_df['SeaLevel_Predicted_OLS'])
    
    print(f"OLS Model - RMSE: {rmse_ols}")
    print(f"OLS Model - MAE: {mae_ols}")
    print(f"OLS Model - R²: {r2_ols}")
    
    # Plotting the actual vs predicted sea level for the test data with OLS model
    plt.figure(figsize=(10, 6))
    plt.scatter(train_df['year'], train_df['SeaLevel'], color='blue',  s=5,alpha=0.9)
    plt.scatter(test_df['year'], test_df['SeaLevel'], color='blue', label='Actual Sea Level', s=5,alpha=0.9)
    plt.plot(train_df['year'], train_df['SeaLevel_Predicted_OLS'], color='green', label='Train Predicted Sea Level (OLS)', alpha=0.6)
    plt.plot(test_df['year'], test_df['SeaLevel_Predicted_OLS'], color='red', label='Test Predicted Sea Level (OLS)', alpha=0.6)
    plt.xlabel('Year')
    plt.ylabel('Sea Level (mm)')
    plt.title('Actual vs Predicted Sea Level (OLS Model)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Prepare the features for future prediction
    future_X = future_df[features]
    X_encoded_future = future_X.values
    # X_encoded_future=X_encoded_future.drop(columns=['month'])
    X_encoded_future = sm.add_constant(X_encoded_future)
    
    # Make predictions on the future data
    future_predictions = ols_model.predict(X_encoded_future)
    future_df['SeaLevel_Predicted'] = future_predictions
    
    for co2_level in [400, 1200]:
        preds = future_df[future_df['average'] == co2_level]['SeaLevel_Predicted'].values
        print(f"Predicted Sea Levels at CO2 = {co2_level} ppm for each month in 2100: {preds}")

# Evaluate model with optional columns removal
# create_and_evaluate_model(remove_columns=['moon_phase_sin','moon_phase_cos','month','Temperature'])
# create_and_evaluate_model(remove_columns=['moon_phase_sin','moon_phase_cos','month'])
# create_and_evaluate_model(remove_columns=['moon_phase_sin','moon_phase_cos'])
# create_and_evaluate_model(remove_columns=['moon_phase_sin','moon_phase_cos','Temperature'])
# create_and_evaluate_model(remove_columns=['Temperature','month'])
create_and_evaluate_model(remove_columns=['month'])
# create_and_evaluate_model(remove_columns=['Temperature'])
