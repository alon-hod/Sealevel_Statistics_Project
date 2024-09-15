import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define the file paths
co2_path = 'raw_data/co2.csv'
sealevel_path = 'raw_data/sealevel.csv'
temp_path = 'raw_data/temp.csv'
sealevel_df.to_csv('sealevel_df.csv')
# Read the CSV files
co2_df = pd.read_csv(co2_path)
sealevel_df = pd.read_csv(sealevel_path)
temp_df = pd.read_csv(temp_path, index_col=None)

# Apply nanmedian to fill missing values in 'TOPEX/Poseidon', 'Jason-1', 'Jason-2', 'Jason-3', 'Sentinel-6MF'
sealevel_df['SeaLevel'] = sealevel_df[['TOPEX/Poseidon', 'Jason-1', 'Jason-2', 'Jason-3', 'Sentinel-6MF']].apply(lambda row: np.nanmedian(row), axis=1)
sealevel_df = sealevel_df.drop(columns=['TOPEX/Poseidon', 'Jason-1', 'Jason-2', 'Jason-3', 'Sentinel-6MF'])

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

sealevel_df['month_sin'] = np.sin(2 * np.pi * sealevel_df['month'] / 12)
sealevel_df['month_cos'] = np.cos(2 * np.pi * sealevel_df['month'] / 12)

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
max_train_year = 2015

# Split the data into training and test sets
train_df = sealevel_df[sealevel_df['year'] >= min_train_year]
train_df = train_df[train_df['year'] <= max_train_year]
test_df = sealevel_df[sealevel_df['year'] > max_train_year]

# Check if month column is empty
print("Training set month column value counts:\n", train_df['month'].value_counts())

# Add rows for 2100 with 'average' values of 400 and 1200 ppm, one row for each month
future_year = 2100
future_months = list(range(1, 13))
future_data = []

for co2_level in [400, 1200]:
    for month in future_months:
        future_data.append({
            'year': future_year,
            'month': month,
            'decimal': future_year + (month - 1) / 12,
            'average': co2_level,
            'Temperature': 10,
            'moon_phase_sin': np.sin(2 * np.pi * ((future_year + (month - 1) / 12) % lunar_cycle_days) / lunar_cycle_days),
            'moon_phase_cos': np.cos(2 * np.pi * ((future_year + (month - 1) / 12) % lunar_cycle_days) / lunar_cycle_days),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
        })

future_df = pd.DataFrame(future_data)

# Prepare the data for Prophet
train_df['ds'] = pd.to_datetime(train_df['actual_year'].astype(str) + '-' + train_df['month'].astype(str).str.zfill(2), format='%Y-%m')
train_df['y'] = train_df['SeaLevel']

# Fit Prophet model
prophet_model = Prophet()
prophet_model.add_regressor('average')
prophet_model.add_regressor('Temperature')
prophet_model.add_regressor('moon_phase_sin')
prophet_model.add_regressor('moon_phase_cos')
prophet_model.add_regressor('month_sin')
prophet_model.add_regressor('month_cos')

prophet_model.fit(train_df[['ds', 'y', 'average', 'Temperature', 'moon_phase_sin', 'moon_phase_cos', 'month_sin', 'month_cos']])

# Make future predictions
future_df['ds'] = pd.to_datetime(future_df['year'].astype(str) + '-' + future_df['month'].astype(str).str.zfill(2), format='%Y-%m')
forecast = prophet_model.predict(future_df[['ds', 'average', 'Temperature', 'moon_phase_sin', 'moon_phase_cos', 'month_sin', 'month_cos']])

# Plot the forecast
prophet_model.plot(forecast)
plt.title('Future Sea Level Predictions')
plt.show()

# Print the predictions for CO2 levels 400 and 1200 ppm
for co2_level in [400, 1200]:
    preds = forecast[future_df['average'] == co2_level]['yhat'].values
    print(f"Predicted Sea Levels at CO2 = {co2_level} ppm for each month in 2100: {preds}")

# Scatter plots of actual vs predicted sea levels
# Predicting on the test data using Prophet model
test_df['ds'] = pd.to_datetime(test_df['actual_year'].astype(str) + '-' + test_df['month'].astype(str).str.zfill(2), format='%Y-%m')
test_forecast = prophet_model.predict(test_df[['ds', 'average', 'Temperature', 'moon_phase_sin', 'moon_phase_cos', 'month_sin', 'month_cos']])

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(test_df['SeaLevel'], test_forecast['yhat']))
mae = mean_absolute_error(test_df['SeaLevel'], test_forecast['yhat'])
r2 = r2_score(test_df['SeaLevel'], test_forecast['yhat'])

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Plotting the actual vs predicted sea level for the test data with Prophet model
plt.figure(figsize=(10, 6))
plt.scatter(train_df['ds'], train_df['SeaLevel'], color='blue', s=5, alpha=0.9)
plt.scatter(test_df['ds'], test_df['SeaLevel'], color='blue', s=5, alpha=0.9, label='Actual Sea Level')
plt.plot(train_df['ds'], train_df['y'], color='green', label='Train Predicted Sea Level (Prophet)', alpha=0.6)
plt.plot(test_df['ds'], test_forecast['yhat'], color='red', label='Test Predicted Sea Level (Prophet)', alpha=0.6)
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
# plt.title('Actual vs Predicted Sea Level (Prophet Model)')
plt.legend()
plt.grid(True)
plt.show()

