import pandas as pd

# Read the CSV file
df = pd.read_csv('open-meteo-28.62N77.25E218m (2).csv')

# Drop the first 3 rows
df = df.iloc[3:]

# Write the DataFrame back to CSV
df.to_csv('open-meteo-28.62N77.25E218m (2).csv', index=False)