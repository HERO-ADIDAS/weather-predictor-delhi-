import pandas as pd

# Read the CSV file, skipping the first 3 rows
df = pd.read_csv('open-meteo-28.62N77.25E218m (2).csv', skiprows=3)

# Write the DataFrame back to CSV
df.to_csv('open-meteo-28.62N77.25E218m (2).csv', index='time')