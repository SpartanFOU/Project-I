import sys
sys.path.insert(0, r"c:\Projects Python\Project-I\Project-I\Project_I\src")

import os
os.chdir(r"c:\Projects Python\Project-I\Project-I\Project_I")

from project_i.data_loader import load_energy_data

# Update this path to point to your CSV file
DATA_PATH = "data/HistorianTable.csv"

df = load_energy_data(DATA_PATH)

print("=== INFO ===")
print(df.info())
print("\n=== HEAD ===")
print(df.head())
print("\n=== DESCRIBE ===")
print(df.describe())
df.to_csv("data/clean_energy_data.csv", index=True)