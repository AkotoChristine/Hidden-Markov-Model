import pandas as pd

# Read the merged dataset
df = pd.read_csv('combined_test_merged.csv')

# Print unique values to debug
print("Unique subjects:", df['subject'].unique())
print("\nUnique activities:", df['activity'].unique())
print("\nUnique sessions:", df['session'].unique())

# Check the first few rows to see the format
print("\nFirst few rows:")
print(df[['subject', 'activity', 'session']].head())