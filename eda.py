import pandas as pd

print("Loading data...")
try:
    # Load first 1000 rows to understand the structure quickly
    df = pd.read_csv('DataSet.csv', nrows=1000)
    print("Columns:", list(df.columns))
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    
    potential_targets = [col for col in df.columns if 'fraud' in col.lower() or 'label' in col.lower() or 'target' in col.lower() or 'is_mule' in col.lower()]
    print("\nPotential target columns:", potential_targets)
    
    if potential_targets:
        # Load the full column just to see the distribution
        full_df = pd.read_csv('DataSet.csv', usecols=potential_targets)
        for col in potential_targets:
            print(f"\nValue counts for {col} in full dataset:")
            print(full_df[col].value_counts(dropna=False))
            
except Exception as e:
    print("Error:", e)
