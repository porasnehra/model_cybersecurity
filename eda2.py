import pandas as pd

try:
    df = pd.read_csv('DataSet.csv', usecols=['F3920', 'F3921', 'F3922', 'F3923', 'F3924'])
    print("Value counts for last 5 columns:")
    for col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False))
except Exception as e:
    print("Error:", e)
