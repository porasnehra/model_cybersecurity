import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import joblib
import sys

try:
    print("Loading dataset...")
    df = pd.read_csv('DataSet.csv')

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    print("Preparing data...")
    X = df.drop(columns=['F3924'])
    y = df['F3924']
    
    # Drop string/object columns to avoid conversion errors
    numeric_cols = X.select_dtypes(include=['number']).columns
    X = X[numeric_cols]
    
    print(f"Using {len(numeric_cols)} numeric features.")
    joblib.dump(list(numeric_cols), 'numeric_features.pkl')

    print("Handling missing values...")
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = imputer.fit_transform(X)

    print("Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Random Forest model with Anti-Overfitting constraints...")
    # Added max_depth and min_samples_leaf to prevent the model from memorizing the data
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,               # Restrict how deep the tree can grow
        min_samples_leaf=4,         # Require at least 4 samples in a leaf node
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)

    print("\n--- Training Set Evaluation (Checking for Overfitting) ---")
    y_train_pred = clf.predict(X_train)
    print(classification_report(y_train, y_train_pred))

    print("\n--- Test Set Evaluation ---")
    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, y_test_pred))

    print("\nSaving model to disk...")
    joblib.dump(clf, 'mule_account_model.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    print("Done! Anti-overfit Model saved.")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
