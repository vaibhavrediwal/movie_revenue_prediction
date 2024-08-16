import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set the path to your CSV file here
FILE_PATH = r"D:\\MTECH_PROJECT_PHASE_1\\ANOVA\\2000-2009 Movies Box Ofice Collection.csv"

def preprocess_data(df):
    # Convert currency columns to numeric
    currency_columns = ['Worldwide', 'Domestic', 'Foreign']
    for col in currency_columns:
        df[col] = df[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    
    # Convert percentage columns to float
    percentage_columns = ['Domestic_percent', 'Foreign_percent']
    for col in percentage_columns:
        df[col] = df[col].str.rstrip('%').replace('<0.1', '0.05')
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100

    # Extract year from 'Release Group' if it's not already a separate column
    if 'year' not in df.columns:
        df['year'] = df['Release Group'].str.extract(r'(\d{4})')
    
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    return df

def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def predict_revenue_rf(model, scaler, domestic, foreign, domestic_percent, foreign_percent, year, is_summer_release):
    input_data = np.array([[domestic, foreign, domestic_percent, foreign_percent, year, 
                            (foreign - domestic) / foreign, is_summer_release]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

def predict_revenue_dl(model, scaler, domestic, foreign, domestic_percent, foreign_percent, year, is_summer_release):
    input_data = np.array([[domestic, foreign, domestic_percent, foreign_percent, year, 
                            (foreign - domestic) / foreign, is_summer_release]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0][0]

# Main execution
if __name__ == "__main__":
    # Load the data
    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv(FILE_PATH)
        df = preprocess_data(df)
    except Exception as e:
        print(f"Error during data loading or preprocessing: {e}")
        print("Exiting the program.")
        exit()

    # Validate the data
    if df.isnull().any().any():
        print("Warning: The dataset contains null values. These will be removed.")
        df = df.dropna()

    # Check for infinite values
    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        print("Warning: The dataset contains infinite values. These will be removed.")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Print data info
    print("\nDataset Info:")
    print(df.info())

    print("\nSample of preprocessed data:")
    print(df.head())

    # Exploratory Data Analysis
    print("\nPerforming Exploratory Data Analysis...")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='year', y='Worldwide', hue='Domestic_percent', palette='viridis')
    plt.title('Worldwide Box Office by Year')
    plt.savefig('worldwide_boxoffice_trend.png')
    plt.close()

    # Feature Engineering
    print("Engineering features...")
    df['profit_margin'] = (df['Worldwide'] - df['Domestic']) / df['Worldwide']
    df['is_summer_release'] = df['Release Group'].str.contains('Jun|Jul|Aug').astype(int)

    # Prepare features and target
    features = ['Domestic', 'Foreign', 'Domestic_percent', 'Foreign_percent', 'year', 'profit_margin', 'is_summer_release']
    X = df[features]
    y = df['Worldwide']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    rf_pred = rf_model.predict(X_test_scaled)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    print("Random Forest Results:")
    print(f"Mean Squared Error: {rf_mse}")
    print(f"R-squared Score: {rf_r2}")

    # Cross-validation for Random Forest
    rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"Random Forest Cross-validation scores: {rf_cv_scores}")
    print(f"Mean CV score: {np.mean(rf_cv_scores)}")

    # Deep Learning Model
    print("Training Deep Learning model...")
    dl_model = create_model(X_train_scaled.shape[1])

    # Train the deep learning model
    history = dl_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=0
    )

    # Evaluate the deep learning model
    dl_pred = dl_model.predict(X_test_scaled).flatten()
    dl_mse = mean_squared_error(y_test, dl_pred)
    dl_r2 = r2_score(y_test, dl_pred)

    print("Deep Learning Results:")
    print(f"Mean Squared Error: {dl_mse}")
    print(f"R-squared Score: {dl_r2}")

    # Visualize training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Deep Learning Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('dl_training_history.png')
    plt.close()

    # Compare predictions
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, rf_pred, alpha=0.5, label='Random Forest')
    plt.scatter(y_test, dl_pred, alpha=0.5, label='Deep Learning')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Worldwide Revenue')
    plt.ylabel('Predicted Worldwide Revenue')
    plt.title('Actual vs Predicted Revenue')
    plt.legend()
    plt.savefig('actual_vs_predicted_revenue.png')
    plt.close()

    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance (Random Forest)')
    plt.savefig('feature_importance.png')
    plt.close()

    # Example prediction
    print("\nMaking example predictions...")
    example_input = [100000000, 200000000, 0.33, 0.67, 2023, 1]
    rf_prediction = predict_revenue_rf(rf_model, scaler, *example_input)
    dl_prediction = predict_revenue_dl(dl_model, scaler, *example_input)

    print(f"Random Forest Prediction: ${rf_prediction:,.2f}")
    print(f"Deep Learning Prediction: ${dl_prediction:,.2f}")

    print("\nAnalysis complete. Check the generated PNG files for visualizations.")