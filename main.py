import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
from prefect import flow, task

# CONFIG
DATA_FILE = 'riga.csv'
MODEL_FILE = 'latvia_rent_model_xgb.pkl'
ENCODER_FILE = 'latvia_rent_encoder.pkl'

COLUMNS = [
    'listing_type','area','address','rooms','area_sqm','floor','total_floors',
    'building_type','construction','amenities','price','latitude','longitude'
]
NUMERICAL = ['rooms','area_sqm','floor','total_floors','latitude','longitude']
CATEGORICAL = ['listing_type','area','building_type','construction','amenities']
TARGET = 'price'

# TASKS
@task
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found")
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"CSV column count mismatch: expected {len(COLUMNS)}, got {df.shape[1]}")
    df.columns = COLUMNS
    return df

@task
def preprocess_data(df):
    df = df.drop(columns=['address'])
    for col in NUMERICAL + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[NUMERICAL] = df[NUMERICAL].fillna(df[NUMERICAL].median())
    df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    return df

@task
def encode_categorical(df):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = encoder.fit_transform(df[CATEGORICAL])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(CATEGORICAL))
    return encoder, X_cat_df

@task
def prepare_features(df, X_cat_df):
    X_num = df[NUMERICAL].reset_index(drop=True)
    X = pd.concat([X_num, X_cat_df], axis=1)
    y = df[TARGET]
    return X, y

@task
def train_model(X, y):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)
    return model

@task
def save_artifacts(model, encoder):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    print("Model and encoder saved")

@task
def predict_rent(user_input):
    df = pd.DataFrame([user_input])
    encoder = joblib.load(ENCODER_FILE)
    model = joblib.load(MODEL_FILE)
    X_cat = encoder.transform(df[CATEGORICAL])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(CATEGORICAL))
    X_num = df[NUMERICAL]
    X_final = pd.concat([X_num, X_cat_df], axis=1)
    price = model.predict(X_final)[0]
    return price

@task
def update_model_with_real_price(user_input, real_price):
    """Append new data and retrain model incrementally"""
    df = pd.read_csv(DATA_FILE, header=None)
    df.columns = COLUMNS
    new_row = user_input.copy()
    new_row['price'] = real_price
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False, header=False)
    # retrain model
    df = preprocess_data(df)
    encoder, X_cat_df = encode_categorical(df)
    X, y = prepare_features(df, X_cat_df)
    model = train_model(X, y)
    save_artifacts(model, encoder)
    print(f"Model updated with new price: €{real_price}")

# FLOW
@flow
def rent_price_flow():
    # Training
    df = load_csv(DATA_FILE)
    df = preprocess_data(df)
    encoder, X_cat_df = encode_categorical(df)
    X, y = prepare_features(df, X_cat_df)
    model = train_model(X, y)
    save_artifacts(model, encoder)

    # Prediction
    user_input = {
        'listing_type': input("Listing type: "),
        'area': input("Area: "),
        'rooms': float(input("Rooms: ")),
        'area_sqm': float(input("Area sqm: ")),
        'floor': float(input("Floor: ")),
        'total_floors': float(input("Total floors: ")),
        'building_type': input("Building type: "),
        'construction': input("Construction: "),
        'amenities': input("Amenities: "),
        'latitude': float(input("Latitude: ")),
        'longitude': float(input("Longitude: "))
    }
    predicted_price = predict_rent(user_input)
    print(f"\n💰 Predicted rent: €{predicted_price:.2f}")

    # ask user if they know the real price
    feedback = input("Do you know the real price for this property? (y/n): ").lower()
    if feedback == 'y':
        real_price = float(input("Enter the real price: "))
        update_model_with_real_price(user_input, real_price)

if __name__ == "__main__":
    rent_price_flow()
