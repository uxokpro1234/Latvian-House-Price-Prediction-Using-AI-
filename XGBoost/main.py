import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_predict
import joblib
import os

data_file = 'riga.csv'
model_file = 'latvia_rent_model_xgb.pkl'
encoder_file = 'latvia_rent_encoder.pkl'

COLUMNS = ['listing_type','area','address','rooms','area_sqm','floor',
           'total_floors','building_type','construction','amenities',
           'price','latitude','longitude']

NUMERICAL = ['rooms','area_sqm','floor','total_floors','latitude','longitude']
CATEGORICAL = ['listing_type','area','building_type','construction','amenities']
TARGET = 'price'

if not os.path.exists(data_file):
    raise FileNotFoundError("Dataset not found")

data = pd.read_csv(data_file, header=None)

if data.shape[1] != len(COLUMNS):
    raise ValueError(f"CSV column count mismatch: expected {len(COLUMNS)}, got {data.shape[1]}")

data.columns = COLUMNS
data = data.drop(columns=['address'])

for col in NUMERICAL + [TARGET]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data[NUMERICAL] = data[NUMERICAL].fillna(data[NUMERICAL].median())
data[TARGET] = data[TARGET].fillna(data[TARGET].median())

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = encoder.fit_transform(data[CATEGORICAL])
X_num = data[NUMERICAL].reset_index(drop=True)
X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(CATEGORICAL))

X = pd.concat([X_num, X_cat_df], axis=1)
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_regression(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    if len(y_true) < 2:
        r2 = float('nan')
    else:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
    
    return {"R²": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

joblib.dump(model, model_file)
joblib.dump(encoder, encoder_file)
print("Model trained and saved successfully")

y_pred_test = model.predict(X_test)
metrics_test = evaluate_regression(y_test.values, y_pred_test)
print("📊 Test Set Metrics:")
for k,v in metrics_test.items():
    print(f"{k}: {v:.2f}")

y_pred_cv = cross_val_predict(model, X, y, cv=5)
metrics_cv = evaluate_regression(y, y_pred_cv)
print("\n📊 5-Fold Cross-Validation Metrics:")
for k,v in metrics_cv.items():
    print(f"{k}: {v:.2f}")

def predict_rent():
    print("\nEnter rental details\n")
    user = {
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

    df_user = pd.DataFrame([user])
    enc = joblib.load(encoder_file)
    mdl = joblib.load(model_file)
    Xc = enc.transform(df_user[CATEGORICAL])
    Xn = df_user[NUMERICAL]
    Xc_df = pd.DataFrame(Xc, columns=enc.get_feature_names_out(CATEGORICAL))
    X_final = pd.concat([Xn, Xc_df], axis=1)
    price = mdl.predict(X_final)[0]
    print(f"Predicted rent: €{price:.2f}")