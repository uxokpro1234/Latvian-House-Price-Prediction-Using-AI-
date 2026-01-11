import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

data_file = 'riga.csv'
model_file = 'latvia_rent_model_xgb.pkl'
encoder_file = 'latvia_rent_encoder.pkl'

COLUMNS = ['listing_type','area','address','rooms','area_sqm','floor','total_floors','building_type','construction','amenities','price','latitude','longitude']

NUMERICAL = ['rooms','area_sqm','floor','total_floors','latitude','longitude']

CATEGORICAL = ['listing_type','area','building_type','construction','amenities']

TARGET = 'price'

if not os.path.exists(data_file):
    raise FileNotFoundError("Dataset not found")

data = pd.read_csv(data_file, header=None)

if data.shape[1] != len(COLUMNS):
    raise ValueError(
        f"CSV column count mismatch: expected {len(COLUMNS)}, got {data.shape[1]}"
    )

data.columns = COLUMNS
data = data.drop(columns=['address'])
for col in NUMERICAL + [TARGET]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data[NUMERICAL] = data[NUMERICAL].fillna(data[NUMERICAL].median())
data[TARGET] = data[TARGET].fillna(data[TARGET].median())
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = encoder.fit_transform(data[CATEGORICAL])
X_num = data[NUMERICAL].reset_index(drop=True)
X_cat_df = pd.DataFrame(
    X_cat,
    columns=encoder.get_feature_names_out(CATEGORICAL)
)

X = pd.concat([X_num, X_cat_df], axis=1)
y = data[TARGET]

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, model_file)
joblib.dump(encoder, encoder_file)
print("Model trained successfully")

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

    df = pd.DataFrame([user])
    enc = joblib.load(encoder_file)
    mdl = joblib.load(model_file)
    Xc = enc.transform(df[CATEGORICAL])
    Xn = df[NUMERICAL]
    Xc_df = pd.DataFrame(
        Xc,
        columns=enc.get_feature_names_out(CATEGORICAL)
    )

    X_final = pd.concat([Xn, Xc_df], axis=1)
    price = mdl.predict(X_final)[0]
    print(f"\nPredicted rent: €{price:.2f}")
predict_rent()
