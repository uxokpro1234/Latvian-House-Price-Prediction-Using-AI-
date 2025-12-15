<h1>Can AI Predict House Prices?</h1>

<p><strong>Note:</strong> The CSV file used for this project was taken from <a href="https://www.kaggle.com/datasets/dmitryyemelyanov/riga-real-estate-dataset-cleaned" target="_blank">Kaggle: Riga Real Estate Dataset (cleaned)</a>.</p>
**Warning: This data is 6 years old!!!**
<h1>Explanation on **construction** types in csv file.</h1>

<br>
<table border="1">
  <thead>
    <tr>
      <th>Value</th>
      <th>Likely Meaning / Construction Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LT proj.</td>
      <td>“Latvian Project” – typical prefabricated panel building (Soviet-era style)</td>
    </tr>
    <tr>
      <td>602.</td>
      <td>Panel building series from the 1960s–1980s</td>
    </tr>
    <tr>
      <td>P. kara</td>
      <td>“Pirmā kara” – Pre-war buildings (older masonry)</td>
    </tr>
    <tr>
      <td>Jaun.</td>
      <td>Jauna / New – Modern construction</td>
    </tr>
    <tr>
      <td>Specpr.</td>
      <td>Special project (often customized or unusual type)</td>
    </tr>
    <tr>
      <td>Hrušč.</td>
      <td>Khrushchyovka – Soviet 1–5 story panel buildings</td>
    </tr>
    <tr>
      <td>M. ģim.</td>
      <td>Mazā ģimenes – small family house, low-rise</td>
    </tr>
    <tr>
      <td>Renov.</td>
      <td>Renovated building</td>
    </tr>
    <tr>
      <td>103.</td>
      <td>Another panel building series, similar to 602</td>
    </tr>
    <tr>
      <td>Brick</td>
      <td>Brick construction</td>
    </tr>
    <tr>
      <td>Masonry</td>
      <td>Masonry construction (brick/stone)</td>
    </tr>
    <tr>
      <td>Brick-Panel</td>
      <td>Hybrid – lower floors brick, upper floors panel</td>
</table>

<h2>Why it CAN be done:</h2>
<ul>
  <li><strong>Patterns in historical data:</strong> AI can learn relationships between features (location, size, age) and prices.</li>
  <li><strong>Complex relationships:</strong> AI models capture nonlinear effects that simple regression cannot.</li>
  <li><strong>Multiple factors:</strong> AI can use many features like neighborhood, schools, crime rates, economic indicators.</li>
  <li><strong>Continuous learning:</strong> With new data and feedback, AI models can improve predictions over time.</li>
</ul>

<h2>Why it CANNOT be done perfectly:</h2>
<ul>
  <li><strong>Market volatility:</strong> Sudden economic crises, policy changes, or disasters are unpredictable.</li>
  <li><strong>Incomplete or biased data:</strong> Poor or skewed historical data leads to unreliable predictions.</li>
  <li><strong>Human factors:</strong> Emotional or speculative decisions by buyers/sellers cannot be fully modeled.</li>
  <li><strong>Overfitting risk:</strong> Models may memorize past data and fail on new properties.</li>
</ul>

<h2>Conclusion:</h2>
<p>AI can provide useful estimates and help make informed decisions, but it cannot guarantee exact future prices. It works best as a supporting tool, not a crystal ball.</p>
<br>
<h2>Okay, enough of the annoying text, lets get to coding.</h2>
<p>This code trains an XGBoost regression model to predict rent prices from a CSV dataset, using numerical and one-hot encoded categorical features, saves the model and encoder, and includes a function to predict rent interactively based on user input.</p>
<br>

```python
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

data_file = 'riga.csv'
model_file = 'latvia_rent_model_xgb.pkl'
encoder_file = 'latvia_rent_encoder.pkl'

COLUMNS = [
    'listing_type',
    'area',
    'address',
    'rooms',
    'area_sqm',
    'floor',
    'total_floors',
    'building_type',
    'construction',
    'amenities',
    'price',
    'latitude',
    'longitude'
]

NUMERICAL = [
    'rooms',
    'area_sqm',
    'floor',
    'total_floors',
    'latitude',
    'longitude'
]

CATEGORICAL = [
    'listing_type',
    'area',
    'building_type',
    'construction',
    'amenities'
]

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
print("✅ Model trained successfully")

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
    print(f"\n💰 Predicted rent: €{price:.2f}")
predict_rent()
```

```
Listing type: For sale
Area: Riga
Rooms: 3
Area sqm: 50
Floor: 3
Total floors: 5
Building type: Brick
Construction: All amenities
Amenities: LT proj.
Latitude: 56.9750922
Longitude: 24.1398842

💰 Predicted rent: €37197.15
```

<h1>Implementing flow method in previous code(Tensorflow is complicated, i dont need that)</h1>
<p>Flow is a visual programming environment where you can create AI, ML pipelines, or automation without writing tons of code. You connect “blocks” (nodes) representing data input, transformations, models, or outputs, forming a “flow” from start to finish.</p>
    <br>
    
```python
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
```

