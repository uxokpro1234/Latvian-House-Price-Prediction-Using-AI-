<h1>Can AI Predict House Prices?</h1>
<p>I will be using Tensorflow and XGBoosting.</p>
<br>
<p>TensorFlow is mainly used for deep learning and neural networks.</p>
<br>
<p>XGBoost is mainly used for gradient boosting on structured/tabular data. I will start from XGBoost.</p>
<br>

<p><strong>Note:</strong> The CSV file used for this project was taken from <a href="https://www.kaggle.com/datasets/dmitryyemelyanov/riga-real-estate-dataset-cleaned" target="_blank">Kaggle: Riga Real Estate Dataset (cleaned)</a>.</p>

***Warning: This data is 6 years old!!!***
<br>

<h1>Explanation on <strong>construction</strong> types in csv file.</h1>

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
<br>
<h2>Why it CAN be done:</h2>
<ul>
  <li><strong>Patterns in historical data:</strong> AI can learn relationships between features (location, size, age) and prices.</li>
  <li><strong>Complex relationships:</strong> AI models capture nonlinear effects that simple regression cannot.</li>
  <li><strong>Multiple factors:</strong> AI can use many features like neighborhood, schools, crime rates, economic indicators.</li>
  <li><strong>Continuous learning:</strong> With new data and feedback, AI models can improve predictions over time.</li>
</ul>
<br>
<h2>Why it CANNOT be done perfectly:</h2>
<ul>
  <li><strong>Market volatility:</strong> Sudden economic crises, policy changes, or disasters are unpredictable.</li>
  <li><strong>Incomplete or biased data:</strong> Poor or skewed historical data leads to unreliable predictions.</li>
  <li><strong>Human factors:</strong> Emotional or speculative decisions by buyers/sellers cannot be fully modeled.</li>
  <li><strong>Overfitting risk:</strong> Models may memorize past data and fail on new properties.</li>
</ul>
<br>
<h2>Conclusion:</h2>
<p>AI can provide useful estimates and help make informed decisions, but it cannot guarantee exact future prices. It works best as a supporting tool, not a crystal ball.</p>
<br>


<h1>House Price Prediction Models</h1>
<h2>XGBoost Model</h2>
<p>Trains an XGBoost regressor to predict property prices based on numerical and categorical features. Preprocessing includes handling missing values and one-hot encoding categorical variables.</p>
<br>

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_predict
import joblib
import os

DATA_FILE = 'riga.csv'
MODEL_FILE = 'latvia_rent_model_xgb.pkl'
ENCODER_FILE = 'latvia_rent_encoder.pkl'

COLUMNS = ['listing_type','area','address','rooms','area_sqm','floor',
           'total_floors','building_type','construction','amenities',
           'price','latitude','longitude']

NUMERICAL = ['rooms','area_sqm','floor','total_floors','latitude','longitude']
CATEGORICAL = ['listing_type','area','building_type','construction','amenities']
TARGET = 'price'

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("Dataset not found")

data = pd.read_csv(DATA_FILE, header=None)

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

joblib.dump(model, MODEL_FILE)
joblib.dump(encoder, ENCODER_FILE)
print("Model trained and saved successfully")

y_pred_test = model.predict(X_test)
metrics_test = evaluate_regression(y_test.values, y_pred_test)
print("Test Set Metrics:")
for k,v in metrics_test.items():
    print(f"{k}: {v:.2f}")

y_pred_cv = cross_val_predict(model, X, y, cv=5)
metrics_cv = evaluate_regression(y, y_pred_cv)
print("\n5-Fold Cross-Validation Metrics:")
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
    enc = joblib.load(ENCODER_FILE)
    mdl = joblib.load(MODEL_FILE)
    Xc = enc.transform(df_user[CATEGORICAL])
    Xn = df_user[NUMERICAL]
    Xc_df = pd.DataFrame(Xc, columns=enc.get_feature_names_out(CATEGORICAL))
    X_final = pd.concat([Xn, Xc_df], axis=1)
    price = mdl.predict(X_final)[0]
    print(f"Predicted rent: €{price:.2f}")
```
<br>
<h2>TensorFlow Model</h2>
<p>Trains a neural network on the same features. Includes preprocessing, scaling, and early stopping for improved accuracy. Both the encoder and scaler are saved for consistent predictions.</p>
<br>

```python
import os
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = 'riga.csv'
MODEL_FILE = 'latvia_sale_model_tf.keras'
ENCODER_FILE = 'latvia_sale_encoder.pkl'
SCALER_FILE = 'latvia_sale_scaler.pkl'

COLUMNS = ['listing_type','area','address','rooms','area_sqm','floor',
           'total_floors','building_type','construction','amenities',
           'price','latitude','longitude']

NUMERICAL = ['rooms','area_sqm','floor','total_floors','latitude','longitude']
CATEGORICAL = ['listing_type','area','building_type','construction','amenities']
TARGET = 'price'

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"CSV column count mismatch: expected {len(COLUMNS)}, got {df.shape[1]}")
    df.columns = COLUMNS
    return df

def preprocess_data(df):
    df = df[df['listing_type'] == 'For sale'].copy()
    df = df.drop(columns=['address'])
    
    for col in NUMERICAL + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[NUMERICAL] = df[NUMERICAL].fillna(df[NUMERICAL].median())
    df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    return df

def encode_and_scale(df, fit=True, encoder=None, scaler=None):
    if fit:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat = encoder.fit_transform(df[CATEGORICAL])
    else:
        X_cat = encoder.transform(df[CATEGORICAL])

    if fit:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[NUMERICAL])
    else:
        X_num = scaler.transform(df[NUMERICAL])

    X = np.hstack([X_num, X_cat])
    return X, encoder, scaler

def build_tf_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.25e-4),
        loss='mse',
        metrics=['mae']
    )
    return model

def train_tf_model():
    df = load_csv(DATA_FILE)
    df = preprocess_data(df)

    y = df[TARGET].values
    X, encoder, scaler = encode_and_scale(df, fit=True)

    model = build_tf_model(X.shape[1])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    model.fit(
        X, y,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    model.save(MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("TensorFlow model, encoder, and scaler saved for 'For sale'")

def predict_sale_tf(user_input: dict) -> float:
    model = tf.keras.models.load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)

    df = pd.DataFrame([user_input])
    for col in NUMERICAL:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[NUMERICAL] = df[NUMERICAL].fillna(df[NUMERICAL].median())

    X, _, _ = encode_and_scale(df, fit=False, encoder=encoder, scaler=scaler)
    price = model.predict(X)[0][0]
    return float(price)

def update_model_with_real_price_tf(user_input: dict, real_price: float):
    df = load_csv(DATA_FILE)
    new_row = user_input.copy()
    new_row['price'] = real_price
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False, header=False)
    train_tf_model()

if __name__ == "__main__":
    train_tf_model()

    user_input = {
        'listing_type': 'For sale',
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

    predicted_price = predict_sale_tf(user_input)
    print(f"\nPredicted sale price (TF): €{predicted_price:.2f}")

    feedback = input("Do you know the real price for this property? (y/n): ").lower()
    if feedback == 'y':
        real_price = float(input("Enter the real price: "))
        update_model_with_real_price_tf(user_input, real_price)
        print(f"Model updated with new price: €{real_price}")
```
<h2>Features Used</h2>
<ul>
  <li>listing_type, area, building_type, construction, amenities</li>
  <li>rooms, area_sqm, floor, total_floors, latitude, longitude</li>
</ul>

<h2>Usage</h2>
<p>Load the model and preprocessing artifacts to predict prices for new properties.</p>

<h2>Notes</h2>
<ul>
  <li>Preprocessing: missing values handled, numerical features scaled, categorical features one-hot encoded</li>
  <li>Models: XGBoost for regression, TensorFlow neural network with dropout and early stopping</li>
  <li>Artifacts saved for consistent prediction</li>
</ul>

