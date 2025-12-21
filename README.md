<h1>Can AI Predict House Prices?</h1>
<p>I will be using Tensorflow and XGBoosting.</p>
<br>
TensorFlow is mainly used for deep learning and neural networks.
<br>
XGBoost is mainly used for gradient boosting on structured/tabular data.
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
<br>
<h1>Lets look at the output:</h1>
<p>Ive found a house for sale from 4 years ago, its price was €61500. So yea it did a good job. But I AM NOT finished yeat. Hear me out. </p>]
<br>

```
✅ Model trained successfully

Enter rental details

Listing type: For sale
Area: Riga
Rooms: 3
Area sqm: 50
Floor: 3
Total floors: 5
Building type: Brick
Construction: Hrušč.
Amenities: All amentities
Latitude: 56.9750922
Longitude: 24.1398842

💰 Predicted rent: €61743.69
```
<br>
<h1>Improving Prediction Accuracy</h1>

<p>
Prediction accuracy is improved by using a TensorFlow-based model instead of a visual flow approach.
The pipeline is fully code-driven, giving precise control over preprocessing, training, and inference.
</p>

<p>
Numerical features are scaled, categorical features are one-hot encoded, and a neural network
is trained for rent price prediction. All preprocessing artifacts are saved to ensure
consistent results during prediction.
</p>

<br>
    
```python
import os
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# CONFIG
DATA_FILE = 'riga.csv'
MODEL_FILE = 'latvia_rent_model_tf.keras'
ENCODER_FILE = 'latvia_rent_encoder.pkl'
SCALER_FILE = 'latvia_rent_scaler.pkl'

COLUMNS = ['listing_type','area','address','rooms','area_sqm','floor',
           'total_floors','building_type','construction','amenities',
           'price','latitude','longitude']

NUMERICAL = ['rooms','area_sqm','floor','total_floors','latitude','longitude']
CATEGORICAL = ['listing_type','area','building_type','construction','amenities']
TARGET = 'price'

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found")
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"CSV column count mismatch: expected {len(COLUMNS)}, got {df.shape[1]}")
    df.columns = COLUMNS
    return df

def preprocess_data(df):
    df = df.drop(columns=['address'])
    for col in NUMERICAL + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[NUMERICAL] = df[NUMERICAL].fillna(df[NUMERICAL].median())
    df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    return df

def encode_and_scale(df, fit=True, encoder=None, scaler=None):
    # categorial
    if fit:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat = encoder.fit_transform(df[CATEGORICAL])
    else:
        X_cat = encoder.transform(df[CATEGORICAL])

    # numeric
    if fit:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[NUMERICAL])
    else:
        X_num = scaler.transform(df[NUMERICAL])

    X = pd.DataFrame(
        data = np.hstack([X_num, X_cat]),
        columns = [f"num_{c}" for c in NUMERICAL] + list(encoder.get_feature_names_out(CATEGORICAL))
    )

    return X, encoder, scaler

def build_tf_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # regression
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def train_tf_model():
    df = load_csv(DATA_FILE)
    df = preprocess_data(df)

    # introducing X, y
    X, encoder, scaler = encode_and_scale(df, fit=True)
    y = df[TARGET].values

    # build, and learning
    model = build_tf_model(X.shape[1])
    model.fit(X.values, y, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    # saving artiffacts
    model.save(MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("TensorFlow model, encoder and scaler saved")

def predict_rent_tf(user_input: dict) -> float:
    # loading artiffacts
    model = tf.keras.models.load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)

    df = pd.DataFrame([user_input])
    # types
    for col in NUMERICAL:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[NUMERICAL] = df[NUMERICAL].fillna(df[NUMERICAL].median())

    # coding and scaling (без fit)
    X, _, _ = encode_and_scale(df, fit=False, encoder=encoder, scaler=scaler)
    price = float(model.predict(X.values)[0][0])
    return price

def update_model_with_real_price_tf(user_input, real_price):
    df = pd.read_csv(DATA_FILE, header=None)
    df.columns = COLUMNS
    new_row = user_input.copy()
    new_row['price'] = real_price
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False, header=False)

    train_tf_model()

if __name__ == "__main__":
    import numpy as np

    # learning
    train_tf_model()

    # simple CLI-predict
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

    predicted_price = predict_rent_tf(user_input)
    print(f"\n💰 Predicted rent (TF): €{predicted_price:.2f}")

    feedback = input("Do you know the real price for this property? (y/n): ").lower()
    if feedback == 'y':
        real_price = float(input("Enter the real price: "))
        update_model_with_real_price_tf(user_input, real_price)
        print(f"Model updated with new price: €{real_price}")
```
<br>
<h1>Heres the output, it became more realistic(closer to the real price).</h1>
<p>Tensorflow has epochs. An epoch - one full pass over the entire training dataset.</p>
<ul>
  <li>The model sees all training data 50 times.</li>
  <li>Each time, it slightly adjusts its weights to reduce error.</li>
</ul>
<p>So, in real life the more you do sth, the better you become in it. BUT THAT doesnt work with epohs. Ive read that to further improve the code, i have to add EarlyStop, which adjusts the number of echoes, as long as the model improves. When it stops improving, it stops training basically.</p>
<br>
<h1>Ive made other significant improvements, which made the code better.</h1>
<br>

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

```python
model.fit(
    X.values, y,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)
```
<p>This code stops epohs when the logaritm stops improving. So instead of wasting time, and space on my hard drive, i've added this.</p>
<br>
<h1>Also i log-transformed the target (price).</h1>
<p>Rent prices are usually skewed, so we have to IMPROVE regression as much as we can. Later it will predict better.</p>
<p>I've replaced THIS</p>

```python
y = df[TARGET].values
```
<br>
<p>With</p>

```python
y = np.log1p(df[TARGET].values)
```
<br>
<p>And prediction line too</p>

```python
    log_price = model.predict(X.values)[0][0]
```
<p>Improvement of learning speed</p>

```python
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)
```
<br>
<h1>Neuron dropout</h1>
<h3>3. Why 30% for first layer, 20% for second layer?</h3>

<p>First layer sees raw features, so it has more neurons → higher dropout (0.3) helps prevent over-reliance on any single input feature.</p>

<p>Second layer is smaller (64 neurons) → we reduce dropout (0.2) because:</p>
<ul>
  <li>Fewer neurons, already a bottleneck.</li>
  <li>Too much dropout could destroy learning in the smaller layer.</li>
</ul>

<p><strong>Rule of thumb:</strong></p>
<ul>
  <li>Larger layers → slightly higher dropout</li>
  <li>Smaller layers → lower dropout</li>
</ul>

```python
def build_tf_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
```

<br>
<h1>After all of the coding, i have a final product. For now.</h1>

```python
import os
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# CONFIG
DATA_FILE = 'riga.csv'
MODEL_FILE = 'latvia_rent_model_tf.keras'
ENCODER_FILE = 'latvia_rent_encoder.pkl'
SCALER_FILE = 'latvia_rent_scaler.pkl'

COLUMNS = ['listing_type','area','address','rooms','area_sqm','floor',
           'total_floors','building_type','construction','amenities',
           'price','latitude','longitude']

NUMERICAL = ['rooms','area_sqm','floor','total_floors','latitude','longitude']
CATEGORICAL = ['listing_type','area','building_type','construction','amenities']
TARGET = 'price'

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found")
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"CSV column count mismatch: expected {len(COLUMNS)}, got {df.shape[1]}")
    df.columns = COLUMNS
    return df

def preprocess_data(df):
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

    X = pd.DataFrame(
        data=np.hstack([X_num, X_cat]),
        columns=[f"num_{c}" for c in NUMERICAL] + list(encoder.get_feature_names_out(CATEGORICAL))
    )
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

def train_tf_model():
    df = load_csv(DATA_FILE)
    df = preprocess_data(df)

    # log-transform target to stabilize skewed prices
    y = np.log1p(df[TARGET].values)

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
        X.values, y,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # save everything
    model.save(MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("TensorFlow model, encoder, and scaler saved")

def predict_rent_tf(user_input: dict) -> float:
    model = tf.keras.models.load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)

    df = pd.DataFrame([user_input])
    for col in NUMERICAL:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[NUMERICAL] = df[NUMERICAL].fillna(df[NUMERICAL].median())

    X, _, _ = encode_and_scale(df, fit=False, encoder=encoder, scaler=scaler)
    log_price = model.predict(X.values)[0][0]
    price = np.expm1(log_price)  # invert log1p transform
    return float(price)

def update_model_with_real_price_tf(user_input, real_price):
    df = pd.read_csv(DATA_FILE, header=None)
    df.columns = COLUMNS
    new_row = user_input.copy()
    new_row['price'] = real_price
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False, header=False)
    train_tf_model()

if __name__ == "__main__":
    train_tf_model()

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

    predicted_price = predict_rent_tf(user_input)
    print(f"\n💰 Predicted rent (TF): €{predicted_price:.2f}")

    feedback = input("Do you know the real price for this property? (y/n): ").lower()
    if feedback == 'y':
        real_price = float(input("Enter the real price: "))
        update_model_with_real_price_tf(user_input, real_price)
        print(f"Model updated with new price: €{real_price}")
```

