# Real Estate Price Predictor

Predicts apartment prices in Riga from CSV data. Has a CLI trainer, a Tkinter GUI,
a listing scraper (ss.lv, city24.lv, latio.lv) and a Telegram bot.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

CLI:
```
python -m src.model_training --data data --output models/best_model.pkl
```

GUI:
```
python -m src.gui
```

Pick one or more CSVs and run training. The trainer tries a few sklearn models
plus optional xgboost/lightgbm/catboost, picks the best by MAE, and writes
`models/best_model.pkl` + a `.json` with the metrics.

## CSV columns

Required: `price`, `location` (or district/city/address/street), `area`,
`rooms`, `floor`, `total_floors`, `building_type` (or house_seria/house_type),
`year`, `rental_or_sale` ("rent" / "sale").
Optional: `listed_date`, `condition`.

## Predict in code

```python
from src.predictor import predict_apartment_price

features = {
    "location": "Riga",
    "area": 50,
    "rooms": 2,
    "floor": 3,
    "total_floors": 9,
    "building_type": "Panel",
    "year": 1985,
    "rental_or_sale": "sale",
}
print(predict_apartment_price(features))
```

## Telegram bot

Put the token in `.env`:
```
TELEGRAM_BOT_TOKEN=...
```

Then `python -m src.bot`. Buttons: Predict / About. Predict supports either a
listing URL or a manual step-by-step builder. Returns current price and +1/+5/+10
year forecasts.

## Layout

```
data/      CSVs here
models/    trained .pkl saved here
src/       training, predictor, scraper, gui, bot
```
