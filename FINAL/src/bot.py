import os
import sys
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")

import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import PricePredictor
from src.scraper import parse_listing_url, SUPPORTED_DOMAINS


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


MODE_SELECT, AWAITING_URL, MANUAL_FIELD = range(3)


def get_predictor(model_path: Optional[str] = None) -> PricePredictor:
    path = Path(model_path) if model_path else PROJECT_ROOT / "models" / "best_model.pkl"
    return PricePredictor(path)


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([["Predict"], ["About"]], resize_keyboard=True)


import html


def _trunc(text: any, max_len: int = 150) -> str:
    """Escape HTML and truncate text."""
    if text is None:
        return ""
    s = str(text).strip()
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return html.escape(s)


def _format_response(result) -> str:
    lines = []

    FEATURE_LABELS = {
        "area": "Total Area (m²)",
        "rooms": "Number of Rooms",
        "age": "Building Age",
        "floor": "Floor Level",
        "total_floors": "Total Floors",
        "distance_from_center": "Distance to City Center (km)",
        "loc_smooth_price": "District Average Price",
        "building_type": "Building Series/Type",
        "condition_age_score": "Condition & Age Factor",
        "is_top_floor": "Top Floor Premium/Discount",
        "is_ground_floor": "Ground Floor Factor",
        "area_per_room": "Sq.m per Room",
        "rooms_x_area": "Room/Size Interaction",
        "area_log": "Size Scaling (Log)",
        "lat": "Latitude",
        "lon": "Longitude"
    }

    lines.append("🎯 <b>Price Estimate</b>")
    lines.append(f"Market Value: <b>{result.current_price:,.0f} EUR</b>")
    if result.scraped_price:
        delta = result.current_price - result.scraped_price
        diff_text = "below" if delta > 0 else "above"
        lines.append(f"Listing Price: {result.scraped_price:,.0f} EUR (<i>{abs(delta):,.0f} EUR {diff_text} market</i>)")
    
    lines.append("\n" + "─" * 20)

    lines.append("🏠 <b>Property Specifications</b>")

    loc_val = _trunc(result.location or "Unknown", 100)
    if result.street:
        loc_val += f", {_trunc(result.street, 100)}"
    lines.append(f"📍 Location: {loc_val}")

    specs = []
    if result.rooms:
        specs.append(f"{result.rooms} rooms")
    if result.area:
        specs.append(f"{result.area} m²")
    if result.floor:
        floor_str = f"{result.floor}"
        if result.total_floors:
            floor_str += f"/{result.total_floors}"
        specs.append(f"Floor {floor_str}")
    if specs:
        lines.append(f"📐 Specs: {' &#8226; '.join(specs)}")

    build = []
    if result.building_type:
        build.append(_trunc(result.building_type, 50))
    if result.year:
        build.append(f"Built {result.year}")
    if hasattr(result, "condition") and result.condition:
        build.append(_trunc(result.condition, 50))
    if build:
        lines.append(f"🏢 Building: {' &#8226; '.join(build)}")

    lines.append("\n" + "─" * 20)

    if hasattr(result, "explanation") and result.explanation:
        lines.append("📊 <b>Price Drivers</b>")
        sorted_exp = sorted(result.explanation.items(), key=lambda x: abs(x[1]), reverse=True)
        top_factors = sorted_exp[:5]
        for name, val in top_factors:
            label = _trunc(FEATURE_LABELS.get(name, str(name)), 40)
            sign = "+" if val > 0 else "-"
            lines.append(f"&#8226; {label}: {sign}{abs(val):,.0f} EUR")
        
        lines.append("\n" + "─" * 20)

    lines.append("📈 <b>Value Forecast</b>")
    lines.append(f"📆 +1 Year: <b>{result.price_1y:,.0f} EUR</b>")
    lines.append(f"📅 +5 Years: <b>{result.price_5y:,.0f} EUR</b>")
    lines.append(f"⏳ +10 Years: <b>{result.price_10y:,.0f} EUR</b>")
    
    lines.append("\n<i>Forecast based on historical Riga growth rates.</i>")
    
    resp_str = "\n".join(lines)
    if len(resp_str) > 4000:
        resp_str = resp_str[:3997] + "..."
    return resp_str


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Welcome! I help estimate apartment prices in Latvia. Use the buttons below to get started.",
        reply_markup=main_keyboard(),
    )
    return ConversationHandler.END


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    about_text = (
        "🤖 <b>About this Bot</b>\n"
        "This bot is designed to analyze real estate market data in Riga and provide price estimates based on property characteristics. "
        "It uses advanced machine learning models to predict current market value and forecast future price trends (+1, +5, +10 years).\n\n"
        
        "👨‍💻 <b>Creators</b>\n"
        "&#8226; Ruslans Popakuls\n"
        "&#8226; Markuss Muiznieks\n\n"
        
        "📊 <b>Data & Training</b>\n"
        "The AI was trained using a dataset of Riga property listings. The process involved:\n"
        "&#8226; <b>Feature Engineering</b>: calculating distances from the center and district-specific price statistics.\n"
        "&#8226; <b>Algorithms</b>: Comparison of various models (RandomForest, XGBoost, CatBoost), with <b>HistGradientBoosting</b> showing the best performance.\n"
        "&#8226; <b>Target Transformation</b>: Using logarithmic scaling for prices to handle market variance.\n\n"
        
        "🧪 <b>Model Performance (Cross-Validation)</b>\n"
        "&#8226; <b>R² Score</b>: 0.92 (92% of price variance explained)\n"
        "&#8226; <b>MAE</b>: ~8,984 EUR (Average prediction error)\n\n"
        
        "📅 <b>Dataset Info</b>\n"
        "The primary dataset (riga_re.csv) was originally sourced from the internet. <b>Last updated approx. 6 years ago.</b> "
        "Current predictions are adjusted based on historical growth rates.\n\n"
        
        "⚠️ <b>Disclaimer</b>\n"
        "This bot was created as part of a research project (<i>ZPD</i>) for <i>Riga Secondary School No. 80</i>. "
        "It is intended for educational and research purposes only and may show inaccurate values. "
        "Always consult a professional for real estate decisions."
    )
    
    await update.message.reply_text(
        about_text,
        reply_markup=main_keyboard(),
        parse_mode="HTML"
    )
    return ConversationHandler.END


async def predict_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Choose how to predict:\n- Send listing URL\n- Build manually",
        reply_markup=ReplyKeyboardMarkup([["Send URL"], ["Manual Builder"], ["Cancel"]], resize_keyboard=True),
    )
    return MODE_SELECT


def _validate_url(url: str) -> bool:
    lowered = url.lower()
    return any(domain in lowered for domain in SUPPORTED_DOMAINS)


BUILDER_FIELDS = [
    ("rental_or_sale", "Choose deal type:", ["sale", "rent"]),
    ("location", "Enter city/district (e.g., Riga, Jurmala):", None),
    ("area", "Enter area in m2:", ["30", "40", "50", "60", "80", "100"]),
    ("rooms", "Enter number of rooms:", ["1", "2", "3", "4", "5"]),
    ("floor", "Enter floor number:", ["1", "2", "3", "4", "5", "6", "7", "8", "9"]),
    ("total_floors", "Enter total floors:", ["2", "3", "4", "5", "6", "7", "8", "9", "12", "16"]),
    ("building_type", "Enter building type (e.g., Panel, Brick, New project):", ["Panel", "Brick", "New project"]),
    ("year", "Enter build year:", ["1960", "1975", "1990", "2005", "2015", "2022"]),
]


async def _ask_field(update: Update, field_idx: int) -> int:
    key, prompt, options = BUILDER_FIELDS[field_idx]
    if options:
        kb = ReplyKeyboardMarkup([options[i : i + 3] for i in range(0, len(options), 3)] + [["Skip", "Cancel"]], resize_keyboard=True)
    else:
        kb = ReplyKeyboardMarkup([["Skip", "Cancel"]], resize_keyboard=True)
    await update.message.reply_text(f"{prompt}", reply_markup=kb)
    return MANUAL_FIELD


def _store_answer(context: ContextTypes.DEFAULT_TYPE, field_idx: int, text: str) -> None:
    key, _, _ = BUILDER_FIELDS[field_idx]
    val: Optional[object] = text.strip()
    if val.lower() == "skip":
        val = None
    else:
        if key in {"area", "rooms", "floor", "total_floors", "year"}:
            try:
                val = float(val)
                if key in {"rooms", "floor", "total_floors", "year"}:
                    val = int(val)
            except Exception:
                val = None
        if key == "rental_or_sale":
            val = val.lower()
    builder = context.user_data.setdefault("builder", {})
    builder[key] = val


async def handle_mode_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    choice = update.message.text.strip().lower()
    context.user_data.pop("builder", None)
    context.user_data.pop("builder_idx", None)
    if choice == "send url":
        await update.message.reply_text(
            "Please send the listing URL (ss.lv/ss.com, city24.lv, or latio.lv).",
            reply_markup=ReplyKeyboardRemove(),
        )
        return AWAITING_URL
    if choice == "manual builder":
        context.user_data["builder_idx"] = 0
        return await _ask_field(update, 0)
        
    await update.message.reply_text("Canceled.", reply_markup=main_keyboard())
    return ConversationHandler.END


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    url = update.message.text.strip()
    if not _validate_url(url):
        await update.message.reply_text("Unsupported URL. Please send ss.lv, city24.lv, or latio.lv.", reply_markup=main_keyboard())
        return ConversationHandler.END

    await update.message.reply_text("Fetching listing data...")
    try:
        listing = parse_listing_url(url)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scraping failed")
        await update.message.reply_text(f"Could not parse listing: {exc}", reply_markup=main_keyboard())
        return ConversationHandler.END

    try:
        predictor = get_predictor()
    except FileNotFoundError:
        await update.message.reply_text(
            "Model file missing. Train a model first (GUI or CLI) to create models/best_model.pkl.",
            reply_markup=main_keyboard(),
        )
        return ConversationHandler.END
    except Exception as exc:  # noqa: BLE001
        logger.exception("Predictor load failed")
        await update.message.reply_text(f"Failed to load model: {exc}", reply_markup=main_keyboard())
        return ConversationHandler.END

    try:
        result = predictor.predict_with_horizons(listing.to_feature_dict())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        await update.message.reply_text(f"Prediction failed: {exc}", reply_markup=main_keyboard())
        return ConversationHandler.END

    response = _format_response(result)
    await update.message.reply_text(response, reply_markup=main_keyboard(), parse_mode="HTML")

    if result.lat and result.lon:
        await update.message.reply_location(latitude=result.lat, longitude=result.lon)

    if result.images:
        from telegram import InputMediaPhoto
        media_group = [InputMediaPhoto(media=img) for img in result.images[:10]]
        try:
            await update.message.reply_media_group(media=media_group)
        except Exception as e:
            logger.warning(f"Failed to send images: {e}")
            await update.message.reply_text(f"Found {len(result.images)} images but could not send them.")

    return ConversationHandler.END


async def handle_manual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    if text.lower() == "cancel":
        return await cancel(update, context)

    idx = context.user_data.get("builder_idx", 0)
    _store_answer(context, idx, text)
    idx += 1
    if idx >= len(BUILDER_FIELDS):
        builder = context.user_data.get("builder", {})
        try:
            predictor = get_predictor()
        except FileNotFoundError:
            await update.message.reply_text(
                "Model file missing. Train a model first (GUI or CLI) to create models/best_model.pkl.",
                reply_markup=main_keyboard(),
            )
            return ConversationHandler.END
        except Exception as exc:  # noqa: BLE001
            logger.exception("Predictor load failed")
            await update.message.reply_text(f"Failed to load model: {exc}", reply_markup=main_keyboard())
            return ConversationHandler.END

        try:
            result = predictor.predict_with_horizons(builder)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed")
            await update.message.reply_text(f"Prediction failed: {exc}", reply_markup=main_keyboard())
            return ConversationHandler.END

        response = _format_response(result)
        await update.message.reply_text(response, reply_markup=main_keyboard(), parse_mode="HTML")
        context.user_data.pop("builder", None)
        context.user_data.pop("builder_idx", None)
        return ConversationHandler.END

    context.user_data["builder_idx"] = idx
    return await _ask_field(update, idx)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.pop("builder", None)
    context.user_data.pop("builder_idx", None)
    await update.message.reply_text("Canceled.", reply_markup=main_keyboard())
    return ConversationHandler.END


def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in .env or environment variables.")

    application = ApplicationBuilder().token(token).build()

    conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex("^Predict$"), predict_button),
            MessageHandler(filters.Regex("^/start$"), start),
        ],
        states={
            MODE_SELECT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_mode_choice)],
            AWAITING_URL: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url)],
            MANUAL_FIELD: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_manual)],
        },
        fallbacks=[CommandHandler("cancel", cancel), MessageHandler(filters.Regex("^Cancel$"), cancel)],
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("about", about))
    application.add_handler(conv)
    application.add_handler(MessageHandler(filters.Regex("^About$"), about))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, start))

    logger.info("Bot is running...")
    application.run_polling()


if __name__ == "__main__":
    main()
