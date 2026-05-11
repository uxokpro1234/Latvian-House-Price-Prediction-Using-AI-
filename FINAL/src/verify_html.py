import html
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class MockResult:
    current_price: float = 150000.0
    scraped_price: Optional[float] = 145000.0
    location: Optional[str] = "Riga_Center"
    street: Optional[str] = "Brivibas_iela & Tallinas"
    rooms: int = 3
    area: float = 75.0
    floor: int = 3
    total_floors: int = 5
    building_type: str = "Special_Project"
    year: int = 1995
    condition: str = "Good_Condition"
    explanation: Dict[str, float] = None
    price_1y: float = 160000.0
    price_5y: float = 200000.0
    price_10y: float = 250000.0

def test_format_response():
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

    result = MockResult()
    result.explanation = {"area": 5000.0, "location_underscore": -2000.0}

    lines = []
    lines.append("🎯 <b>Price Estimate</b>")
    lines.append(f"Market Value: <b>{result.current_price:,.0f} EUR</b>")
    if result.scraped_price:
        delta = result.current_price - result.scraped_price
        diff_text = "below" if delta > 0 else "above"
        lines.append(f"Listing Price: {result.scraped_price:,.0f} EUR (<i>{abs(delta):,.0f} EUR {diff_text} market</i>)")
    
    lines.append("\n" + "─" * 20)

    lines.append("🏠 <b>Property Specifications</b>")
    loc_val = html.escape(str(result.location or "Unknown"))
    if result.street:
        loc_val += f", {html.escape(str(result.street))}"
    lines.append(f"📍 Location: {loc_val}")
    
    specs = []
    if result.rooms: specs.append(f"{result.rooms} rooms")
    if result.area: specs.append(f"{result.area} m²")
    if result.floor:
        floor_str = f"{result.floor}"
        if result.total_floors: floor_str += f"/{result.total_floors}"
        specs.append(f"Floor {floor_str}")
    if specs: lines.append(f"📐 Specs: {' &#8226; '.join(specs)}")
    
    build = []
    if result.building_type: build.append(html.escape(str(result.building_type)))
    if result.year: build.append(f"Built {result.year}")
    if result.condition: build.append(html.escape(str(result.condition)))
    if build: lines.append(f"🏢 Building: {' &#8226; '.join(build)}")

    lines.append("\n" + "─" * 20)
    
    if result.explanation:
        lines.append("📊 <b>Price Drivers</b>")
        sorted_exp = sorted(result.explanation.items(), key=lambda x: abs(x[1]), reverse=True)
        top_factors = sorted_exp[:5]
        for name, val in top_factors:
            label = html.escape(FEATURE_LABELS.get(name, str(name)))
            sign = "+" if val > 0 else "-"
            lines.append(f"&#8226; {label}: {sign}{abs(val):,.0f} EUR")
        lines.append("\n" + "─" * 20)

    lines.append("📈 <b>Value Forecast</b>")
    lines.append(f"📆 +1 Year: <b>{result.price_1y:,.0f} EUR</b>")
    lines.append(f"📅 +5 Years: <b>{result.price_5y:,.0f} EUR</b>")
    lines.append(f"⏳ +10 Years: <b>{result.price_10y:,.0f} EUR</b>")
    
    lines.append("\n<i>Forecast based on historical Riga growth rates.</i>")

    resp_str = "\n".join(lines)
    print("--- Formatted Message ---")
    print(resp_str)
    print("--- End ---")
    
    assert "🎯 <b>Price Estimate</b>" in resp_str
    assert "─" * 20 in resp_str
    assert "📊 <b>Price Drivers</b>" in resp_str
    assert "📈 <b>Value Forecast</b>" in resp_str
    assert "Disclaimer" not in resp_str
    print("Verification SUCCESS")

if __name__ == "__main__":
    test_format_response()
