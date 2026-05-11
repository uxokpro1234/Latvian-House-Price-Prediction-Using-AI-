from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

SUPPORTED_DOMAINS = ("ss.lv", "ss.com", "city24.lv", "latio.lv")


@dataclass
class ListingFeatures:
    url: str
    location: Optional[str] = None
    area: Optional[float] = None
    rooms: Optional[int] = None
    floor: Optional[int] = None
    total_floors: Optional[int] = None
    building_type: Optional[str] = None
    year: Optional[int] = None
    rental_or_sale: Optional[str] = None
    price: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    condition: Optional[str] = None
    street: Optional[str] = None
    images: Optional[list[str]] = None

    def to_feature_dict(self) -> Dict:
        return {
            "location": self.location,
            "area": self.area,
            "rooms": self.rooms,
            "floor": self.floor,
            "total_floors": self.total_floors,
            "building_type": self.building_type,
            "year": self.year,
            "rental_or_sale": self.rental_or_sale,
            "price": self.price,
            "lat": self.lat,
            "lon": self.lon,
            "condition": self.condition,
            "street": self.street,
            "images": self.images,
        }


def fetch_html(url: str) -> BeautifulSoup:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RealEstatePredictor/2.0)"}
    resp = requests.get(url, headers=headers, timeout=12)
    resp.raise_for_status()
    if "ss.lv" in url or "ss.com" in url:
        resp.encoding = "utf-8"
    return BeautifulSoup(resp.text, "html.parser")


def _parse_price(text: str) -> Optional[float]:
    match = re.search(r"([0-9][0-9\.,\s]+)\s*(?:€|eur|eur\.?)", text, flags=re.IGNORECASE)
    if match:
        value = match.group(1).replace(" ", "").replace(",", ".")
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_coords(soup: BeautifulSoup) -> tuple[Optional[float], Optional[float]]:
    for link in soup.find_all("a", onclick=True):
        onclick = link["onclick"]
        if "c=" in onclick:
            m = re.search(r"c=([\d\.]+),\s*([\d\.]+)", onclick)
            if m:
                return float(m.group(1)), float(m.group(2))
    return None, None


def _parse_images(soup: BeautifulSoup, base_url: str) -> list[str]:
    images = []

    main_img = soup.find(id="main_photo")
    if main_img and main_img.get("src"):
        src = main_img["src"]
        if src.startswith("//"):
            src = "https:" + src
        images.append(src)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".jpg") and "gallery" in href:
             if href.startswith("//"):
                 href = "https:" + href
             elif href.startswith("/"):
                 parsed = urlparse(base_url)
                 href = f"{parsed.scheme}://{parsed.netloc}{href}"
             if href not in images:
                 images.append(href)

    return images[:10]


def _parse_area(text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(?:m2|m\u00b2|kv\.m|m²)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def _parse_rooms(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*(?:room|rooms|ist)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _parse_floor(text: str) -> tuple[Optional[int], Optional[int]]:
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def _extract_table_data(soup: BeautifulSoup) -> Dict[str, str]:
    data = {}
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            key_text = cells[0].get_text(strip=True).lower().rstrip(":")
            if any(k in key_text for k in ["apraksts", "description", "описание"]):
                continue
            val = cells[1].get_text(strip=True)
            data[key_text] = val
    return data


def _generic_parse(url: str, soup: BeautifulSoup) -> ListingFeatures:
    text = soup.get_text(" | ", strip=True)
    price = _parse_price(text)
    area = _parse_area(text)
    rooms = _parse_rooms(text)
    floor, total_floors = _parse_floor(text)
    location = None
    building_type = None
    year = None
    street = None
    condition = None

    table_data = _extract_table_data(soup)
    for key, val in table_data.items():
        if not location and any(k in key for k in ["city", "district", "location", "address", "pils"]):
            location = val
        if not street and any(k in key for k in ["street", "iela"]):
            street = val
        if not building_type and any(k in key for k in ["type", "serija", "series"]):
            building_type = val
        if not condition and any(k in key for k in ["amenities", "ertibas", "ērtības"]):
            condition = val
        if not year and any(k in key for k in ["year", "built", "gads"]):
            m = re.search(r"(\d{4})", val)
            if m:
                year = int(m.group(1))
        if not area:
            a = _parse_area(val)
            if a:
                area = a
        if not rooms:
            r = _parse_rooms(val)
            if r:
                rooms = r
        if not floor:
            f, t = _parse_floor(val)
            floor = floor or f
            total_floors = total_floors or t
        if not price:
            p = _parse_price(val)
            if p:
                price = p

    lat, lon = _parse_coords(soup)
    images = _parse_images(soup, url)

    return ListingFeatures(
        url=url,
        location=location,
        area=area,
        rooms=rooms,
        floor=floor,
        total_floors=total_floors,
        building_type=building_type,
        year=year,
        rental_or_sale=None,
        price=price,
        lat=lat,
        lon=lon,
        street=street,
        condition=condition,
        images=images
    )


def parse_ss(url: str) -> ListingFeatures:
    soup = fetch_html(url)
    listing = _generic_parse(url, soup)
    price_candidates = []
    og_price = soup.find("meta", {"property": "og:price:amount"})
    if og_price and og_price.get("content"):
        price_candidates.append(_parse_price(og_price["content"] + " EUR"))
    tdo_price = soup.find(id="tdo_8")
    if tdo_price:
        price_candidates.append(_parse_price(tdo_price.get_text(" ", strip=True)))

    if not price_candidates and listing.price:
        price_candidates.append(listing.price)

    for val in price_candidates:
        if val:
            listing.price = val
            break

    if not listing.location:
        head = soup.find(class_="head_title")
        if head:
            listing.location = head.get_text(" / ", strip=True)

    if not listing.location:
        heading = soup.find("h1")
        if heading:
            h_text = heading.get_text(" / ", strip=True)
            if len(h_text) < 300:
                listing.location = h_text

    listing.rental_or_sale = "rent" if "rent" in url.lower() else "sale"
    return listing


def parse_city24(url: str) -> ListingFeatures:
    soup = fetch_html(url)
    listing = _generic_parse(url, soup)
    listing.rental_or_sale = "rent" if "rent" in url.lower() else "sale"
    if not listing.location:
        meta = soup.find("meta", {"property": "og:title"})
        if meta and meta.get("content"):
            listing.location = meta["content"]
    return listing


def parse_latio(url: str) -> ListingFeatures:
    soup = fetch_html(url)
    listing = _generic_parse(url, soup)
    listing.rental_or_sale = "rent" if "rent" in url.lower() else "sale"
    return listing


def parse_listing_url(url: str) -> ListingFeatures:
    domain = urlparse(url).netloc.lower()
    if not any(d in domain for d in SUPPORTED_DOMAINS):
        raise ValueError("Unsupported domain. Use ss.lv/ss.com, city24.lv, or latio.lv.")
    if "ss.lv" in domain or "ss.com" in domain:
        return parse_ss(url)
    if "city24.lv" in domain:
        return parse_city24(url)
    if "latio.lv" in domain:
        return parse_latio(url)
    raise ValueError("Unsupported domain.")
