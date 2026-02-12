# main.py
# pip install fastapi uvicorn pydantic

import json
import time
import uuid
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any, Union

from fastapi import FastAPI, Header, Request
from pydantic import BaseModel, Field

LOG_PATH = Path("tool_calls.jsonl")

app = FastAPI(
    title="Resolver + Shipping Tools",
    version="0.2.0",
    description="Hardcoded resolver tools + shipping quote tool for agentic demos.",
)

# -----------------------------
# Hardcoded "truth" dataset
# 5 countries x 10 cities
# -----------------------------
COUNTRY_ALIASES = {
    "deutschland": "DE",
    "germany": "DE",
    "österreich": "AT",
    "austria": "AT",
    "schweiz": "CH",
    "switzerland": "CH",
    "niederlande": "NL",
    "netherlands": "NL",
    "frankreich": "FR",
    "france": "FR",
}

CITY_TO_POSTAL = {
    "DE": {
        "berlin": "10115",
        "hamburg": "20095",
        "münchen": "80331",
        "koeln": "50667",
        "köln": "50667",
        "frankfurt": "60311",
        "stuttgart": "70173",
        "düsseldorf": "40213",
        "leipzig": "04109",
        "dortmund": "44135",
        "borken": "48455",
    },
    "AT": {
        "wien": "1010",
        "graz": "8010",
        "linz": "4020",
        "salzburg": "5020",
        "innsbruck": "6020",
        "klagenfurt": "9020",
        "villach": "9500",
        "wels": "4600",
        "st. pölten": "3100",
        "st poelten": "3100",
        
    },
    "CH": {
        "zürich": "8001",
        "zurich": "8001",
        "genf": "1201",
        "geneva": "1201",
        "basel": "4001",
        "bern": "3001",
        "lausanne": "1003",
        "winterthur": "8400",
        "luzern": "6003",
        "lugano": "6900",
        "st. gallen": "9000",
        "st gallen": "9000",
    },
    "NL": {
        "amsterdam": "1012",
        "rotterdam": "3011",
        "den haag": "2511",
        "utrecht": "3511",
        "eindhoven": "5611",
        "tilburg": "5038",
        "groningen": "9711",
        "almere": "1315",
        "breda": "4811",
        "nijmegen": "6511",
    },
    "FR": {
        "paris": "75001",
        "marseille": "13001",
        "lyon": "69001",
        "toulouse": "31000",
        "nice": "06000",
        "nantes": "44000",
        "montpellier": "34000",
        "strasbourg": "67000",
        "bordeaux": "33000",
        "lille": "59000",
    },
}


# -----------------------------
# Logging middleware
# -----------------------------
def log_jsonl(obj: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@app.middleware("http")
async def trace_and_log(request: Request, call_next):
    t0 = time.time()
    trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())

    body_bytes = await request.body()
    body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
    try:
        body_json = json.loads(body_text) if body_text else None
    except Exception:
        body_json = {"_raw": body_text[:2000]}

    response = await call_next(request)
    latency_s = round(time.time() - t0, 4)

    log_jsonl({
        "trace_id": trace_id,
        "ts": t0,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_s": latency_s,
        "request_body": body_json,
    })
    response.headers["x-trace-id"] = trace_id
    return response


# -----------------------------
# Tool schemas
# -----------------------------
class ResolveCountryRequest(BaseModel):
    name: str = Field(min_length=2, description="Country name, e.g. Deutschland, Austria, Schweiz")


class ResolveCountryResponse(BaseModel):
    iso2: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None
    trace_id: str


class ResolvePostalRequest(BaseModel):
    country: Optional[str] = Field(default=None, description="ISO2 if known")
    city: Optional[str] = Field(default=None, description="City name if given")
    value: Optional[str] = Field(default=None, description="Raw user value (city or postal), for validation")
    mode: Literal["lookup_city", "validate_postal"] = "lookup_city"


class ResolvePostalResponse(BaseModel):
    postal_code: Optional[str] = None
    city: Optional[str] = None
    error: Optional[str] = None
    trace_id: str


class BatchResolveItem(BaseModel):
    tool: Literal["resolve_country", "resolve_postal_code"]
    args: Dict[str, Any]


class BatchResolveRequest(BaseModel):
    requests: List[BatchResolveItem]


class BatchResolveResponse(BaseModel):
    results: List[Dict[str, Any]]
    trace_id: str


class ShippingQuoteRequest(BaseModel):
    country: str = Field(min_length=2, max_length=2, description="ISO-3166-1 alpha-2 country code (e.g. 'DE', 'AT', 'FR')")
    postal_code: str = Field(min_length=3, max_length=12, description="Valid postal code for the destination")
    weight_kg: float = Field(gt=0.0, lt=50.0, description="Package weight in kilograms")
    service: Literal["standard", "express"] = Field(default="standard", description="Shipping service: 'standard' or 'express'")


class ShippingQuoteResponse(BaseModel):
    currency: str = "EUR"
    price: float
    service: Literal["standard", "express"]
    trace_id: str


# -----------------------------
# Tools implementation
# -----------------------------
@app.post(
    "/v1/resolve/country",
    response_model=ResolveCountryResponse,
    operation_id="resolve_country",
    summary="Convert country name to ISO2 code",
    description="Resolves a country name or alias (e.g. 'Deutschland', 'Germany', 'Österreich') to ISO-3166-1 alpha-2 code. Do NOT call if you already have a 2-letter code like DE, FR, AT.",
)
def resolve_country(payload: ResolveCountryRequest, x_trace_id: Optional[str] = Header(default=None, alias="x-trace-id")):
    trace_id = x_trace_id or str(uuid.uuid4())
    key = payload.name.strip().lower()
    iso2 = COUNTRY_ALIASES.get(key)
    if iso2:
        return ResolveCountryResponse(iso2=iso2, confidence=1.0, trace_id=trace_id)
    return ResolveCountryResponse(error="not_found", confidence=0.0, trace_id=trace_id)


@app.post(
    "/v1/resolve/postal",
    response_model=ResolvePostalResponse,
    operation_id="resolve_postal_code",
    summary="Lookup postal code for a city or validate existing postal code",
    description="Use mode='lookup_city' with ISO2 country and city name to get postal code. Use mode='validate_postal' to check if a value is a valid postal code format.",
)
def resolve_postal(payload: ResolvePostalRequest, x_trace_id: Optional[str] = Header(default=None, alias="x-trace-id")):
    trace_id = x_trace_id or str(uuid.uuid4())

    # mode: validate_postal -> tool must NOT guess, only validate exact postal format + known mapping optionally
    if payload.mode == "validate_postal":
        raw = (payload.value or "").strip()
        # accept only digits for this demo (FR has leading 0 sometimes, still digits)
        if raw.isdigit():
            return ResolvePostalResponse(postal_code=raw, trace_id=trace_id)
        return ResolvePostalResponse(error="not_a_postal_code", trace_id=trace_id)

    # mode: lookup_city -> requires country+city for deterministic lookup
    if not payload.country or not payload.city:
        return ResolvePostalResponse(error="missing_country_or_city", trace_id=trace_id)

    c = payload.country.strip().upper()
    city = payload.city.strip().lower()

    m = CITY_TO_POSTAL.get(c, {})
    postal = m.get(city)
    if postal:
        return ResolvePostalResponse(postal_code=postal, city=payload.city, trace_id=trace_id)
    return ResolvePostalResponse(error="not_found", trace_id=trace_id)


def calc_quote(weight_kg: float, service: str) -> float:
    base = 4.90
    mult = 1.0 if service == "standard" else 1.8
    return round(base + max(weight_kg, 0.1) * 1.2 * mult, 2)


@app.post(
    "/v1/shipping/quote",
    response_model=ShippingQuoteResponse,
    operation_id="get_shipping_quote",
    summary="Calculate shipping quote",
    description="Calculate shipping price. Requires ISO2 country code and valid postal code. Call resolve_country and resolve_postal_code first if needed.",
)
def get_shipping_quote(payload: ShippingQuoteRequest, x_trace_id: Optional[str] = Header(default=None, alias="x-trace-id")):
    trace_id = x_trace_id or str(uuid.uuid4())
    price = calc_quote(payload.weight_kg, payload.service)
    return ShippingQuoteResponse(price=price, service=payload.service, trace_id=trace_id)
