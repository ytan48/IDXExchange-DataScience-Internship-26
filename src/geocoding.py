"""Address geocoding for the California valuation app."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CENSUS_GEOCODER_URL = (
    "https://geocoding.geo.census.gov/geocoder/"
    "geographies/onelineaddress"
)
_BENCHMARK = "Public_AR_Current"
_VINTAGE = "Current_Current"
_LAYERS = "Counties,Unified School Districts"
_MAX_ADDRESS_LENGTH = 100


class GeocodingError(RuntimeError):
    """Base error for address lookup failures."""


class AddressNotFoundError(GeocodingError):
    """Raised when the Census Geocoder cannot match an address."""


class AddressOutsideServiceAreaError(GeocodingError):
    """Raised when a matched address is outside California."""


class GeocodingServiceError(GeocodingError):
    """Raised when the remote geocoding service is unavailable or invalid."""


@dataclass(frozen=True)
class GeocodedAddress:
    query_address: str
    matched_address: str
    latitude: float
    longitude: float
    city: str | None
    state: str | None
    postal_code: str | None
    county: str | None
    unified_school_district: str | None


UrlOpener = Callable[..., Any]


def geocode_address(
    address: str,
    *,
    timeout: float = 12.0,
    opener: UrlOpener = urlopen,
) -> GeocodedAddress:
    """Resolve one California street address with the U.S. Census Geocoder."""

    query_address = " ".join(str(address).split())
    if not query_address:
        raise GeocodingError("Enter a property address.")
    if len(query_address) > _MAX_ADDRESS_LENGTH:
        raise GeocodingError(
            f"The address must be {_MAX_ADDRESS_LENGTH} characters or fewer."
        )

    parameters = urlencode(
        {
            "address": query_address,
            "benchmark": _BENCHMARK,
            "vintage": _VINTAGE,
            "layers": _LAYERS,
            "format": "json",
        }
    )
    request = Request(
        f"{CENSUS_GEOCODER_URL}?{parameters}",
        headers={
            "Accept": "application/json",
            "User-Agent": "IDX-Exchange-Residential-Valuation/1.0",
        },
    )

    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise GeocodingServiceError(
            "The address lookup service is temporarily unavailable."
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
        raise GeocodingServiceError(
            "The address lookup service returned an invalid response."
        ) from error

    match = _first_match(payload)
    coordinates = _mapping(match.get("coordinates"))
    try:
        longitude = float(coordinates["x"])
        latitude = float(coordinates["y"])
    except (KeyError, TypeError, ValueError) as error:
        raise GeocodingServiceError(
            "The matched address did not include valid coordinates."
        ) from error
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise GeocodingServiceError(
            "The matched address did not include valid coordinates."
        )

    components = _mapping(match.get("addressComponents"))
    state = _optional_text(components.get("state"))
    if state is not None and state.upper() != "CA":
        raise AddressOutsideServiceAreaError(
            "The valuation model supports California addresses only."
        )

    geographies = _mapping(match.get("geographies"))
    county = _geography_name(geographies, "Counties")
    school_district = _geography_name(
        geographies,
        "Unified School Districts",
    )

    return GeocodedAddress(
        query_address=query_address,
        matched_address=(
            _optional_text(match.get("matchedAddress")) or query_address
        ),
        latitude=latitude,
        longitude=longitude,
        city=_optional_text(components.get("city")),
        state=state,
        postal_code=_optional_text(components.get("zip")),
        county=county,
        unified_school_district=school_district,
    )


def match_known_category(
    value: str | None,
    options: Iterable[str],
) -> str | None:
    """Return the model category matching a geocoder label, if one exists."""

    if value is None:
        return None

    normalized_value = _normalize_category(value)
    for option in options:
        if _normalize_category(str(option)) == normalized_value:
            return str(option)
    return None


def _first_match(payload: Any) -> Mapping[str, Any]:
    root = _mapping(payload)
    result = _mapping(root.get("result"))
    matches = result.get("addressMatches")
    if not isinstance(matches, list):
        raise GeocodingServiceError(
            "The address lookup service returned an invalid response."
        )
    if not matches:
        raise AddressNotFoundError(
            "No matching address was found. Check the street, city, state, and ZIP."
        )

    match = matches[0]
    if not isinstance(match, Mapping):
        raise GeocodingServiceError(
            "The address lookup service returned an invalid response."
        )
    return match


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _geography_name(
    geographies: Mapping[str, Any],
    layer_name: str,
) -> str | None:
    records = geographies.get(layer_name)
    if not isinstance(records, list) or not records:
        return None
    record = records[0]
    if not isinstance(record, Mapping):
        return None
    return _optional_text(record.get("BASENAME") or record.get("NAME"))


def _normalize_category(value: str) -> str:
    normalized = " ".join(value.casefold().replace(",", " ").split())
    for suffix in (" school district", " county"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip()
    return normalized
