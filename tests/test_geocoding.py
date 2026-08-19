import io
import json
import unittest
from urllib.error import URLError

from src.geocoding import (
    AddressNotFoundError,
    AddressOutsideServiceAreaError,
    GeocodingServiceError,
    geocode_address,
    match_known_category,
)


def census_payload(*, state="CA", matches=True):
    address_matches = []
    if matches:
        address_matches.append(
            {
                "matchedAddress": (
                    "6175 ONEIDA DR, SAN JOSE, CA, 95123"
                ),
                "coordinates": {
                    "x": -121.832935903267,
                    "y": 37.234464735549,
                },
                "addressComponents": {
                    "city": "SAN JOSE",
                    "state": state,
                    "zip": "95123",
                },
                "geographies": {
                    "Counties": [
                        {
                            "BASENAME": "Santa Clara",
                            "NAME": "Santa Clara County",
                        }
                    ],
                    "Unified School Districts": [
                        {"NAME": "San Jose Unified School District"}
                    ],
                },
            }
        )
    return {"result": {"addressMatches": address_matches}}


def payload_opener(payload):
    def opener(request, timeout):
        opener.request = request
        opener.timeout = timeout
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    return opener


class CensusGeocoderTests(unittest.TestCase):
    def test_parses_california_address_and_geographies(self):
        opener = payload_opener(census_payload())

        result = geocode_address(
            "  6175  Oneida Drive, San Jose, CA 95123  ",
            opener=opener,
        )

        self.assertEqual(
            result.query_address,
            "6175 Oneida Drive, San Jose, CA 95123",
        )
        self.assertEqual(result.city, "SAN JOSE")
        self.assertEqual(result.postal_code, "95123")
        self.assertEqual(result.county, "Santa Clara")
        self.assertEqual(
            result.unified_school_district,
            "San Jose Unified School District",
        )
        self.assertAlmostEqual(result.latitude, 37.234464735549)
        self.assertAlmostEqual(result.longitude, -121.832935903267)
        self.assertIn("layers=Counties%2CUnified+School+Districts", opener.request.full_url)
        self.assertEqual(opener.timeout, 12.0)

    def test_rejects_address_without_a_match(self):
        with self.assertRaises(AddressNotFoundError):
            geocode_address(
                "1 Missing Street, Los Angeles, CA",
                opener=payload_opener(census_payload(matches=False)),
            )

    def test_rejects_address_outside_california(self):
        with self.assertRaises(AddressOutsideServiceAreaError):
            geocode_address(
                "4600 Silver Hill Road, Washington, DC 20233",
                opener=payload_opener(census_payload(state="DC")),
            )

    def test_wraps_network_errors(self):
        def failing_opener(request, timeout):
            raise URLError("offline")

        with self.assertRaises(GeocodingServiceError):
            geocode_address(
                "6175 Oneida Drive, San Jose, CA 95123",
                opener=failing_opener,
            )

    def test_matches_geocoder_names_to_model_categories(self):
        self.assertEqual(
            match_known_category(
                "Santa Clara County",
                ["Los Angeles", "Santa Clara"],
            ),
            "Santa Clara",
        )
        self.assertEqual(
            match_known_category(
                "San Jose Unified School District",
                ["Los Angeles Unified", "San Jose Unified"],
            ),
            "San Jose Unified",
        )
        self.assertIsNone(
            match_known_category("Unknown County", ["Los Angeles"])
        )


if __name__ == "__main__":
    unittest.main()
