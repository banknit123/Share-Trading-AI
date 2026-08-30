from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

PROFILE_URL = "https://api.dhan.co/v2/profile"


def main() -> int:
    client_id = os.getenv("DHAN_CLIENT_ID", "").strip()
    access_token = os.getenv("DHAN_ACCESS_TOKEN", "").strip()

    print("Share-Trading-AI v29A.1 DhanHQ connection test")
    print("NO TRADES ARE SENT. This only validates credentials/data access.")

    missing = []
    if not client_id:
        missing.append("DHAN_CLIENT_ID")
    if not access_token:
        missing.append("DHAN_ACCESS_TOKEN")

    if missing:
        print("\nSTATUS: NOT CONFIGURED")
        print("Missing environment variables:", ", ".join(missing))
        print("Set them locally in this CMD window, then rerun this script.")
        return 2

    req = urllib.request.Request(
        PROFILE_URL,
        method="GET",
        headers={
            "access-token": access_token,
            "dhanClientId": client_id,
            "Accept": "application/json",
            "User-Agent": "Share-Trading-AI-v29A1",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"\nSTATUS: HTTP ERROR {exc.code}")
        print(body[:2000])
        return 3
    except Exception as exc:
        print(f"\nSTATUS: CONNECTION ERROR: {type(exc).__name__}: {exc}")
        return 4

    # Never print token/client secrets. Only non-secret account capability fields.
    print("\nSTATUS: CONNECTED")
    print("Token validity:", payload.get("tokenValidity", "unknown"))
    print("Active segment:", payload.get("activeSegment", "unknown"))
    print("Data plan:", payload.get("dataPlan", "unknown"))
    print("Data validity:", payload.get("dataValidity", "unknown"))

    data_plan = str(payload.get("dataPlan", "")).strip().lower()
    if data_plan == "active":
        print("\nV29 DATA STATUS: READY FOR HISTORICAL-DATA BACKFILL")
        return 0

    print("\nV29 DATA STATUS: ACCOUNT CONNECTED, BUT DATA API PLAN IS NOT REPORTED ACTIVE")
    print("Historical data access may require an active Dhan Data API subscription.")
    return 5


if __name__ == "__main__":
    sys.exit(main())
