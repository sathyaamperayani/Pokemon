"""
Task 1: Data Acquisition - PokeAPI
No API key required. Docs: https://pokeapi.co/docs/v2

Step 1: paginate through the list endpoint (limit/offset) to collect
        every Pokemon's name + detail URL.
Step 2: call the detail URL for each Pokemon to get stats/types, and
        flatten the nested JSON into flat columns.
"""

import time
import requests
import pandas as pd

LIST_URL = "https://pokeapi.co/api/v2/pokemon"
PAGE_LIMIT = 100          # how many names per list page
SLEEP_BETWEEN_CALLS = 0.2 # be polite to the free API


def fetch_all_pokemon_urls() -> list[dict]:
    """Step 1: page through the list endpoint until `next` is null."""
    urls = []
    next_url = LIST_URL
    params = {"limit": PAGE_LIMIT, "offset": 0}

    while next_url:
        resp = requests.get(next_url, params=params if next_url == LIST_URL else None)
        resp.raise_for_status()
        data = resp.json()

        urls.extend(data["results"])  # each item: {"name": ..., "url": ...}
        print(f"Listed {len(urls)} pokemon so far...")

        next_url = data.get("next")   # PokeAPI gives you the full next-page URL
        params = None                 # only needed for the very first call
        time.sleep(SLEEP_BETWEEN_CALLS)

    return urls


def flatten_pokemon(detail: dict) -> dict:
    """Step 2: flatten one Pokemon's detail JSON into a flat dict."""
    stats = {s["stat"]["name"]: s["base_stat"] for s in detail.get("stats", [])}
    types = [t["type"]["name"] for t in detail.get("types", [])]

    return {
        "id": detail.get("id"),
        "name": detail.get("name"),
        "height": detail.get("height"),
        "weight": detail.get("weight"),
        "base_experience": detail.get("base_experience"),
        "hp": stats.get("hp"),
        "attack": stats.get("attack"),
        "defense": stats.get("defense"),
        "special_attack": stats.get("special-attack"),
        "special_defense": stats.get("special-defense"),
        "speed": stats.get("speed"),
        "primary_type": types[0] if len(types) > 0 else None,
        "secondary_type": types[1] if len(types) > 1 else None,
        "n_types": len(types),
    }


def fetch_all_records(limit_pokemon: int | None = None) -> list[dict]:
    """Full pipeline: get all list entries, then fetch + flatten each detail."""
    entries = fetch_all_pokemon_urls()

    if limit_pokemon:
        entries = entries[:limit_pokemon]  # handy for a quick test run

    records = []
    for i, entry in enumerate(entries, start=1):
        detail_resp = requests.get(entry["url"])
        detail_resp.raise_for_status()
        records.append(flatten_pokemon(detail_resp.json()))

        if i % 50 == 0:
            print(f"Fetched details for {i}/{len(entries)} pokemon")

        time.sleep(SLEEP_BETWEEN_CALLS)

    return records


if __name__ == "__main__":
    # For a first quick test, you can pass e.g. fetch_all_records(limit_pokemon=200)
    # to only pull the first 200 while you check everything works, then
    # remove the limit to pull the full dataset (1300+ pokemon).
    records = fetch_all_records()
    df = pd.DataFrame(records)
    df.to_csv("data/raw_data.csv", index=False)
    print(f"Saved {len(df)} rows to data/raw_data.csv")
