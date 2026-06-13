import requests
import pandas as pd
import time

# ── NHTSA API ──────────────────────────────────────────────────────────────────

def get_recalls_by_vehicle(make: str, model: str, model_year: int | str,
                           retries: int = 3, backoff: float = 2.0) -> dict:
    base_url = "https://api.nhtsa.gov/recalls/recallsByVehicle"
    params = {"make": make, "model": model, "modelYear": model_year}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(base_url, params=params, timeout=10)

            if response.status_code == 400:
                return {"results": [], "count": 0}

            if response.status_code in (503, 429, 500, 502, 504):
                wait = backoff * attempt
                print(f"      ⚠️  HTTP {response.status_code} — retrying in {wait}s (attempt {attempt}/{retries})")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            wait = backoff * attempt
            print(f"      ⚠️  Connection error — retrying in {wait}s (attempt {attempt}/{retries})")
            time.sleep(wait)

        except requests.exceptions.Timeout:
            wait = backoff * attempt
            print(f"      ⚠️  Timeout — retrying in {wait}s (attempt {attempt}/{retries})")
            time.sleep(wait)

    # All retries exhausted — return empty so the row is skipped, not crashed
    print(f"      ❌ Failed after {retries} attempts — skipping {model_year} {make} {model}")
    return {"results": [], "count": 0, "error": True}


def has_recall(make: str, model: str, model_year: int | str) -> tuple[bool, int, bool]:
    """Returns (was_recalled, number_of_recalls, had_error)"""
    data = get_recalls_by_vehicle(make, model, model_year)
    results = data.get("results", [])
    had_error = data.get("error", False)
    return len(results) > 0, len(results), had_error


# ── FILE PROCESSING ────────────────────────────────────────────────────────────

def process_vehicles(input_file: str, output_file: str,
                     make_col: str = "make",
                     model_col: str = "model",
                     year_col: str = "year"):
    # ── Load input ─────────────────────────────────────────────────────────────
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file, dtype={year_col: str})
    else:
        df = pd.read_csv(input_file, dtype={year_col: str})

    print(f"Loaded {len(df)} vehicles from '{input_file}'")
    print(f"Columns detected: {list(df.columns)}\n")

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors='coerce')
    df["YEAR"] = df["YEAR"].astype(int)
    #df = df.loc[(df["YEAR"] >= 2021) & (df["YEAR"] <= 2026)]
    df = df.drop_duplicates(subset = ['MAKE', 'MODEL', 'YEAR'], ignore_index = True)

    recalled_rows = []
    skipped_rows  = []

    for i, row in df.iterrows():
        make  = str(row[make_col]).strip()
        model = str(row[model_col]).strip()
        year  = str(row[year_col]).strip()

        print(f"  [{i+1}/{len(df)}] Checking {year} {make} {model}...", end=" ")

        recalled, count, had_error = has_recall(make, model, year)

        if had_error:
            skipped_rows.append(f"{year} {make} {model}")
        elif recalled:
            print(f"✅ {count} recall(s) found")
            enriched = row.to_dict()
            enriched["recall_count"] = count
            recalled_rows.append(enriched)
        else:
            print("❌ No recalls")

        time.sleep(0.3)

    # ── Write output ───────────────────────────────────────────────────────────
    if recalled_rows:
        out_df = pd.DataFrame(recalled_rows)
        if output_file.endswith(".xlsx"):
            out_df.to_excel(output_file, index=False)
        else:
            out_df.to_csv(output_file, index=False)
        print(f"\n✅ Done! {len(out_df)} recalled vehicle(s) written to '{output_file}'")
    else:
        print("\nNo recalled vehicles found.")

    if skipped_rows:
        print(f"\n⚠️  {len(skipped_rows)} vehicle(s) skipped due to API errors:")
        for v in skipped_rows:
            print(f"     - {v}")
        print("   Re-run the script to retry these.")

# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    process_vehicles(
        input_file="complaints_raw.csv",    # ← your input file  (.csv or .xlsx)
        output_file="recalled_2010_2020.csv",   # ← output file      (.csv or .xlsx)
        make_col="MAKE",              # ← column name in your file
        model_col="MODEL",
        year_col="YEAR",
    )