import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd

# ========= CONFIG =========

TICKER = "GOOG"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

OUT_PATH = os.path.join(DATA_DIR, f"finviz_{TICKER}.csv")


def parse_finviz_news(ticker: str) -> pd.DataFrame:
    """
    Scrape FinViz news table for a given ticker.

    Returns:
        DataFrame with columns: date (YYYY-MM-DD), text (headline)
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    print(f"Fetching FinViz page: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; stock-sentiment-project/1.0)"
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # FinViz keeps news in a table with id "news-table"
    news_table = soup.find("table", {"id": "news-table"})
    if news_table is None:
        print("Could not find news table on FinViz page.")
        return pd.DataFrame(columns=["date", "text"])

    rows = news_table.find_all("tr")

    records = []
    current_date_str = None

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue

        time_str = cols[0].get_text(strip=True)
        headline = cols[1].get_text(strip=True)

        # Example formats:
        #  "Jul-26-23 09:30AM"
        #  "09:45AM"  (date omitted, same as previous row)
        if " " in time_str:
            # Contains date and time
            date_part, time_part = time_str.split(" ", 1)
            current_date_str = date_part
        else:
            # Only time -> use last known date
            time_part = time_str
            if current_date_str is None:
                # If we don't have a date yet, skip
                continue

        # Convert date like "Jul-26-23" -> datetime.date
        try:
            dt = datetime.strptime(current_date_str, "%b-%d-%y").date()
        except Exception:
            # If parsing fails, skip this row
            continue

        records.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "text": headline,
            }
        )

    if not records:
        print("No news rows parsed from FinViz.")
        return pd.DataFrame(columns=["date", "text"])

    df = pd.DataFrame(records)

    # Drop duplicates: same date + text
    df = df.drop_duplicates(subset=["date", "text"])

    return df


def main():
    df = parse_finviz_news(TICKER)
    print(f"Collected {len(df)} FinViz news rows.")

    if len(df) == 0:
        print("Nothing to save.")
        return

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved FinViz CSV to {OUT_PATH}")
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()
