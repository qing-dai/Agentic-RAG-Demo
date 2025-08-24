import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import time



def _to_date(d):
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(d, "%Y-%m-%d").date()

def is_weekend(date):
    date = _to_date(date)
    return date.weekday() >= 5

def is_future(date):
    date = _to_date(date)
    return date > datetime.today().date()

def get_yahoo_finance_price(ticker, retrieved_date, ticker_name=None):
    if not ticker or not retrieved_date:
        return {"error": "Ticker and date must be provided"}
    if is_future(retrieved_date):
        return {"error": f"{retrieved_date} is in the future, no data available"}
    elif is_weekend(retrieved_date):
        return {"error": f"{retrieved_date} is a weekend, no data available"}
    else:
        d = _to_date(retrieved_date)
        ticker_data = yf.Ticker(ticker)
        start = d
        end = start + timedelta(days=1)  # exclusive end -> fetch only that calendar day
        df = ticker_data.history(interval="1d", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        # print(f"Trying date {start} (requested {d}), data:\n{df}")
        if not df.empty:
            # daily bar -> use Close of the first/only row
            close = df["Close"].iloc[0]
            ts = df.index[0]
            return {
                "source": "yahoo_finance",
                "ticker": ticker,
                "price": float(close),
                "currency": "USD",
                "asof": ts.isoformat(),
                "display_name": ticker_name if ticker_name else "None",
            }


# Example usage:
if __name__ == "__main__":
    result = get_yahoo_finance_price('^GSPC',"2025-08-22")
    print("Gold price:", result)