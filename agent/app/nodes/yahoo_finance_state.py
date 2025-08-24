from ..tools.yahoo_finance_api import get_yahoo_finance_price


def yahoo_search(state):
    docs = state["documents"]
    print(f"YAHOO SEARCH RES: {docs}")
    query = state["question"]
    ticker = docs["symbol"]
    date = docs["date"]
    ticker_name = docs["display_name"]
    price = get_yahoo_finance_price(ticker, date, ticker_name)
    return {"documents": price, "question": query}