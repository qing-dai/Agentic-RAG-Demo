from ..services.extract_finance_info import FinanceQueryExtractor

def extract_ticker(state):
    question = state["question"]
    extractor = FinanceQueryExtractor()
    ticker = extractor.extract(question)
    if ticker:
        print(f"---EXTRACTED TICKER: {ticker}---")
        return {"question": question, "documents": ticker}
    else:
        print("---NO TICKER FOUND---")
        return {"question": question, "documents": None}
