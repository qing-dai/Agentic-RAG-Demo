### Retrieval Evaluator
from dotenv import load_dotenv
load_dotenv()  
import os
from typing import Optional, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime
from pydantic import BaseModel, Field

QUERY_EXTRACTOR_MODEL = os.getenv("QUERY_EXTRACTOR_MODEL", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

today = datetime.today().strftime("%Y-%m-%d")

print(today)

class QueryExtractorOutput(BaseModel):
    symbol: str = Field(description="Yahoo Finance ticker, e.g. GC=F or XAUUSD=X")
    date: str = Field(description="YYYY-MM-DD")
    display_name: str = Field(description="A short, human-readable name of the ticker")

class FinanceQueryExtractor:
    """
    Turn a natural-language question into {symbol, date} for Yahoo Finance.
    """

    SYSTEM_TMPL = """
You are a question extractor that converts a user question into a JSON object
for Yahoo Finance search. The JSON must have exactly three fields:
- "symbol": the Yahoo Finance ticker
- "date": the date in YYYY-MM-DD format
- "display_name": a short, human-readable name of the ticker 
   (e.g. GC=F → "Gold Futures", ^GSPC → "S&P 500 Index", AAPL → "Apple Inc.")

Rules for choosing the ticker:
- Always prefer stable tickers: futures contracts (like GC=F, CL=F, ZW=F), 
  major stock indices (like ^GSPC, ^DJI), large-cap stocks (like AAPL, MSFT, TSLA), 
  or liquid ETFs (like SPY, GLD, SLV).
- Do NOT use FX spot symbols (e.g. XAUUSD=X, EURUSD=X), as they are unreliable.
- If unsure, pick the most common futures, index, stock, or ETF ticker.

Today's date is {today}.

Return ONLY a valid JSON object, with no prose and no code fences. Example:

{{
  "symbol": "GC=F",
  "date": "2025-08-22",
  "display_name": "Gold Futures"
}}
"""

    def __init__(
        self,
        api_key: Optional[str] = OPENAI_API_KEY,
        model: str = QUERY_EXTRACTOR_MODEL,
        temperature: float = 0.0,
        reasoning_effort: str = "low",
        today: Optional[str] = None,
    ):

        if not api_key:
            raise RuntimeError(f"OPENAI_API_KEY not set in environment.")

        self.today = today or datetime.today().strftime("%Y-%m-%d")

        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            reasoning={"effort": reasoning_effort},
        )

        # Build prompt once; {today} is filled at runtime.
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.SYSTEM_TMPL), ("user", "{question}")]
        )

        # Bind schema to force JSON output that matches QueryExtractorOutput
        self.structured_llm = self.llm.with_structured_output(QueryExtractorOutput)

        # Final chain
        self.chain = self.prompt | self.structured_llm

    def extract(self, question: str, *, today: Optional[str] = None) -> Dict[str, str]:
        """Return {'symbol': ..., 'date': ...}"""
        _today = today or self.today
        out: QueryExtractorOutput = self.chain.invoke({"today": _today, "question": question})
        return out.model_dump()

# ----------------------------
if __name__ == "__main__":
    extractor = FinanceQueryExtractor()
    print(extractor.extract("What is the price of gold yesterday?"))
    print(extractor.extract("price of wheat on 2024-12-31"))