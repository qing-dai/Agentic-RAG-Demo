from dotenv import load_dotenv
load_dotenv()

import os
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ----------------------------
# Config
# ----------------------------
QUERY_EVAL_MODEL = os.getenv("QUERY_EVAL_MODEL", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ----------------------------
# Schema
# ----------------------------
class QueryEvaluatorOutput(BaseModel):
    """Structured, minimal output for downstream logic."""
    binary_score: Literal["yes", "no"] = Field(
        description="If the question is related to ticker price, output 'yes' or 'no'"
    )


# ----------------------------
# Service
# ----------------------------
class QueryEvaluatorService:
    """
    Wraps: prompt → structured output → easy evaluate(question).
    """

    def __init__(
        self,
        model_name: str = QUERY_EVAL_MODEL,
        api_key: Optional[str] = OPENAI_API_KEY,
        system_prompt: Optional[str] = None,
    ):
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Set it in your environment."
            )
        self.model = ChatOpenAI(
            model=model_name,
            reasoning={"effort":"low"},
            api_key=api_key,
            temperature=0,
        )

        sys_msg = system_prompt or (
            """You are a router that labels questions as price-related or not.
            Output 'yes' only if the user explicitly asks for a numeric market price/quote (current or for a specific date)
            for a financial instrument, commodity (e.g., wheat/corn/gold/silver/oil), index, stock, or ETF. Otherwise output 'no'.

            Examples:
            Q: What is the price of wheat today?      A: yes
            Q: Gold price yesterday?                  A: yes
            Q: Close of ^GSPC on 2024-12-31?          A: yes
            Q: What is the tariff situation EU–US?    A: no
            Q: Summarize wheat export bans in 2024.   A: no
            """
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_msg),
                ("human", "User question: {question}"),
            ]
        )

        # Bind the structured output schema
        self.structured_llm = self.model.with_structured_output(QueryEvaluatorOutput)

        # Full runnable chain
        self.chain = self.prompt | self.structured_llm

    def evaluate(self, question: str) -> QueryEvaluatorOutput:
        """
        Returns QueryEvaluatorOutput with binary_score ∈ {'yes','no'}.
        Raises on transport/schema errors.
        """
        return self.chain.invoke({"question": question})

    # Optional: convenience that returns plain 'yes'/'no'
    def score(self, question: str) -> str:
        return self.evaluate(question).binary_score




# ----------------------------
# CLI demo
# ----------------------------
def _demo():
    load_dotenv()  # read env if present
    svc = QueryEvaluatorService()

    q = "How do I look today?"

    result = svc.evaluate(q)
    print(f"result: {result}")
    print("Binary score:", result.binary_score)  # 'yes' or 'no'


if __name__ == "__main__":
    _demo()