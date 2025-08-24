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
DEFAULT_MODEL = os.getenv("RETRIEVAL_EVAL_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ----------------------------
# Schema
# ----------------------------
class RetrievalEvaluatorOutput(BaseModel):
    """Structured, minimal output for downstream logic."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# ----------------------------
# Service
# ----------------------------
class RetrievalEvaluatorService:
    """
    Wraps: prompt → structured output → easy evaluate(document, question).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = OPENAI_API_KEY,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ):
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Set it in your environment."
            )
        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
        )

        sys_msg = system_prompt or (
            "You are a document retrieval evaluator responsible for checking "
            "the relevancy of a retrieved document to the user's question.\n"
            "If the document contains keyword(s) or semantic meaning related "
            "to the question, grade it as relevant.\n"
            "Output a binary score 'yes' or 'no' to indicate whether the "
            "document is relevant to the question."
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_msg),
                ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
            ]
        )

        # Bind the structured output schema
        self.structured_llm = self.model.with_structured_output(RetrievalEvaluatorOutput)

        # Full runnable chain
        self.chain = self.prompt | self.structured_llm

    def evaluate(self, document: str, question: str) -> RetrievalEvaluatorOutput:
        """
        Returns RetrievalEvaluatorOutput with binary_score ∈ {'yes','no'}.
        Raises on transport/schema errors.
        """
        return self.chain.invoke({"document": document, "question": question})

    # Optional: convenience that returns plain 'yes'/'no'
    def score(self, document: str, question: str) -> str:
        return self.evaluate(document, question).binary_score




# ----------------------------
# CLI demo
# ----------------------------
def _demo():
    load_dotenv()  # read env if present
    svc = RetrievalEvaluatorService()

    doc = "The EU and US discussed tariff de-escalation measures in 2024 around steel and aluminum."
    q = "What is the tariff situation between the US and the EU?"

    result = svc.evaluate(doc, q)
    print("Binary score:", result.binary_score)  # 'yes' or 'no'


if __name__ == "__main__":
    _demo()