### Generate
import os
from dotenv import load_dotenv

load_dotenv()  
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Prompt
rag_prompt = hub.pull("rlm/rag-prompt")
# LLM
rag_llm =  ChatOpenAI(
                    model="gpt-5", 
                    api_key=os.getenv("OPENAI_API_KEY"),
                    reasoning={"effort":"low"},
                    temperature=0
)
# Post-processing
def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)
# Chain
rag_chain = rag_prompt | rag_llm | StrOutputParser()
# print(rag_prompt.messages[0].prompt.template)

# docs = [
#     type("Doc", (), {"page_content": "Most EU exports to the US now face a 15% tariff after a July 28 deal."}),
#     type("Doc", (), {"page_content": "The US had earlier threatened 30% tariffs starting August 1."})
# ]
# question = "What is the tariff situation between the US and the EU?"

# result = rag_chain.invoke({"context": format_docs(docs), "question": question})
# print(result)