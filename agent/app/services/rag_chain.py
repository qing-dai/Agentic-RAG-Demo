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

# Chain
rag_chain = rag_prompt | rag_llm | StrOutputParser()
