from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class LLM_wrapper:

    def __init__(self,llm,model,temperature = 0) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature

    def choose_llm(self):
        if self.llm == "ollama":
            model = ChatOllama(
                model=self.model,
                temperature= self.temperature
            )
            return model
        elif self.llm == "gemini":
            model = ChatGoogleGenerativeAI(
                model = self.model,
                temperature = self.temperature
            )
            return model
        elif self.llm == "openai":
            model = ChatOpenAI(
                model = self.model,
                temperature = self.temperature
            )
            return model