import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel,ValidationError

load_dotenv()

groq_client=Groq(api_key=os.getenv("GROQ_API_KEY"))

gemini_api_key=os.getenv("GEMINI_API_KEY")
Model={
    "small":"llama-3.1-8b-instant",
    "large": "llama-3.3-70b-versatile"
}
