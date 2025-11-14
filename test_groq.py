from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv(".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Say hello!"},
    ],
)

print("\nAI Response:\n", response.choices[0].message.content)

