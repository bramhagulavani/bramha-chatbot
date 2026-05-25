import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv()

# Connect to Groq
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

print("Connected to Groq! ✅")
print("Sending first message to LLaMA 3...")
print("-" * 40)

# Send your first message!
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Hello! Who are you and what can you do?"
        }
    ]
)

# Print the response
print(response.choices[0].message.content)
print("-" * 40)
print("✅ LLaMA 3 is working perfectly!")