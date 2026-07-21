import os
from openai import OpenAI

# Initialize client (uses OpenAI-compatible API)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # Replace with your provider base URL if needed
    api_key=os.environ.get("GROQ_API_KEY")      # Replace with your API key environment variable
)

def run_simple_agent(prompt: str) -> None:
    # 1. Initialize message history
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 2. Call the model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    # 3. Print the output
    print("Agent Response:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    run_simple_agent("Hello! Who are you?")