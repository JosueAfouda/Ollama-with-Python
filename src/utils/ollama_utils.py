import ollama
from ollama import AsyncClient

def pull_ollama_model(model_name: str):
    """Pulls an Ollama model."""
    print(f"Pulling Ollama model: {model_name}...")
    ollama.pull(model_name)
    print(f"Model {model_name} pulled successfully.")

async def generate_text_ollama(model: str, prompt: str):
    """Generates text using Ollama's generate API."""
    client = AsyncClient()
    response = await client.generate(model=model, prompt=prompt)
    return response.get("response", "")

async def chat_ollama(model: str, messages: list):
    """Performs a chat interaction with Ollama."""
    client = AsyncClient()
    response = await client.chat(model=model, messages=messages)
    return response.get("message", {})
