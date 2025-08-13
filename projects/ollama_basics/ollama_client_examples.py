import ollama
import asyncio
from src.utils.ollama_utils import chat_ollama, generate_text_ollama, pull_ollama_model


response = ollama.list()

# print(response)

# == Chat example ==
async def run_chat_example():
    res_chat = await chat_ollama(
        model="llama3.2",
        messages=[
            {"role": "user", "content": "why is the sky blue?"},
        ],
    )
    print(res_chat["content"])

asyncio.run(run_chat_example())
# print(res["message"]["content"])

# == Chat example streaming ==
# res = ollama.chat(
#     model="llama3.2",
#     messages=[
#         {
#             "role": "user",
#             "content": "why is the ocean so salty?",
#         },
#     ],
#     stream=True,
# )
# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)


# ==================================================================================
# ==== The Ollama Python library's API is designed around the Ollama REST API ====
# ==================================================================================

# == Generate example ==
async def run_generate_example():
    res_generate = await generate_text_ollama(
        model="llama3.2",
        prompt="why is the sky blue?",
    )
    print(res_generate)

asyncio.run(run_generate_example())

# show
# print(ollama.show("llama3.2"))


# Create a new model with modelfile
modelfile = """
FROM llama3.2
SYSTEM You are very smart assistant who knows everything about oceans. You are very succinct and informative.
PARAMETER temperature 0.1
"""

ollama.create(model="knowitall", modelfile=modelfile)

async def run_custom_model_generate_example():
    res_custom_model = await generate_text_ollama(model="knowitall", prompt="why is the ocean so salty?")
    print(res_custom_model)

asyncio.run(run_custom_model_generate_example())


# delete model
ollama.delete("knowitall")