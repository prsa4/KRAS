from pathlib import Path
from llama_cpp import Llama

MODEL_PATH = Path(__file__).resolve().parent / "models" / "Gemma3-UNCENSORED-1B.Q4_K_M.gguf"

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=0,
    verbose=False,
    chat_format="gemma",
)

messages = [
    {
        "role": "system",
        "content": "Ты полезный ассистент. Отвечай на русском языке кратко и понятно."
    }
]

while True:
    user_text = input("Ты: ")

    if user_text.lower() in ["exit", "quit", "выход"]:
        break

    messages.append({
        "role": "user",
        "content": user_text
    })

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        top_p=0.9,
        max_tokens=300,
    )

    answer = response["choices"][0]["message"]["content"].strip()

    messages.append({
        "role": "assistant",
        "content": answer
    })

    print("Gemma:", answer)