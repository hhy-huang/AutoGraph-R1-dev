from openai import OpenAI

client = OpenAI(base_url="http://localhost:8130/v1", api_key="test")
respo = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Say this is a test!"}
    ],
    response_format={"type": "text"}  # <------ added this line
)
print(respo.choices[0].message.content)