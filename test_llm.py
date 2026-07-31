from openai import OpenAI

client = OpenAI(
    base_url="https://api.xah.io/v1",
    api_key="sk-8597e27f6a58b06d558a5764d4df8c29f87eb5b29978ae395d4690e3576b1eec"
)

r = client.chat.completions.create(
    model="w3leee/CodeX GPT 5.5",
    messages=[
        {
            "role":"user",
            "content":"Extract relation from: BERT is evaluated on GLUE benchmark. Return JSON only."
        }
    ]
)

print(r.choices[0].message.content)
