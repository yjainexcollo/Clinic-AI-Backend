import asyncio
from clinicai.core.ai_factory import get_ai_client

async def main():
    client = get_ai_client()

    print("Sending test request to Azure OpenAI…")

    response = await client.chat(
        messages=[
            {"role": "user", "content": "Say OK only."}
        ],
        max_tokens=5,
        temperature=0.0,
    )

    print("Response:", response.choices[0].message.content)
    print("Request ID:", response.id)
    print("Prompt tokens:", response.usage.prompt_tokens)
    print("Completion tokens:", response.usage.completion_tokens)

if __name__ == "__main__":
    asyncio.run(main())
