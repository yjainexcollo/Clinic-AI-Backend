#!/usr/bin/env python3
"""
Verification script for Azure AI Foundry integration.

Tests Azure OpenAI connectivity and logs to Application Insights.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
script_dir = Path(__file__).parent
backend_dir = script_dir.parent
src_dir = backend_dir / "src"
sys.path.insert(0, str(src_dir))

# Load environment variables from .env if it exists
env_file = backend_dir / ".env"
if env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(env_file)

import time

from clinicai.core.ai_factory import get_ai_client
from clinicai.core.config import get_settings


async def main():
    print("🔍 Verifying Azure AI Foundry integration...\n")

    try:
        client = get_ai_client()
        print("✅ Client initialized successfully")

        # Get endpoint and API version from settings
        settings = get_settings()
        endpoint = settings.azure_openai.endpoint or "unknown"
        api_version = settings.azure_openai.api_version or "unknown"
        deployment = settings.azure_openai.deployment_name or "unknown"

        print(f"📍 Endpoint: {endpoint}")
        print(f"🔖 API Version: {api_version}")
        print(f"🤖 Deployment: {deployment}\n")

        print("🚀 Sending test request to Azure OpenAI…")
        start_time = time.time()

        # AzureAIClient.chat() returns just the response, not a tuple
        response = await client.chat(
            messages=[{"role": "user", "content": "Say OK only."}],
            max_tokens=5,
            temperature=0.0,
        )

        latency = time.time() - start_time

        print("✅ Response: OK")
        print(f"🆔 Request ID: {response.id}")
        print(f"📊 Prompt tokens: {response.usage.prompt_tokens}")
        print(f"📊 Completion tokens: {response.usage.completion_tokens}")
        print(f"📊 Total tokens: {response.usage.total_tokens}")
        print(f"⏱️  Latency: {latency*1000:.2f}ms")
        print(f"🏁 Finish reason: {response.choices[0].finish_reason}")
        print(f"\n💬 Response content: {response.choices[0].message.content}")

        print("\n🔗 Next steps:")
        print(
            f"   - Azure AI Foundry → Monitoring → Requests → filter by Request ID `{response.id}` or deployment `{deployment}`"
        )
        print(
            f"   - Log Analytics → Logs → run: AzureOpenAIRequests | where RequestId == '{response.id}'"
        )
        print(
            "   - Application Insights → Logs → query dependencies by operation_Id to cross-check"
        )
        print(
            "\n✅ Foundry verification complete! Data should be visible in all three surfaces within a few minutes."
        )

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
