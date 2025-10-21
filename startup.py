import os
import sys
import uvicorn

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "clinicai.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info"
    )
