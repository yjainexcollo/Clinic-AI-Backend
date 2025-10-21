import os
import sys
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force OpenAI Whisper API usage for Azure free tier
# This prevents the "No module named 'whisper'" error
os.environ["TRANSCRIPTION_SERVICE"] = "openai"

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {current_dir}")
logger.info(f"Source path: {src_path}")
logger.info(f"TRANSCRIPTION_SERVICE forced to: {os.environ.get('TRANSCRIPTION_SERVICE')}")

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"Starting application on port {port}")
        
        # Test import before starting
        from clinicai.app import app
        logger.info("Successfully imported clinicai.app")
        
        uvicorn.run(
            "clinicai.app:app",
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
