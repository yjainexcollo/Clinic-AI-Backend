import os
import sys
import uvicorn
import logging
import traceback

# Configure logging to stdout (Azure App Service reads from here)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Let .env file control transcription service (defaults to azure_speech)
# No hardcoding - respect environment configuration

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

logger.info("=" * 60)
logger.info("Clinic-AI Backend Startup")
logger.info("=" * 60)
logger.info(f"Python version: {sys.version.split()[0]}")
logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {current_dir}")
logger.info(f"Source path: {src_path}")

# Log critical environment variables (without exposing full secrets)
logger.info("\nEnvironment Configuration:")
logger.info(f"  PORT: {os.environ.get('PORT', '8000')}")
logger.info(f"  PYTHON_VERSION: {os.environ.get('PYTHON_VERSION', 'not set')}")
logger.info(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
logger.info(f"  APP_ENV: {os.environ.get('APP_ENV', 'not set')}")
logger.info(f"  AZURE_KEY_VAULT_NAME: {os.environ.get('AZURE_KEY_VAULT_NAME', 'not set')}")
logger.info(f"  TRANSCRIPTION_SERVICE: {os.environ.get('TRANSCRIPTION_SERVICE', 'azure_speech')}")
logger.info(f"  AZURE_OPENAI_ENDPOINT: {'✅ set' if os.environ.get('AZURE_OPENAI_ENDPOINT') else '❌ not set'}")
logger.info(f"  AZURE_OPENAI_API_KEY: {'✅ set' if os.environ.get('AZURE_OPENAI_API_KEY') else '❌ not set'}")
logger.info(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', '❌ not set')}")
logger.info(f"  AZURE_OPENAI_API_VERSION: {os.environ.get('AZURE_OPENAI_API_VERSION', 'not set')}")
logger.info(f"  MONGO_URI: {'✅ set' if os.environ.get('MONGO_URI') else '❌ not set'}")
logger.info(f"  MONGO_DB_NAME: {os.environ.get('MONGO_DB_NAME', 'not set')}")
logger.info(f"  SECURITY_SECRET_KEY: {'✅ set' if os.environ.get('SECURITY_SECRET_KEY') else '⚠️  not set (using default)'}")
if os.environ.get('SECURITY_SECRET_KEY'):
    key_len = len(os.environ.get('SECURITY_SECRET_KEY', ''))
    logger.info(f"  SECURITY_SECRET_KEY length: {key_len} chars {'✅' if key_len >= 32 else '❌ (must be >= 32)'}")

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8000))
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting application on port {port}")
        logger.info(f"{'=' * 60}\n")
        
        # Test import before starting
        logger.info("Step 1: Importing clinicai.app...")
        try:
            from clinicai.app import app
            logger.info("✅ Successfully imported clinicai.app")
        except Exception as import_error:
            logger.error(f"❌ Failed to import clinicai.app: {import_error}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            sys.exit(1)
        
        # Test settings loading (this will catch config validation errors early)
        logger.info("Step 2: Loading application settings...")
        try:
            from clinicai.core.config import get_settings
            settings = get_settings()
            logger.info("✅ Settings loaded successfully")
            logger.info(f"  App name: {settings.app_name}")
            logger.info(f"  App version: {settings.app_version}")
            logger.info(f"  App environment: {settings.app_env}")
            logger.info(f"  Debug mode: {settings.debug}")
        except ValueError as ve:
            logger.error(f"❌ Configuration validation failed: {ve}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            logger.error("\n⚠️  Common configuration issues:")
            logger.error("  1. SECURITY_SECRET_KEY must be >= 32 characters")
            logger.error("  2. MONGO_URI must be set and valid")
            logger.error("  3. AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set")
            sys.exit(1)
        except Exception as settings_error:
            logger.error(f"❌ Failed to load settings: {settings_error}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            sys.exit(1)
        
        # Start uvicorn with better error handling
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Step 3: Starting uvicorn server on 0.0.0.0:{port}...")
        logger.info(f"{'=' * 60}\n")
        uvicorn.run(
            "clinicai.app:app",
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level="info",
            access_log=True,
            # Add timeout settings for Azure App Service
            timeout_keep_alive=75,
            timeout_graceful_shutdown=30,
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Shutting down due to keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n{'=' * 60}")
        logger.error(f"❌ CRITICAL: Failed to start application")
        logger.error(f"{'=' * 60}")
        logger.error(f"Error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error(f"\n{'=' * 60}")
        logger.error("Troubleshooting steps:")
        logger.error("1. Check Azure App Service logs (Log stream) for detailed errors")
        logger.error("2. Verify all Key Vault secrets are present and accessible")
        logger.error("3. Verify SECURITY_SECRET_KEY is >= 32 characters")
        logger.error("4. Verify Azure OpenAI deployment exists and is accessible")
        logger.error("5. Verify MongoDB connection string is correct")
        logger.error("6. Check if the app is binding to the correct port (should be 0.0.0.0:8000)")
        logger.error(f"{'=' * 60}\n")
        sys.exit(1)
