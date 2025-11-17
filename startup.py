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

# Use print with flush to ensure logs appear immediately in Azure
print("=" * 60, flush=True)
print("Clinic-AI Backend Startup", flush=True)
print("=" * 60, flush=True)
logger.info("=" * 60)
logger.info("Clinic-AI Backend Startup")
logger.info("=" * 60)
python_version = sys.version.split()[0]
print(f"Python version: {python_version}", flush=True)
logger.info(f"Python version: {python_version}")
print(f"Python path: {sys.path}", flush=True)
logger.info(f"Python path: {sys.path}")
print(f"Current directory: {current_dir}", flush=True)
logger.info(f"Current directory: {current_dir}")
print(f"Source path: {src_path}", flush=True)
logger.info(f"Source path: {src_path}")

# Log critical environment variables (without exposing full secrets)
print("\nEnvironment Configuration:", flush=True)
logger.info("\nEnvironment Configuration:")
port_val = os.environ.get('PORT', '8000')
print(f"  PORT: {port_val}", flush=True)
logger.info(f"  PORT: {port_val}")
py_ver = os.environ.get('PYTHON_VERSION', 'not set')
print(f"  PYTHON_VERSION: {py_ver}", flush=True)
logger.info(f"  PYTHON_VERSION: {py_ver}")
py_path = os.environ.get('PYTHONPATH', 'not set')
print(f"  PYTHONPATH: {py_path}", flush=True)
logger.info(f"  PYTHONPATH: {py_path}")
app_env = os.environ.get('APP_ENV', 'not set')
print(f"  APP_ENV: {app_env}", flush=True)
logger.info(f"  APP_ENV: {app_env}")
kv_name = os.environ.get('AZURE_KEY_VAULT_NAME', 'not set')
print(f"  AZURE_KEY_VAULT_NAME: {kv_name}", flush=True)
logger.info(f"  AZURE_KEY_VAULT_NAME: {kv_name}")
trans_serv = os.environ.get('TRANSCRIPTION_SERVICE', 'azure_speech')
print(f"  TRANSCRIPTION_SERVICE: {trans_serv}", flush=True)
logger.info(f"  TRANSCRIPTION_SERVICE: {trans_serv}")
openai_endpoint = '✅ set' if os.environ.get('AZURE_OPENAI_ENDPOINT') else '❌ not set'
print(f"  AZURE_OPENAI_ENDPOINT: {openai_endpoint}", flush=True)
logger.info(f"  AZURE_OPENAI_ENDPOINT: {openai_endpoint}")
openai_key = '✅ set' if os.environ.get('AZURE_OPENAI_API_KEY') else '❌ not set'
print(f"  AZURE_OPENAI_API_KEY: {openai_key}", flush=True)
logger.info(f"  AZURE_OPENAI_API_KEY: {openai_key}")
openai_deploy = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', '❌ not set')
print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {openai_deploy}", flush=True)
logger.info(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {openai_deploy}")
openai_ver = os.environ.get('AZURE_OPENAI_API_VERSION', 'not set')
print(f"  AZURE_OPENAI_API_VERSION: {openai_ver}", flush=True)
logger.info(f"  AZURE_OPENAI_API_VERSION: {openai_ver}")
mongo_uri = '✅ set' if os.environ.get('MONGO_URI') else '❌ not set'
print(f"  MONGO_URI: {mongo_uri}", flush=True)
logger.info(f"  MONGO_URI: {mongo_uri}")
mongo_db = os.environ.get('MONGO_DB_NAME', 'not set')
print(f"  MONGO_DB_NAME: {mongo_db}", flush=True)
logger.info(f"  MONGO_DB_NAME: {mongo_db}")
secret_key = '✅ set' if os.environ.get('SECURITY_SECRET_KEY') else '⚠️  not set (using default)'
print(f"  SECURITY_SECRET_KEY: {secret_key}", flush=True)
logger.info(f"  SECURITY_SECRET_KEY: {secret_key}")
if os.environ.get('SECURITY_SECRET_KEY'):
    key_len = len(os.environ.get('SECURITY_SECRET_KEY', ''))
    key_status = f"  SECURITY_SECRET_KEY length: {key_len} chars {'✅' if key_len >= 32 else '❌ (must be >= 32)'}"
    print(key_status, flush=True)
    logger.info(key_status)

if __name__ == "__main__":
    try:
        from clinicai.core.config import get_settings
        settings = get_settings()
        port = int(os.environ.get("PORT", settings.port))
        host = os.environ.get("HOST", settings.host)

        sep = "=" * 60
        print(f"\n{sep}", flush=True)
        print(f"Starting application on {host}:{port}", flush=True)
        print(f"{sep}\n", flush=True)
        logger.info(f"\n{sep}")
        logger.info(f"Starting application on {host}:{port}")
        logger.info(f"{sep}\n")
        
        # Test import before starting
        print("Step 1: Importing clinicai.app...", flush=True)
        logger.info("Step 1: Importing clinicai.app...")
        try:
            from clinicai.app import app
            print("✅ Successfully imported clinicai.app", flush=True)
            logger.info("✅ Successfully imported clinicai.app")
        except Exception as import_error:
            error_msg = f"❌ Failed to import clinicai.app: {import_error}"
            print(error_msg, flush=True)
            logger.error(error_msg)
            print("Full traceback:", flush=True)
            logger.error("Full traceback:")
            tb = traceback.format_exc()
            print(tb, flush=True)
            logger.error(tb)
            sys.exit(1)
        
        # Test settings loading (this will catch config validation errors early)
        print("Step 2: Loading application settings...", flush=True)
        logger.info("Step 2: Loading application settings...")
        try:
            print("✅ Settings loaded successfully", flush=True)
            logger.info("✅ Settings loaded successfully")
            print(f"  App name: {settings.app_name}", flush=True)
            logger.info(f"  App name: {settings.app_name}")
            print(f"  App version: {settings.app_version}", flush=True)
            logger.info(f"  App version: {settings.app_version}")
            print(f"  App environment: {settings.app_env}", flush=True)
            logger.info(f"  App environment: {settings.app_env}")
            print(f"  Debug mode: {settings.debug}", flush=True)
            logger.info(f"  Debug mode: {settings.debug}")
        except ValueError as ve:
            error_msg = f"❌ Configuration validation failed: {ve}"
            print(error_msg, flush=True)
            logger.error(error_msg)
            print("Full traceback:", flush=True)
            logger.error("Full traceback:")
            tb = traceback.format_exc()
            print(tb, flush=True)
            logger.error(tb)
            print("\n⚠️  Common configuration issues:", flush=True)
            logger.error("\n⚠️  Common configuration issues:")
            print("  1. SECURITY_SECRET_KEY must be >= 32 characters", flush=True)
            logger.error("  1. SECURITY_SECRET_KEY must be >= 32 characters")
            print("  2. MONGO_URI must be set and valid", flush=True)
            logger.error("  2. MONGO_URI must be set and valid")
            print("  3. AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set", flush=True)
            logger.error("  3. AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set")
            sys.exit(1)
        except Exception as settings_error:
            error_msg = f"❌ Failed to load settings: {settings_error}"
            print(error_msg, flush=True)
            logger.error(error_msg)
            print("Full traceback:", flush=True)
            logger.error("Full traceback:")
            tb = traceback.format_exc()
            print(tb, flush=True)
            logger.error(tb)
            sys.exit(1)
        
        # Start uvicorn with better error handling
        sep = "=" * 60
        print(f"\n{sep}", flush=True)
        print(f"Step 3: Starting uvicorn server on {host}:{port}...", flush=True)
        print(f"{sep}\n", flush=True)
        logger.info(f"\n{sep}")
        logger.info(f"Step 3: Starting uvicorn server on {host}:{port}...")
        logger.info(f"{sep}\n")
        uvicorn.run(
            "clinicai.app:app",
            host=host,
            port=port,
            workers=1,
            log_level="info",
            access_log=True,
            # Add timeout settings for Azure App Service
            timeout_keep_alive=75,
            timeout_graceful_shutdown=30,
        )
    except KeyboardInterrupt:
        msg = "\n⚠️  Shutting down due to keyboard interrupt"
        print(msg, flush=True)
        logger.info(msg)
        sys.exit(0)
    except Exception as e:
        sep = "=" * 60
        print(f"\n{sep}", flush=True)
        print(f"❌ CRITICAL: Failed to start application", flush=True)
        print(f"{sep}", flush=True)
        print(f"Error: {e}", flush=True)
        print(f"Error type: {type(e).__name__}", flush=True)
        print("\nFull traceback:", flush=True)
        tb = traceback.format_exc()
        print(tb, flush=True)
        print(f"\n{sep}", flush=True)
        print("Troubleshooting steps:", flush=True)
        print("1. Check Azure App Service logs (Log stream) for detailed errors", flush=True)
        print("2. Verify all Key Vault secrets are present and accessible", flush=True)
        print("3. Verify SECURITY_SECRET_KEY is >= 32 characters", flush=True)
        print("4. Verify Azure OpenAI deployment exists and is accessible", flush=True)
        print("5. Verify MongoDB connection string is correct", flush=True)
        print("6. Check if the app is binding to the correct port (should be 0.0.0.0:8000)", flush=True)
        print(f"{sep}\n", flush=True)
        logger.error(f"\n{sep}")
        logger.error(f"❌ CRITICAL: Failed to start application")
        logger.error(f"{sep}")
        logger.error(f"Error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("\nFull traceback:")
        logger.error(tb)
        logger.error(f"\n{sep}")
        logger.error("Troubleshooting steps:")
        logger.error("1. Check Azure App Service logs (Log stream) for detailed errors")
        logger.error("2. Verify all Key Vault secrets are present and accessible")
        logger.error("3. Verify SECURITY_SECRET_KEY is >= 32 characters")
        logger.error("4. Verify Azure OpenAI deployment exists and is accessible")
        logger.error("5. Verify MongoDB connection string is correct")
        logger.error("6. Check if the app is binding to the correct port (should be 0.0.0.0:8000)")
        logger.error(f"{sep}\n")
        sys.exit(1)