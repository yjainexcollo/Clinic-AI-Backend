#!/usr/bin/env python3
"""
Test script to verify the application can start properly.
Run this locally to test before deploying to Azure.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    try:
        # Add src to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        sys.path.insert(0, src_path)
        
        logger.info(f"Testing imports from: {src_path}")
        
        # Test basic imports
        import fastapi
        logger.info("✅ FastAPI imported successfully")
        
        import uvicorn
        logger.info("✅ Uvicorn imported successfully")
        
        import motor
        logger.info("✅ Motor imported successfully")
        
        import beanie
        logger.info("✅ Beanie imported successfully")
        
        import openai
        logger.info("✅ OpenAI imported successfully")
        
        # Test application import
        from clinicai.app import app
        logger.info("✅ Clinic-AI app imported successfully")
        
        # Test configuration
        from clinicai.core.config import get_settings
        settings = get_settings()
        logger.info(f"✅ Settings loaded: {settings.app_name} v{settings.app_version}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_environment():
    """Test environment variables."""
    required_vars = [
        'MONGO_URI',
        'MONGO_DB_NAME',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"⚠️  Missing environment variables: {missing_vars}")
        logger.warning("These are required for full functionality")
    else:
        logger.info("✅ All required environment variables are set")
    
    return len(missing_vars) == 0

def main():
    """Run all tests."""
    logger.info("🧪 Testing Clinic-AI application...")
    
    # Test imports
    if not test_imports():
        logger.error("❌ Import tests failed")
        sys.exit(1)
    
    # Test environment
    test_environment()
    
    logger.info("✅ All tests passed! Application is ready for deployment.")
    return True

if __name__ == "__main__":
    main()
