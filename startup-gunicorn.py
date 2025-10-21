#!/usr/bin/env python3
"""
Alternative startup script using Gunicorn for Azure App Service.
This is more reliable than uvicorn for Azure deployments.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {current_dir}")
logger.info(f"Source path: {src_path}")

def main():
    try:
        port = os.environ.get("PORT", "8000")
        logger.info(f"Starting application on port {port}")
        
        # Test import before starting
        from clinicai.app import app
        logger.info("Successfully imported clinicai.app")
        
        # Start with gunicorn
        cmd = [
            "gunicorn",
            "--config", "gunicorn.conf.py",
            "--bind", f"0.0.0.0:{port}",
            "--workers", "1",
            "--worker-class", "uvicorn.workers.UvicornWorker",
            "--timeout", "30",
            "--keep-alive", "2",
            "--max-requests", "1000",
            "--max-requests-jitter", "50",
            "--log-level", "info",
            "--access-logfile", "-",
            "--error-logfile", "-",
            "clinicai.app:app"
        ]
        
        logger.info(f"Starting with command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
