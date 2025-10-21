#!/bin/bash
# Post-build script for Azure deployment

echo "🚀 Running post-build script..."

# Test if the application can be imported
python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from clinicai.app import app
    print('✅ Application import successful')
except Exception as e:
    print(f'❌ Application import failed: {e}')
    sys.exit(1)
"

echo "✅ Post-build script completed"
