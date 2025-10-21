#!/bin/bash

# Azure Deployment Script for Clinic-AI Backend
# This script helps prepare the codebase for Azure deployment

echo "🚀 Preparing Clinic-AI Backend for Azure Deployment..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the backend directory."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p .github/workflows
mkdir -p logs

# Copy requirements for Azure if it doesn't exist
if [ ! -f "requirements-azure.txt" ]; then
    echo "📦 Creating Azure-optimized requirements..."
    cp requirements.txt requirements-azure.txt
fi

# Check if startup.py exists
if [ ! -f "startup.py" ]; then
    echo "❌ Error: startup.py not found. Please create it first."
    exit 1
fi

# Check if web.config exists
if [ ! -f "web.config" ]; then
    echo "❌ Error: web.config not found. Please create it first."
    exit 1
fi

# Check if .deployment exists
if [ ! -f ".deployment" ]; then
    echo "❌ Error: .deployment not found. Please create it first."
    exit 1
fi

echo "✅ All required files are present!"

# Display next steps
echo ""
echo "📋 Next Steps:"
echo "1. Initialize Git repository: git init"
echo "2. Add all files: git add ."
echo "3. Create initial commit: git commit -m 'Initial commit for Azure deployment'"
echo "4. Create GitHub repository and push code"
echo "5. Configure Azure App Service with GitHub deployment"
echo "6. Set environment variables in Azure Portal (see azure-env-config.md)"
echo ""
echo "📝 Important Notes:"
echo "- Your codebase only uses OpenAI (no Mistral or Redis)"
echo "- Configure only the environment variables listed in azure-env-config.md"
echo "- Use free tier optimizations for cost savings"
echo ""
echo "🔗 Useful Links:"
echo "- Azure Portal: https://portal.azure.com"
echo "- GitHub: https://github.com"
echo "- Your app will be available at: https://your-app-name.azurewebsites.net"
echo ""
echo "🎉 Ready for Azure deployment!"
