#!/bin/bash

# Knowledge Graph Builder - Startup Script

echo "🧠 Knowledge Graph Builder - Starting Application"
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

# Check if pip3 is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "
import sys
packages = ['streamlit', 'networkx', 'plotly', 'requests', 'bs4', 'openai', 'dotenv', 'pandas', 'numpy']
missing = []
for package in packages:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print(f'❌ Missing packages: {missing}')
    print('💡 Run: pip3 install -r requirements.txt')
    sys.exit(1)
else:
    print('✅ All dependencies are installed')
"

if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  No .env file found"
    echo "💡 Copy .env.example to .env and add your OpenAI API key"
    echo "   Or enter your API key directly in the app sidebar"
fi

# Run functionality test
echo "🧪 Running functionality test..."
python3 test_functionality.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Starting Streamlit application..."
    echo "📱 The app will open in your default web browser"
    echo "🔑 Don't forget to enter your OpenAI API key in the sidebar!"
    echo ""
    
    # Start Streamlit
    streamlit run app.py
else
    echo "❌ Functionality test failed. Please check the implementation."
    exit 1
fi