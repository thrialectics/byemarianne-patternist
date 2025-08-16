#!/bin/bash

echo "======================================"
echo " NLTK Bigram Analyzer - Installation "
echo "======================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (recommended) [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    echo ""
fi

# Install requirements
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Packages installed"
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
for resource in resources:
    print(f'Downloading {resource}...')
    nltk.download(resource, quiet=True)
print('✅ NLTK data downloaded')
"
echo ""

# Test installation
echo "Testing installation..."
python3 -c "
import sys
sys.path.append('src')
try:
    from tokenizer import TextTokenizer
    from bigram_analyzer import BigramAnalyzer
    print('✅ All modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo " Installation Complete! 🎉"
    echo "======================================"
    echo ""
    echo "To get started, try:"
    echo "  python examples/quick_start.py"
    echo ""
    echo "Or analyze your own text:"
    echo "  python examples/quick_start.py --input your_text.txt"
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Remember to activate the virtual environment:"
        echo "  source venv/bin/activate"
    fi
else
    echo ""
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi
