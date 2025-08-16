# ByeMarianne Patternist Puzzle - Diary Text Analysis

A Python toolkit specifically designed to analyze diary text and find patterns like "in everything" using bigram semantic analysis with NLTK.

## 📚 Quick Navigation

### **[COMPLETE BEGINNER? START HERE →](BEGINNER_GUIDE.md)**
If you're new to Python or command line, we have a special guide just for you!

### For Experienced Users
Jump to [Quick Start](#instructions) below or check out the [examples](examples/) folder.

## Features

### Tokenization
- **Multiple tokenization methods** (word, sentence, regex-based)
- **Smart text preprocessing** (lowercase, punctuation removal, stopword filtering)
- **Stemming and lemmatization** support
- **Part-of-speech tagging**
- **Clean word extraction** (no partial words or apostrophes)

### Bigram Analysis
- **Frequency analysis** with customizable thresholds
- **Statistical collocation detection** using multiple measures:
  - Pointwise Mutual Information (PMI)
  - Chi-square test
  - Likelihood ratio
  - Student's t-test
- **Semantic similarity computation** using TF-IDF
- **Context analysis** for specific bigrams
- **Network visualization** of bigram relationships
- **Export to CSV** for further analysis

### Visualizations
- Frequency bar charts
- Word clouds
- Statistical summaries

## 📁 Project Structure

```
nltk-bigram-analyzer/
│
├── src/                      # Core modules
│   ├── tokenizer.py         # Text tokenization functionality
│   └── bigram_analyzer.py   # Bigram analysis functionality
│
├── data/                     # Sample datasets
│   ├── sample_text.txt      # NLP-focused sample text
│   └── diary_text.txt       # Personal diary sample text
│
├── examples/                 # Example usage scripts
│   ├── main.py              # Basic usage example
│   ├── analyze_diary.py    # Diary text analysis
│   └── analyze_diary_complete.py  # Complete analysis with/without stopwords
│
├── output/                   # Generated visualizations (created on run)
│
├── requirements.txt          # Python dependencies
├── setup.py                 # Installation script
├── .gitignore              # Git ignore file
├── LICENSE                  # MIT License
└── README.md               # This file
```

## Instructions:

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/nltk-bigram-analyzer.git
cd nltk-bigram-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data (one-time setup):**
```bash
python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])"
```

### Basic Usage

#### Option 1: Run the example script
```bash
python examples/main.py
```

#### Option 2: Use in your Python code
```python
import sys
sys.path.append('src')
from tokenizer import TextTokenizer
from bigram_analyzer import BigramAnalyzer

# Load your text
with open('data/sample_text.txt', 'r') as f:
    text = f.read()

# Tokenize
tokenizer = TextTokenizer(remove_stopwords=True)
tokens = tokenizer.basic_tokenize(text)

# Analyze bigrams
analyzer = BigramAnalyzer()
bigrams = analyzer.extract_bigrams(tokens)
top_bigrams = analyzer.get_top_bigrams(20)

# Print results
for bigram, freq in top_bigrams:
    print(f"{' '.join(bigram)}: {freq} occurrences")
```

## 📖 Detailed Usage Guide

### 1. Text Tokenization

```python
from tokenizer import TextTokenizer

# Initialize with custom settings
tokenizer = TextTokenizer(
    remove_stopwords=True,    # Filter common words
    lowercase=True,           # Convert to lowercase
    remove_punctuation=True   # Remove punctuation
)

# Different tokenization methods
tokens = tokenizer.basic_tokenize(text)           # Word tokenization
sentences = tokenizer.sentence_tokenize(text)     # Sentence splitting
clean_tokens = tokenizer.regexp_tokenize(text)    # Regex-based (no apostrophes)

# Text processing
stemmed = tokenizer.stem_tokens(tokens)          # Apply stemming
lemmatized = tokenizer.lemmatize_tokens(tokens)  # Apply lemmatization
pos_tags = tokenizer.get_pos_tags(tokens)        # Get POS tags
```

### 2. Bigram Analysis

```python
from bigram_analyzer import BigramAnalyzer

analyzer = BigramAnalyzer()

# Extract bigrams
bigrams = analyzer.extract_bigrams(tokens)

# Get frequency statistics
top_20 = analyzer.get_top_bigrams(20)
stats = analyzer.generate_bigram_statistics(tokens)

# Find collocations
collocations = analyzer.find_collocations(
    tokens, 
    n=20,           # Number of results
    min_freq=2      # Minimum frequency
)

# Analyze specific bigram context
context = analyzer.analyze_bigram_context(text, ('machine', 'learning'))

# Export results
df = analyzer.export_results(tokens, 'results.csv')
```

### 3. Including/Excluding Stopwords

To capture phrases like "in everything" that contain stopwords:

```python
# Include stopwords
tokenizer_with_stops = TextTokenizer(remove_stopwords=False)
tokens_with_stops = tokenizer_with_stops.basic_tokenize(text)

# Exclude stopwords (for content words only)
tokenizer_no_stops = TextTokenizer(remove_stopwords=True)
tokens_no_stops = tokenizer_no_stops.basic_tokenize(text)
```

## Example Outputs

### Sample Analysis Results

When you run the analyzer, you'll get:

1. **Frequency Analysis:**
```
Top Bigrams:
1. 'natural language': 9 occurrences
2. 'machine learning': 7 occurrences
3. 'deep learning': 5 occurrences
```

2. **Statistical Measures:**
```
PMI Scores (words that co-occur more than by chance):
1. neural network
2. sentiment analysis
3. transfer learning
```

3. **Exported CSV** with columns:
- bigram (e.g., "machine learning")
- word1, word2 (individual words)
- frequency
- pmi_score
- chi_square_score

## Use Cases

- **Academic Research**: Analyze research papers for key concept pairs
- **Content Analysis**: Find recurring themes in documents
- **Social Media**: Identify trending phrase combinations
- **Literary Analysis**: Study author writing patterns
- **Market Research**: Extract product feature combinations from reviews

## Working with Your Own Data

### Preparing Your Text File

1. Create a `.txt` file with your text
2. Save it in the `data/` directory
3. Use UTF-8 encoding for best compatibility

### Running Analysis on Custom Text

```bash
# Method 1: Modify the example script
python examples/main.py --input data/your_text.txt

# Method 2: Create your own script
```

```python
# custom_analysis.py
import sys
sys.path.append('src')
from tokenizer import TextTokenizer
from bigram_analyzer import BigramAnalyzer

# Your analysis code here
with open('data/your_text.txt', 'r') as f:
    text = f.read()
    
# ... continue with analysis
```

## Configuration Options

### Tokenizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remove_stopwords` | bool | True | Filter out common words |
| `lowercase` | bool | True | Convert text to lowercase |
| `remove_punctuation` | bool | True | Remove punctuation marks |

### Bigram Analyzer Parameters

| Method | Parameter | Type | Description |
|--------|-----------|------|-------------|
| `get_top_bigrams()` | `n` | int | Number of top bigrams to return |
| `find_collocations()` | `min_freq` | int | Minimum frequency threshold |
| | `measure` | str | Statistical measure (pmi, chi_square, etc.) |

## Understanding the Statistical Measures

- **PMI (Pointwise Mutual Information)**: Measures how much more likely two words appear together than separately
- **Chi-square**: Tests the independence of word occurrences
- **Likelihood Ratio**: Compares observed vs expected frequencies
- **Student's t-test**: Measures confidence in bigram associations

## Troubleshooting

### Common Issues

1. **NLTK Data Error**
```bash
# Download all NLTK data
python -c "import nltk; nltk.download('all')"
```

2. **Import Errors**
```bash
# Ensure you're in the project directory
cd nltk-bigram-analyzer
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

3. **Matplotlib Display Issues**
```python
# For headless systems, use:
import matplotlib
matplotlib.use('Agg')  # Before importing pyplot
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [NLTK](https://www.nltk.org/) - Natural Language Toolkit
- Statistical measures based on Manning & Schütze's "Foundations of Statistical Natural Language Processing"
- Word cloud generation using [wordcloud](https://github.com/amueller/word_cloud)

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Text Mining**