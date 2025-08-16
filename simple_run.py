#!/usr/bin/env python3
"""
DIARY TEXT BIGRAM ANALYZER
Analyzes the diary text to find patterns in word usage
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print(" DIARY TEXT PATTERN ANALYZER ")
print("=" * 60)
print()

# Check if required packages are installed
try:
    import nltk
    from tokenizer import TextTokenizer
    from bigram_analyzer import BigramAnalyzer
except ImportError as e:
    print("❌ Some packages are missing. Installing now...")
    print("This might take a minute...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Packages installed! Please run this script again.")
    input("\nPress Enter to exit...")
    sys.exit(0)

# Download NLTK data if needed
print("Setting up language analysis tools...")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading language data (one-time setup)...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✅ Language data ready!")

# Load the diary text
diary_path = os.path.join('data', 'diary_text.txt')
print(f"\n📖 Loading diary text...")
with open(diary_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"✅ Loaded {len(text)} characters of diary entries")
print()
print("=" * 60)
print(" ANALYZING DIARY PATTERNS ")
print("=" * 60)

# Analysis WITH stopwords to catch "in everything"
print("\n🔍 FULL PHRASE ANALYSIS (including 'in', 'the', etc.)")
print("-" * 40)

tokenizer_with_stops = TextTokenizer(
    remove_stopwords=False,  # Keep all words to find "in everything"
    lowercase=True,
    remove_punctuation=True
)

tokens_with_stops = tokenizer_with_stops.basic_tokenize(text)
print(f"Total words: {len(tokens_with_stops)}")

analyzer_with_stops = BigramAnalyzer()
bigrams_with_stops = analyzer_with_stops.extract_bigrams(tokens_with_stops)

# Count "in everything" specifically
in_everything_count = sum(1 for bigram in bigrams_with_stops 
                         if bigram == ('in', 'everything'))

print(f"\n🎯 'in everything' appears {in_everything_count} times!")

print("\n📊 TOP 20 MOST COMMON PHRASES:")
print("-" * 40)
top_phrases = analyzer_with_stops.get_top_bigrams(20)
for i, (bigram, freq) in enumerate(top_phrases, 1):
    bigram_str = ' '.join(bigram)
    bar = '█' * min(freq * 2, 40)
    if bigram_str == 'in everything':
        print(f"{i:2}. {bigram_str:25} {freq:2}x {bar} ⭐")
    else:
        print(f"{i:2}. {bigram_str:25} {freq:2}x {bar}")

# Analysis WITHOUT stopwords for content words
print("\n\n🔍 CONTENT WORD ANALYSIS (meaningful words only)")
print("-" * 40)

tokenizer_no_stops = TextTokenizer(
    remove_stopwords=True,
    lowercase=True,
    remove_punctuation=True
)

tokens_no_stops = tokenizer_no_stops.basic_tokenize(text)
print(f"Content words: {len(tokens_no_stops)}")

analyzer_no_stops = BigramAnalyzer()
bigrams_no_stops = analyzer_no_stops.extract_bigrams(tokens_no_stops)

print("\n📊 TOP 15 CONTENT WORD PAIRS:")
print("-" * 40)
top_content = analyzer_no_stops.get_top_bigrams(15)
for i, (bigram, freq) in enumerate(top_content, 1):
    bigram_str = ' '.join(bigram)
    bar = '█' * min(freq * 3, 40)
    print(f"{i:2}. {bigram_str:25} {freq:2}x {bar}")

# Theme analysis
print("\n\n🎨 THEMATIC PATTERNS IN THE DIARY:")
print("-" * 40)

themes = {
    'design & creation': ['design', 'designing', 'build', 'building', 'create', 'creating'],
    'emotions & feelings': ['feel', 'feeling', 'emotional', 'intelligence', 'suffer', 'suffering'],
    'observation': ['notice', 'noticing', 'see', 'seeing', 'read', 'reading', 'watch', 'watching'],
    'people & audience': ['people', 'players', 'audience', 'someone', 'everyone'],
    'absence & failure': ['absent', 'absence', 'missing', 'fail', 'failure', 'wrong']
}

for theme, keywords in themes.items():
    theme_bigrams = []
    for bigram, freq in analyzer_no_stops.bigram_freq.items():
        if any(keyword in bigram[0].lower() or keyword in bigram[1].lower() 
               for keyword in keywords):
            theme_bigrams.append((bigram, freq))
    
    if theme_bigrams:
        theme_bigrams.sort(key=lambda x: x[1], reverse=True)
        print(f"\n📌 {theme.upper()}:")
        for bigram, freq in theme_bigrams[:3]:  # Top 3 per theme
            print(f"   • '{' '.join(bigram)}' ({freq}x)")

# Statistical collocations
print("\n\n📈 STATISTICALLY SIGNIFICANT PATTERNS:")
print("(Word pairs that appear together more than random chance)")
print("-" * 40)

collocations = analyzer_no_stops.find_collocations(tokens_no_stops, n=10, min_freq=2)
for i, bigram in enumerate(collocations['pmi'][:10], 1):
    print(f"{i:2}. {' '.join(bigram)}")

# Save results
print("\n\n💾 SAVING DETAILED RESULTS...")
print("-" * 40)

# Export both analyses
df_with = analyzer_with_stops.export_results(tokens_with_stops, 'diary_analysis_with_all_words.csv')
df_without = analyzer_no_stops.export_results(tokens_no_stops, 'diary_analysis_content_words.csv')

print("✅ Created two files:")
print("   1. diary_analysis_with_all_words.csv - includes 'in everything'")
print("   2. diary_analysis_content_words.csv - meaningful words only")

print()
print("=" * 60)
print(" ✨ ANALYSIS COMPLETE! ")
print("=" * 60)
print()
print("KEY FINDINGS:")
print(f"• The phrase 'in everything' appears {in_everything_count} times")
print(f"• Most common phrase: '{' '.join(top_phrases[0][0])}' ({top_phrases[0][1]}x)")
print(f"• Total unique word pairs: {len(set(bigrams_with_stops))}")
print()
print("📁 Open the CSV files in Excel to explore all patterns!")
print()
input("Press Enter to exit...")