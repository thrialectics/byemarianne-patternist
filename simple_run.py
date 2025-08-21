#!/usr/bin/env python3
"""
DIARY TEXT BIGRAM ANALYZER (neutral)
Analyzes the diary text to find patterns in word usage without targeting any specific phrase.
"""

import sys
import os
import argparse

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

# CLI options
parser = argparse.ArgumentParser(description="Neutral bigram analysis over diary text.")
parser.add_argument("--top", type=int, default=20, help="How many top bigrams/collocations to show (default: 20).")
parser.add_argument("--min-freq", type=int, default=2, help="Minimum bigram frequency for ranking (default: 2).")
parser.add_argument("--path", default=os.path.join('data', 'diary_text.txt'),
                    help="Path to diary text (default: data/diary_text.txt).")
args = parser.parse_args()

# Load the diary text
diary_path = args.path
print(f"\n📖 Loading diary text from: {diary_path}")
try:
    with open(diary_path, 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("❌ Could not find the diary text file. Make sure it exists at the given path.")
    input("\nPress Enter to exit...")
    sys.exit(1)

print(f"✅ Loaded {len(text)} characters of diary entries")
print()
print("=" * 60)
print(" ANALYZING DIARY PATTERNS ")
print("=" * 60)

# ---- Analysis WITH stopwords (neutral, no targeted phrase) ----
print("\n🔍 FULL PHRASE ANALYSIS (including stopwords)")
print("-" * 40)

tokenizer_with_stops = TextTokenizer(
    remove_stopwords=False,  # keep all words to reflect actual phrasing
    lowercase=True,
    remove_punctuation=True
)

tokens_with_stops = tokenizer_with_stops.basic_tokenize(text)
print(f"Total words: {len(tokens_with_stops)}")

analyzer_with_stops = BigramAnalyzer()
bigrams_with_stops = analyzer_with_stops.extract_bigrams(tokens_with_stops)

# Top by raw frequency (neutral)
top_phrases = analyzer_with_stops.get_top_bigrams(args.top)
# Filter by minimum frequency if needed
top_phrases = [(bigram, freq) for bigram, freq in top_phrases if freq >= args.min_freq]
print(f"\n📊 TOP {len(top_phrases)} MOST COMMON BIGRAMS (freq ≥ {args.min_freq}):")
print("-" * 40)
for i, (bigram, freq) in enumerate(top_phrases, 1):
    bigram_str = ' '.join(bigram)
    bar = '█' * min(freq * 2, 40)
    print(f"{i:2}. {bigram_str:25} {freq:2}x {bar}")

# ---- Analysis WITHOUT stopwords (content words) ----
print("\n\n🔍 CONTENT WORD ANALYSIS (stopwords removed)")
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

top_content = analyzer_no_stops.get_top_bigrams(args.top)
# Filter by minimum frequency if needed
top_content = [(bigram, freq) for bigram, freq in top_content if freq >= args.min_freq]
print(f"\n📊 TOP {len(top_content)} CONTENT WORD BIGRAMS (freq ≥ {args.min_freq}):")
print("-" * 40)
for i, (bigram, freq) in enumerate(top_content, 1):
    bigram_str = ' '.join(bigram)
    bar = '█' * min(freq * 3, 40)
    print(f"{i:2}. {bigram_str:25} {freq:2}x {bar}")

# ---- Optional: simple thematic pass (keyword buckets) ----
print("\n\n🎨 THEMATIC PATTERNS IN THE DIARY (neutral keyword buckets):")
print("-" * 40)

themes = {
    'design & creation': ['design', 'designing', 'build', 'building', 'create', 'creating'],
    'emotions & insight': ['feel', 'feeling', 'insight', 'emotional', 'intelligence'],
    'observation': ['notice', 'noticing', 'see', 'seeing', 'read', 'reading', 'watch', 'watching'],
    'people & audience': ['people', 'players', 'audience', 'someone', 'everyone'],
    'absence & failure': ['absent', 'absence', 'missing', 'fail', 'failure', 'wrong']
}

for theme, keywords in themes.items():
    theme_bigrams = []
    for bigram, freq in analyzer_no_stops.bigram_freq.items():
        if any(keyword in bigram[0].lower() or keyword in bigram[1].lower() for keyword in keywords):
            theme_bigrams.append((bigram, freq))
    if theme_bigrams:
        theme_bigrams.sort(key=lambda x: x[1], reverse=True)
        print(f"\n📌 {theme.upper()}:")
        for bigram, freq in theme_bigrams[:3]:  # Top 3 per theme
            print(f"   • '{' '.join(bigram)}' ({freq}x)")

# ---- Statistical collocations (PMI) ----
print("\n\n📈 STATISTICALLY SIGNIFICANT COLLOCATIONS (PMI)")
print(f"(Pairs that co-occur more than chance; min freq ≥ {args.min_freq})")
print("-" * 40)

# Expecting analyzer_no_stops.find_collocations to return dict with 'pmi' list
collocations = analyzer_no_stops.find_collocations(tokens_no_stops, n=args.top, min_freq=args.min_freq)
pmi_list = collocations.get('pmi', []) if isinstance(collocations, dict) else []
if pmi_list:
    for i, bigram in enumerate(pmi_list[:args.top], 1):
        print(f"{i:2}. {' '.join(bigram)}")
else:
    print("No PMI collocations found with current thresholds.")

# ---- Save results ----
print("\n\n💾 SAVING DETAILED RESULTS...")
print("-" * 40)

df_with = analyzer_with_stops.export_results(tokens_with_stops, 'diary_analysis_with_all_words.csv')
df_without = analyzer_no_stops.export_results(tokens_no_stops, 'diary_analysis_content_words.csv')

print("✅ Created two files:")
print("   1. diary_analysis_with_all_words.csv  (includes stopwords)")
print("   2. diary_analysis_content_words.csv  (stopwords removed)")

print()
print("=" * 60)
print(" ✨ ANALYSIS COMPLETE! ")
print("=" * 60)
print()
print("KEY FINDINGS (neutral):")
if top_phrases:
    print(f"• Most common bigram (with stopwords): '{' '.join(top_phrases[0][0])}' ({top_phrases[0][1]}x)")
else:
    print("• No bigrams met the frequency threshold in the with-stopwords pass.")
print(f"• Total unique bigrams (with stopwords): {len(set(bigrams_with_stops))}")
print(f"• Total unique bigrams (without stopwords): {len(set(bigrams_no_stops))}")
print()
print("📁 Open the CSV files in Excel to explore all patterns!")
print()
input("Press Enter to exit...")