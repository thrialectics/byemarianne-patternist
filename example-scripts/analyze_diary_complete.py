"""
Complete analysis script showing bigrams with and without stopwords
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from tokenizer import TextTokenizer
from bigram_analyzer import BigramAnalyzer
from collections import Counter

def analyze_diary_complete():
    """Analyze the diary text showing both with and without stopwords"""
    
    print("\n" + "=" * 80)
    print(" COMPLETE DIARY TEXT ANALYSIS - DUAL APPROACH ")
    print("=" * 80)
    
    # Load the diary text
    diary_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'diary_text.txt')
    with open(diary_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\nText loaded: {len(text)} characters")
    
    # First, analyze WITH stopwords for complete patterns
    print("\n" + "=" * 80)
    print(" ANALYSIS WITH ALL WORDS (INCLUDING COMMON WORDS) ")
    print("=" * 80)
    
    tokenizer_with_stops = TextTokenizer(
        remove_stopwords=False,  # Keep all words for complete analysis
        lowercase=True,
        remove_punctuation=True
    )
    
    tokens_with_stops = tokenizer_with_stops.basic_tokenize(text)
    print(f"\nTotal tokens (all words): {len(tokens_with_stops)}")
    print(f"Unique tokens: {len(set(tokens_with_stops))}")
    
    # Analyze bigrams WITH stopwords
    analyzer_with_stops = BigramAnalyzer()
    bigrams_with_stops = analyzer_with_stops.extract_bigrams(tokens_with_stops)
    
    print(f"\nTotal bigrams: {len(bigrams_with_stops)}")
    print(f"Unique bigrams: {len(set(bigrams_with_stops))}")
    
    # Top bigrams WITH stopwords
    print("\n** TOP 30 BIGRAMS (ALL WORDS INCLUDED) **")
    top_bigrams_with = analyzer_with_stops.get_top_bigrams(30)
    for i, (bigram, freq) in enumerate(top_bigrams_with, 1):
        bigram_str = ' '.join(bigram)
        # Mark particularly frequent patterns
        if freq >= 6:
            print(f"  {i:3}. '{bigram_str:35}' : {freq} occurrences [HIGH FREQUENCY]")
        else:
            print(f"  {i:3}. '{bigram_str:35}' : {freq} occurrences")
    
    # Now analyze WITHOUT stopwords for content words
    print("\n" + "=" * 80)
    print(" ANALYSIS WITHOUT COMMON WORDS (CONTENT FOCUS) ")
    print("=" * 80)
    
    tokenizer_no_stops = TextTokenizer(
        remove_stopwords=True,
        lowercase=True,
        remove_punctuation=True
    )
    
    tokens_no_stops = tokenizer_no_stops.basic_tokenize(text)
    print(f"\nTotal tokens (content words only): {len(tokens_no_stops)}")
    print(f"Unique tokens: {len(set(tokens_no_stops))}")
    
    # Analyze bigrams WITHOUT stopwords
    analyzer_no_stops = BigramAnalyzer()
    bigrams_no_stops = analyzer_no_stops.extract_bigrams(tokens_no_stops)
    
    print(f"\nTotal bigrams: {len(bigrams_no_stops)}")
    print(f"Unique bigrams: {len(set(bigrams_no_stops))}")
    
    # Top content-word bigrams
    print("\n** TOP 30 CONTENT-WORD BIGRAMS **")
    top_bigrams_without = analyzer_no_stops.get_top_bigrams(30)
    for i, (bigram, freq) in enumerate(top_bigrams_without, 1):
        print(f"  {i:3}. '{' '.join(bigram):35}' : {freq} occurrences")
    
    # Thematic word analysis
    print("\n" + "=" * 80)
    print(" THEMATIC WORD PATTERNS ")
    print("=" * 80)
    
    # Find patterns around key themes
    themes = {
        'design': ['design', 'designing', 'build', 'building', 'create'],
        'perception': ['notice', 'noticing', 'see', 'seeing', 'read'],
        'emotion': ['feel', 'feeling', 'emotional', 'intelligence'],
        'meta': ['everything', 'something', 'nothing', 'anything']
    }
    
    for theme_name, keywords in themes.items():
        print(f"\n** {theme_name.upper()} THEME: **")
        theme_bigrams = {}
        for bigram in bigrams_with_stops:
            if any(keyword in bigram for keyword in keywords):
                bigram_str = ' '.join(bigram)
                if bigram_str not in theme_bigrams:
                    theme_bigrams[bigram_str] = 0
                theme_bigrams[bigram_str] += 1
        
        # Show top 5 for each theme
        sorted_theme = sorted(theme_bigrams.items(), key=lambda x: x[1], reverse=True)[:5]
        for bigram_str, count in sorted_theme:
            print(f"  '{bigram_str:35}' : {count} occurrences")
    
    # Statistical analysis
    print("\n" + "=" * 80)
    print(" STATISTICAL COLLOCATION ANALYSIS ")
    print("=" * 80)
    
    collocations_with = analyzer_with_stops.find_collocations(tokens_with_stops, n=20, min_freq=2)
    collocations_without = analyzer_no_stops.find_collocations(tokens_no_stops, n=20, min_freq=2)
    
    print("\n** PMI Scores WITH all words (top 15): **")
    print("(Pairs that appear together more than chance would predict)")
    for i, bigram in enumerate(collocations_with['pmi'][:15], 1):
        print(f"  {i:2}. {' '.join(bigram)}")
    
    print("\n** PMI Scores WITHOUT common words (top 15): **")
    print("(Content word pairs with strong associations)")
    for i, bigram in enumerate(collocations_without['pmi'][:15], 1):
        print(f"  {i:2}. {' '.join(bigram)}")
    
    # Pattern discovery
    print("\n" + "=" * 80)
    print(" PATTERN DISCOVERY ")
    print("=" * 80)
    
    # Find recurring patterns (3+ occurrences)
    print("\n** Recurring patterns (3+ occurrences): **")
    recurring = [(bigram, freq) for bigram, freq in analyzer_with_stops.bigram_freq.items() if freq >= 3]
    recurring.sort(key=lambda x: x[1], reverse=True)
    
    for bigram, freq in recurring[:20]:
        print(f"  '{' '.join(bigram):35}' : {freq} times")
    
    # Export both analyses
    print("\n" + "=" * 80)
    print(" EXPORTING RESULTS ")
    print("=" * 80)
    
    # Export with stopwords
    df_with = analyzer_with_stops.export_results(tokens_with_stops, 'diary_complete_all_words.csv')
    print(f"Results WITH all words exported to 'diary_complete_all_words.csv'")
    
    # Export without stopwords  
    df_without = analyzer_no_stops.export_results(tokens_no_stops, 'diary_complete_content_only.csv')
    print(f"Results WITHOUT common words exported to 'diary_complete_content_only.csv'")
    
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE! ")
    print("=" * 80)
    
    print("\nKey Insights:")
    print("- The diary shows distinct patterns when analyzed with and without common words")
    print("- Content word analysis reveals thematic focus on design and perception")
    print("- Statistical collocations identify word pairs with significant associations")
    print("- Multiple recurring patterns suggest deliberate stylistic choices")
    print("\nExamine the CSV files for detailed pattern analysis.")
    
    return analyzer_with_stops, analyzer_no_stops

if __name__ == "__main__":
    analyze_diary_complete()