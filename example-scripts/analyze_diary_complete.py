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
    print(" COMPLETE DIARY TEXT ANALYSIS - INCLUDING 'IN EVERYTHING' ")
    print("=" * 80)
    
    # Load the diary text
    diary_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'diary_text.txt')
    with open(diary_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\nText loaded: {len(text)} characters")
    
    # First, analyze WITH stopwords to see "in everything"
    print("\n" + "=" * 80)
    print(" ANALYSIS WITH STOPWORDS (TO CAPTURE 'IN EVERYTHING') ")
    print("=" * 80)
    
    tokenizer_with_stops = TextTokenizer(
        remove_stopwords=False,  # Keep stopwords to see "in everything"
        lowercase=True,
        remove_punctuation=True
    )
    
    tokens_with_stops = tokenizer_with_stops.basic_tokenize(text)
    print(f"\nTotal tokens (with stopwords): {len(tokens_with_stops)}")
    print(f"Unique tokens: {len(set(tokens_with_stops))}")
    
    # Analyze bigrams WITH stopwords
    analyzer_with_stops = BigramAnalyzer()
    bigrams_with_stops = analyzer_with_stops.extract_bigrams(tokens_with_stops)
    
    print(f"\nTotal bigrams: {len(bigrams_with_stops)}")
    print(f"Unique bigrams: {len(set(bigrams_with_stops))}")
    
    # Count occurrences of "in everything"
    in_everything_count = sum(1 for bigram in bigrams_with_stops 
                             if bigram == ('in', 'everything'))
    print(f"\n** 'in everything' appears {in_everything_count} times **")
    
    # Top bigrams WITH stopwords
    print("\n** TOP 30 BIGRAMS (WITH STOPWORDS) **")
    top_bigrams_with = analyzer_with_stops.get_top_bigrams(30)
    for i, (bigram, freq) in enumerate(top_bigrams_with, 1):
        bigram_str = ' '.join(bigram)
        if bigram_str == 'in everything':
            print(f"  {i:3}. '{bigram_str:35}' : {freq} occurrences *** TARGET BIGRAM ***")
        else:
            print(f"  {i:3}. '{bigram_str:35}' : {freq} occurrences")
    
    # Now analyze WITHOUT stopwords for content words
    print("\n" + "=" * 80)
    print(" ANALYSIS WITHOUT STOPWORDS (CONTENT WORDS ONLY) ")
    print("=" * 80)
    
    tokenizer_no_stops = TextTokenizer(
        remove_stopwords=True,
        lowercase=True,
        remove_punctuation=True
    )
    
    tokens_no_stops = tokenizer_no_stops.basic_tokenize(text)
    print(f"\nTotal tokens (without stopwords): {len(tokens_no_stops)}")
    print(f"Unique tokens: {len(set(tokens_no_stops))}")
    
    # Analyze bigrams WITHOUT stopwords
    analyzer_no_stops = BigramAnalyzer()
    bigrams_no_stops = analyzer_no_stops.extract_bigrams(tokens_no_stops)
    
    print(f"\nTotal bigrams: {len(bigrams_no_stops)}")
    print(f"Unique bigrams: {len(set(bigrams_no_stops))}")
    
    # Top content-word bigrams
    print("\n** TOP 30 CONTENT-WORD BIGRAMS (WITHOUT STOPWORDS) **")
    top_bigrams_without = analyzer_no_stops.get_top_bigrams(30)
    for i, (bigram, freq) in enumerate(top_bigrams_without, 1):
        print(f"  {i:3}. '{' '.join(bigram):35}' : {freq} occurrences")
    
    # Find all bigrams containing "everything"
    print("\n" + "=" * 80)
    print(" ALL BIGRAMS CONTAINING 'EVERYTHING' ")
    print("=" * 80)
    
    everything_bigrams = {}
    for bigram in bigrams_with_stops:
        if 'everything' in bigram:
            bigram_str = ' '.join(bigram)
            if bigram_str not in everything_bigrams:
                everything_bigrams[bigram_str] = 0
            everything_bigrams[bigram_str] += 1
    
    print("\n** With stopwords included: **")
    for bigram_str, count in sorted(everything_bigrams.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{bigram_str:35}' : {count} occurrences")
    
    # Statistical analysis on version WITH stopwords for "in everything"
    print("\n" + "=" * 80)
    print(" STATISTICAL ANALYSIS (WITH STOPWORDS) ")
    print("=" * 80)
    
    collocations = analyzer_with_stops.find_collocations(tokens_with_stops, n=20, min_freq=2)
    
    print("\n** PMI Scores (top 20): **")
    for i, bigram in enumerate(collocations['pmi'][:20], 1):
        bigram_str = ' '.join(bigram)
        if bigram_str == 'in everything':
            print(f"  {i:2}. {bigram_str} *** TARGET BIGRAM ***")
        else:
            print(f"  {i:2}. {bigram_str}")
    
    # Context analysis for "in everything"
    print("\n" + "=" * 80)
    print(" CONTEXT ANALYSIS FOR 'IN EVERYTHING' ")
    print("=" * 80)
    
    context = analyzer_with_stops.analyze_bigram_context(text, ('in', 'everything'))
    print(f"\nOccurrences: {context['occurrences']}")
    print(f"Found in sentences at positions: {context['positions']}")
    
    if context['sentences']:
        print("\n** All sentences containing 'in everything': **")
        for i, sentence in enumerate(context['sentences'], 1):
            # Highlight the phrase
            highlighted = sentence.replace('in everything', '>>> IN EVERYTHING <<<')
            print(f"\n{i}. {highlighted}")
    
    # Export both analyses
    print("\n" + "=" * 80)
    print(" EXPORTING RESULTS ")
    print("=" * 80)
    
    # Export with stopwords
    df_with = analyzer_with_stops.export_results(tokens_with_stops, 'diary_with_stopwords.csv')
    print(f"Results WITH stopwords exported to 'diary_with_stopwords.csv'")
    
    # Export without stopwords  
    df_without = analyzer_no_stops.export_results(tokens_no_stops, 'diary_without_stopwords.csv')
    print(f"Results WITHOUT stopwords exported to 'diary_without_stopwords.csv'")
    
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE! ")
    print("=" * 80)
    
    print("\nKey Findings:")
    print(f"- 'in everything' appears {in_everything_count} times in the text")
    print(f"- It ranks #{[i for i, (b, _) in enumerate(top_bigrams_with, 1) if ' '.join(b) == 'in everything'][0] if any(' '.join(b) == 'in everything' for b, _ in top_bigrams_with) else 'not in top 30'} in frequency when stopwords are included")
    print("- When stopwords are removed, content-word bigrams like 'uncanny forcefulness' become visible")
    print("- The text shows a pattern of using 'everything' to emphasize totality and completeness")
    
    return analyzer_with_stops, analyzer_no_stops

if __name__ == "__main__":
    analyze_diary_complete()
