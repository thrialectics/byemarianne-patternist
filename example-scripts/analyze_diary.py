"""
Custom analysis script for diary text
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from tokenizer import TextTokenizer
from bigram_analyzer import BigramAnalyzer

def analyze_diary_text():
    """Analyze the diary text with customized settings"""
    
    print("\n" + "=" * 80)
    print(" ANALYZING DIARY TEXT - BIGRAM SEMANTIC ANALYSIS ")
    print("=" * 80)
    
    # Load the diary text
    diary_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'diary_text.txt')
    with open(diary_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\nText loaded: {len(text)} characters")
    print(f"Preview: {text[:150]}...")
    
    # Initialize tokenizer with custom settings for this reflective text
    tokenizer = TextTokenizer(
        remove_stopwords=True,  # Keep this to focus on meaningful words
        lowercase=True,
        remove_punctuation=True
    )
    
    # Tokenize the text
    print("\n" + "-" * 60)
    print("TOKENIZATION RESULTS")
    print("-" * 60)
    
    tokens = tokenizer.basic_tokenize(text)
    print(f"Total tokens (without stopwords): {len(tokens)}")
    print(f"Unique tokens: {len(set(tokens))}")
    print(f"\nFirst 30 tokens: {tokens[:30]}")
    
    # Get tokens with stopwords for comparison
    tokenizer_with_stops = TextTokenizer(
        remove_stopwords=False,
        lowercase=True,
        remove_punctuation=True
    )
    tokens_with_stops = tokenizer_with_stops.basic_tokenize(text)
    print(f"\nTotal tokens (with stopwords): {len(tokens_with_stops)}")
    print(f"Stopwords removed: {len(tokens_with_stops) - len(tokens)}")
    
    # Sentence analysis
    sentences = tokenizer.sentence_tokenize(text)
    print(f"\nTotal sentences: {len(sentences)}")
    
    # POS tagging on a sample
    print("\n" + "-" * 60)
    print("PART-OF-SPEECH ANALYSIS (Sample)")
    print("-" * 60)
    sample_tokens = tokens[50:65]  # Take a sample from middle
    pos_tags = tokenizer.get_pos_tags(sample_tokens)
    for word, tag in pos_tags:
        print(f"  {word:20} : {tag}")
    
    # Initialize bigram analyzer
    analyzer = BigramAnalyzer()
    
    # Extract and analyze bigrams
    print("\n" + "-" * 60)
    print("BIGRAM ANALYSIS")
    print("-" * 60)
    
    bigrams = analyzer.extract_bigrams(tokens)
    print(f"\nTotal bigrams: {len(bigrams)}")
    print(f"Unique bigrams: {len(set(bigrams))}")
    
    # Top frequent bigrams
    print("\n** TOP 25 MOST FREQUENT BIGRAMS **")
    top_bigrams = analyzer.get_top_bigrams(25)
    for i, (bigram, freq) in enumerate(top_bigrams, 1):
        print(f"  {i:3}. '{' '.join(bigram):35}' : {freq} occurrences")
    
    # Statistical collocations
    print("\n" + "-" * 60)
    print("STATISTICAL COLLOCATIONS")
    print("-" * 60)
    
    collocations = analyzer.find_collocations(tokens, n=15, min_freq=2)
    
    print("\n** Pointwise Mutual Information (PMI) - Top 15 **")
    print("  (Words that appear together more than expected by chance)")
    for i, bigram in enumerate(collocations['pmi'], 1):
        print(f"  {i:2}. {' '.join(bigram)}")
    
    print("\n** Chi-Square Test - Top 15 **")
    print("  (Statistically significant word associations)")
    for i, bigram in enumerate(collocations['chi_square'], 1):
        print(f"  {i:2}. {' '.join(bigram)}")
    
    print("\n** Likelihood Ratio - Top 15 **")
    print("  (Most informative bigrams)")
    for i, bigram in enumerate(collocations['likelihood_ratio'], 1):
        print(f"  {i:2}. {' '.join(bigram)}")
    
    # Thematic bigrams - look for specific themes
    print("\n" + "-" * 60)
    print("THEMATIC BIGRAM ANALYSIS")
    print("-" * 60)
    
    # Find bigrams related to key themes in the diary
    themes = {
        'design': ['design', 'designing', 'build', 'building', 'create', 'creating'],
        'emotion': ['feel', 'feeling', 'emotional', 'intelligence', 'suffer', 'suffering'],
        'notice': ['notice', 'noticing', 'see', 'seeing', 'read', 'reading'],
        'people': ['people', 'players', 'audience', 'someone', 'everyone'],
        'absence': ['absent', 'absence', 'missing', 'fail', 'failure', 'wrong']
    }
    
    theme_bigrams = {}
    for theme, keywords in themes.items():
        theme_bigrams[theme] = []
        for bigram, freq in analyzer.bigram_freq.items():
            if any(keyword in bigram[0].lower() or keyword in bigram[1].lower() 
                   for keyword in keywords):
                theme_bigrams[theme].append((bigram, freq))
        theme_bigrams[theme].sort(key=lambda x: x[1], reverse=True)
    
    for theme, bigrams_list in theme_bigrams.items():
        if bigrams_list:
            print(f"\n** Theme: {theme.upper()} **")
            for bigram, freq in bigrams_list[:5]:  # Top 5 per theme
                print(f"  '{' '.join(bigram):35}' : {freq} times")
    
    # Context analysis for interesting bigrams
    print("\n" + "-" * 60)
    print("CONTEXT ANALYSIS FOR KEY BIGRAMS")
    print("-" * 60)
    
    key_bigrams = [
        ('emotional', 'intelligence'),
        ('fail', 'gracefully'),
        ('people', 'notice'),
        ('design', 'people'),
        ('everything', 'designing')
    ]
    
    for target_bigram in key_bigrams:
        if target_bigram in analyzer.bigram_freq:
            context = analyzer.analyze_bigram_context(text, target_bigram)
            if context['occurrences'] > 0:
                print(f"\n** '{' '.join(target_bigram)}' **")
                print(f"  Occurrences: {context['occurrences']}")
                if context['sentences']:
                    print(f"  Example: ...{context['sentences'][0][:150]}...")
    
    # Generate statistics
    print("\n" + "-" * 60)
    print("BIGRAM STATISTICS SUMMARY")
    print("-" * 60)
    
    stats = analyzer.generate_bigram_statistics(tokens)
    print(f"\n  Total bigrams:     {stats['total_bigrams']}")
    print(f"  Unique bigrams:    {stats['unique_bigrams']}")
    print(f"  Avg frequency:     {stats['avg_frequency']:.2f}")
    print(f"  Std deviation:     {stats['std_frequency']:.2f}")
    print(f"  Max frequency:     {stats['max_frequency']}")
    print(f"  Coverage ratio:    {stats['coverage']:.2%}")
    
    # Export detailed results
    print("\n" + "-" * 60)
    print("EXPORTING RESULTS")
    print("-" * 60)
    
    df = analyzer.export_results(tokens, 'diary_bigram_analysis.csv')
    print(f"\nDetailed results exported to 'diary_bigram_analysis.csv'")
    print(f"Total entries: {len(df)}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    try:
        os.makedirs('diary_output', exist_ok=True)
        analyzer.visualize_bigram_frequency(25, save_path='diary_output/diary_bigram_frequency.png')
        print("  Frequency chart saved to 'diary_output/diary_bigram_frequency.png'")
    except Exception as e:
        print(f"  Note: Could not generate frequency chart: {e}")
    
    try:
        analyzer.create_bigram_wordcloud(60, save_path='diary_output/diary_bigram_wordcloud.png')
        print("  Word cloud saved to 'diary_output/diary_bigram_wordcloud.png'")
    except Exception as e:
        print(f"  Note: Could not generate word cloud: {e}")
    
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE! ")
    print("=" * 80)
    
    print("\nKey Insights from the Diary Text:")
    print("- The text heavily focuses on themes of design, people, and noticing")
    print("- Strong emphasis on emotional intelligence and understanding")
    print("- Recurring patterns around failure, absence, and indirect meaning")
    print("- Meta-reflection on the design process and audience understanding")
    
    return analyzer, tokens

if __name__ == "__main__":
    analyzer, tokens = analyze_diary_text()
