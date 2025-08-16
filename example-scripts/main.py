"""
Main script for NLTK Tokenizer and Bigram Semantic Analysis
Demonstrates various tokenization and bigram analysis capabilities
"""

import os
import sys
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from tokenizer import TextTokenizer
from bigram_analyzer import BigramAnalyzer
import nltk

def load_diary_text():
    """
    Load the diary text for analysis
        
    Returns:
        str: Diary text content
    """
    diary_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'diary_text.txt')
    with open(diary_file, 'r', encoding='utf-8') as f:
        return f.read()

def demonstrate_tokenization(text):
    """
    Demonstrate various tokenization techniques
    
    Args:
        text (str): Input text
    """
    print("=" * 80)
    print("TOKENIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Initialize tokenizer
    tokenizer = TextTokenizer(remove_stopwords=True, lowercase=True, remove_punctuation=True)
    
    # Basic tokenization
    print("\n1. BASIC TOKENIZATION (with stopword removal):")
    tokens = tokenizer.basic_tokenize(text)
    print(f"   Total tokens: {len(tokens)}")
    print(f"   First 20 tokens: {tokens[:20]}")
    
    # Sentence tokenization
    print("\n2. SENTENCE TOKENIZATION:")
    sentences = tokenizer.sentence_tokenize(text)
    print(f"   Total sentences: {len(sentences)}")
    print(f"   First sentence: {sentences[0] if sentences else 'No sentences found'}")
    
    # Tokenization without stopword removal
    tokenizer_with_stopwords = TextTokenizer(remove_stopwords=False, lowercase=True, remove_punctuation=True)
    tokens_with_stopwords = tokenizer_with_stopwords.basic_tokenize(text)
    print(f"\n3. TOKENS WITH STOPWORDS:")
    print(f"   Total tokens: {len(tokens_with_stopwords)}")
    print(f"   Stopwords removed: {len(tokens_with_stopwords) - len(tokens)} words")
    
    # Stemming
    print("\n4. STEMMED TOKENS:")
    stemmed = tokenizer.stem_tokens(tokens[:10])
    for original, stemmed_word in zip(tokens[:10], stemmed):
        print(f"   {original:20} -> {stemmed_word}")
    
    # Lemmatization
    print("\n5. LEMMATIZED TOKENS:")
    lemmatized = tokenizer.lemmatize_tokens(tokens[:10])
    for original, lemma in zip(tokens[:10], lemmatized):
        print(f"   {original:20} -> {lemma}")
    
    # POS tagging
    print("\n6. PART-OF-SPEECH TAGGING:")
    pos_tags = tokenizer.get_pos_tags(tokens[:15])
    for word, tag in pos_tags:
        print(f"   {word:15} : {tag}")
    
    return tokens

def demonstrate_bigram_analysis(tokens, text):
    """
    Demonstrate bigram analysis capabilities
    
    Args:
        tokens (list): List of tokens
        text (str): Original text
    """
    print("\n" + "=" * 80)
    print("BIGRAM ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = BigramAnalyzer()
    
    # Extract bigrams
    print("\n1. EXTRACTING BIGRAMS:")
    bigrams = analyzer.extract_bigrams(tokens)
    print(f"   Total bigrams: {len(bigrams)}")
    print(f"   Unique bigrams: {len(set(bigrams))}")
    print(f"   First 10 bigrams: {bigrams[:10]}")
    
    # Top frequent bigrams
    print("\n2. TOP FREQUENT BIGRAMS:")
    top_bigrams = analyzer.get_top_bigrams(10)
    for i, (bigram, freq) in enumerate(top_bigrams, 1):
        print(f"   {i:2}. {' '.join(bigram):30} : {freq} occurrences")
    
    # Collocations with different measures
    print("\n3. STATISTICAL COLLOCATIONS:")
    collocations = analyzer.find_collocations(tokens, n=5, min_freq=2)
    
    print("\n   PMI (Pointwise Mutual Information):")
    for bigram in collocations['pmi'][:5]:
        print(f"      {' '.join(bigram)}")
    
    print("\n   Chi-Square Test:")
    for bigram in collocations['chi_square'][:5]:
        print(f"      {' '.join(bigram)}")
    
    print("\n   Likelihood Ratio:")
    for bigram in collocations['likelihood_ratio'][:5]:
        print(f"      {' '.join(bigram)}")
    
    # Bigram scores
    print("\n4. BIGRAM ASSOCIATION SCORES (PMI):")
    scores = analyzer.calculate_bigram_scores(tokens, 'pmi')[:5]
    for bigram, score in scores:
        print(f"   {' '.join(bigram):30} : {score:.4f}")
    
    # Bigram statistics
    print("\n5. BIGRAM STATISTICS:")
    stats = analyzer.generate_bigram_statistics(tokens)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key:20} : {value:.4f}")
        else:
            print(f"   {key:20} : {value}")
    
    # Context analysis for a specific bigram
    if top_bigrams:
        target_bigram = top_bigrams[0][0]
        print(f"\n6. CONTEXT ANALYSIS FOR '{' '.join(target_bigram)}':")
        context = analyzer.analyze_bigram_context(text, target_bigram)
        print(f"   Occurrences: {context['occurrences']}")
        print(f"   Found in sentences: {context['positions']}")
        if context['sentences']:
            print(f"   Example sentence: {context['sentences'][0][:100]}...")
    
    # Network representation
    print("\n7. BIGRAM NETWORK:")
    network = analyzer.create_bigram_network(tokens, min_freq=2)
    print(f"   Nodes (unique words): {len(network['nodes'])}")
    print(f"   Edges (bigram connections): {len(network['edges'])}")
    print(f"   Sample edges: {network['edges'][:3]}")
    
    return analyzer

def save_visualizations(analyzer, output_dir="output"):
    """
    Save visualization outputs
    
    Args:
        analyzer (BigramAnalyzer): Bigram analyzer instance
        output_dir (str): Output directory for visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SAVING VISUALIZATIONS")
    print("=" * 80)
    
    try:
        # Save frequency chart
        print("\n1. Saving bigram frequency chart...")
        analyzer.visualize_bigram_frequency(20, save_path=os.path.join(output_dir, "bigram_frequency.png"))
        print("   Saved to output/bigram_frequency.png")
    except Exception as e:
        print(f"   Could not save frequency chart: {e}")
    
    try:
        # Save word cloud
        print("\n2. Saving bigram word cloud...")
        analyzer.create_bigram_wordcloud(50, save_path=os.path.join(output_dir, "bigram_wordcloud.png"))
        print("   Saved to output/bigram_wordcloud.png")
    except Exception as e:
        print(f"   Could not save word cloud: {e}")

def main():
    """
    Main execution function
    """
    print("\n" + "=" * 80)
    print(" NLTK TOKENIZER AND BIGRAM SEMANTIC ANALYSIS ")
    print("=" * 80)
    
    # Load diary text
    print("\nLoading diary text...")
    text = load_diary_text()
    
    print(f"Text length: {len(text)} characters")
    
    # Demonstrate tokenization
    tokens = demonstrate_tokenization(text)
    
    # Demonstrate bigram analysis
    analyzer = demonstrate_bigram_analysis(tokens, text)
    
    # Export results
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    
    try:
        df = analyzer.export_results(tokens, "bigram_analysis_results.csv")
        print("\nExported analysis to 'bigram_analysis_results.csv'")
        print(f"Total rows exported: {len(df)}")
    except Exception as e:
        print(f"Could not export results: {e}")
    
    # Save visualizations (optional - may not display in terminal)
    try:
        save_visualizations(analyzer)
    except Exception as e:
        print(f"\nNote: Visualizations require matplotlib display backend. {e}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    
    print("\nYou can now:")
    print("1. Check 'bigram_analysis_results.csv' for detailed analysis")
    print("2. View visualizations in the 'output' directory (if matplotlib is configured)")
    print("3. Modify the code to analyze your own text files")
    print("4. Experiment with different tokenization and analysis parameters")

if __name__ == "__main__":
    main()
