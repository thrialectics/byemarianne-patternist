"""
Bigram Semantic Analysis Module
Provides bigram extraction, analysis, and semantic similarity computation
"""

import nltk
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.probability import FreqDist
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class BigramAnalyzer:
    """
    A comprehensive bigram analysis class for semantic text analysis
    """
    
    def __init__(self):
        """Initialize the bigram analyzer"""
        self.bigrams = []
        self.bigram_freq = Counter()
        self.bigram_measures = BigramAssocMeasures()
        
    def extract_bigrams(self, tokens):
        """
        Extract bigrams from tokenized text
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: List of bigram tuples
        """
        bigrams = list(nltk.bigrams(tokens))
        self.bigrams.extend(bigrams)
        self.bigram_freq.update(bigrams)
        return bigrams
    
    def get_top_bigrams(self, n=20):
        """
        Get the most frequent bigrams
        
        Args:
            n (int): Number of top bigrams to return
            
        Returns:
            list: List of (bigram, frequency) tuples
        """
        return self.bigram_freq.most_common(n)
    
    def find_collocations(self, tokens, n=20, min_freq=3):
        """
        Find significant bigram collocations using statistical measures
        
        Args:
            tokens (list): List of tokens
            n (int): Number of collocations to return
            min_freq (int): Minimum frequency threshold
            
        Returns:
            dict: Dictionary with different association measures
        """
        # Create BigramCollocationFinder
        bigram_finder = BigramCollocationFinder.from_words(tokens)
        bigram_finder.apply_freq_filter(min_freq)
        
        # Calculate different association measures
        results = {
            'pmi': bigram_finder.nbest(self.bigram_measures.pmi, n),
            'chi_square': bigram_finder.nbest(self.bigram_measures.chi_sq, n),
            'likelihood_ratio': bigram_finder.nbest(self.bigram_measures.likelihood_ratio, n),
            'student_t': bigram_finder.nbest(self.bigram_measures.student_t, n),
            'raw_freq': bigram_finder.nbest(self.bigram_measures.raw_freq, n)
        }
        
        return results
    
    def calculate_bigram_scores(self, tokens, measure='pmi'):
        """
        Calculate association scores for all bigrams
        
        Args:
            tokens (list): List of tokens
            measure (str): Association measure to use
            
        Returns:
            list: List of (bigram, score) tuples
        """
        bigram_finder = BigramCollocationFinder.from_words(tokens)
        
        measure_funcs = {
            'pmi': self.bigram_measures.pmi,
            'chi_square': self.bigram_measures.chi_sq,
            'likelihood_ratio': self.bigram_measures.likelihood_ratio,
            'student_t': self.bigram_measures.student_t
        }
        
        if measure not in measure_funcs:
            raise ValueError(f"Unknown measure: {measure}")
            
        scores = bigram_finder.score_ngrams(measure_funcs[measure])
        return scores
    
    def semantic_similarity_matrix(self, bigrams):
        """
        Create a semantic similarity matrix for bigrams using TF-IDF
        
        Args:
            bigrams (list): List of bigram tuples
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        # Convert bigrams to strings
        bigram_strings = [' '.join(bigram) for bigram in bigrams]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(bigram_strings)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix
    
    def analyze_bigram_context(self, text, target_bigram):
        """
        Analyze the context around a specific bigram
        
        Args:
            text (str): Original text
            target_bigram (tuple): Target bigram to analyze
            
        Returns:
            dict: Context analysis results
        """
        sentences = nltk.sent_tokenize(text)
        bigram_str = ' '.join(target_bigram)
        
        context = {
            'occurrences': 0,
            'sentences': [],
            'positions': []
        }
        
        for i, sentence in enumerate(sentences):
            if bigram_str.lower() in sentence.lower():
                context['occurrences'] += 1
                context['sentences'].append(sentence)
                context['positions'].append(i)
                
        return context
    
    def create_bigram_network(self, tokens, min_freq=2):
        """
        Create a network representation of bigrams
        
        Args:
            tokens (list): List of tokens
            min_freq (int): Minimum frequency for inclusion
            
        Returns:
            dict: Network data with nodes and edges
        """
        # Count bigram frequencies
        bigram_finder = BigramCollocationFinder.from_words(tokens)
        bigram_finder.apply_freq_filter(min_freq)
        
        # Create network structure
        network = {
            'nodes': set(),
            'edges': []
        }
        
        for (word1, word2), freq in bigram_finder.ngram_fd.items():
            network['nodes'].add(word1)
            network['nodes'].add(word2)
            network['edges'].append({
                'source': word1,
                'target': word2,
                'weight': freq
            })
            
        network['nodes'] = list(network['nodes'])
        return network
    
    def generate_bigram_statistics(self, tokens):
        """
        Generate comprehensive statistics about bigrams
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            dict: Statistical summary
        """
        bigrams = list(nltk.bigrams(tokens))
        bigram_freq = Counter(bigrams)
        
        # Calculate statistics
        stats = {
            'total_bigrams': len(bigrams),
            'unique_bigrams': len(bigram_freq),
            'avg_frequency': np.mean(list(bigram_freq.values())),
            'std_frequency': np.std(list(bigram_freq.values())),
            'max_frequency': max(bigram_freq.values()) if bigram_freq else 0,
            'min_frequency': min(bigram_freq.values()) if bigram_freq else 0,
            'coverage': len(bigram_freq) / len(bigrams) if bigrams else 0
        }
        
        return stats
    
    def visualize_bigram_frequency(self, n=20, save_path=None):
        """
        Create a bar chart of top bigram frequencies
        
        Args:
            n (int): Number of top bigrams to display
            save_path (str): Path to save the figure
        """
        top_bigrams = self.get_top_bigrams(n)
        
        if not top_bigrams:
            print("No bigrams to visualize")
            return
            
        bigrams_str = [' '.join(bigram) for bigram, _ in top_bigrams]
        frequencies = [freq for _, freq in top_bigrams]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(bigrams_str)), frequencies)
        plt.xticks(range(len(bigrams_str)), bigrams_str, rotation=45, ha='right')
        plt.xlabel('Bigrams')
        plt.ylabel('Frequency')
        plt.title(f'Top {n} Most Frequent Bigrams')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def create_bigram_wordcloud(self, max_words=50, save_path=None):
        """
        Create a word cloud from bigrams
        
        Args:
            max_words (int): Maximum number of bigrams in the cloud
            save_path (str): Path to save the figure
        """
        # Convert bigrams to string format for wordcloud
        bigram_dict = {' '.join(bigram): freq 
                      for bigram, freq in self.bigram_freq.items()}
        
        if not bigram_dict:
            print("No bigrams to create wordcloud")
            return
            
        wordcloud = WordCloud(width=800, height=400, 
                             max_words=max_words,
                             background_color='white').generate_from_frequencies(bigram_dict)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Bigram Word Cloud')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def export_results(self, tokens, output_file='bigram_analysis.csv'):
        """
        Export bigram analysis results to CSV
        
        Args:
            tokens (list): List of tokens
            output_file (str): Output file path
        """
        # Calculate various measures
        scores_pmi = self.calculate_bigram_scores(tokens, 'pmi')
        scores_chi = self.calculate_bigram_scores(tokens, 'chi_square')
        
        # Create DataFrame
        data = []
        for i, (bigram, pmi_score) in enumerate(scores_pmi[:100]):  # Top 100
            chi_score = next((score for bg, score in scores_chi if bg == bigram), 0)
            freq = self.bigram_freq.get(bigram, 0)
            
            data.append({
                'bigram': ' '.join(bigram),
                'word1': bigram[0],
                'word2': bigram[1],
                'frequency': freq,
                'pmi_score': pmi_score,
                'chi_square_score': chi_score
            })
            
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")
        return df
