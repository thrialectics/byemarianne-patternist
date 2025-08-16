"""
NLTK Tokenizer Module
Provides various tokenization methods for text processing
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import re

# Download required NLTK data
def download_nltk_data():
    """Download necessary NLTK data packages"""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except:
                nltk.download(resource)

class TextTokenizer:
    """
    A comprehensive tokenizer class for text processing
    """
    
    def __init__(self, remove_stopwords=True, lowercase=True, remove_punctuation=True):
        """
        Initialize the tokenizer with configuration options
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            lowercase (bool): Whether to convert text to lowercase
            remove_punctuation (bool): Whether to remove punctuation
        """
        download_nltk_data()
        
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
        # Initialize tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.regexp_tokenizer = RegexpTokenizer(r'\w+')
        
    def basic_tokenize(self, text):
        """
        Basic word tokenization
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        tokens = word_tokenize(text)
        
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
            
        if self.remove_punctuation:
            # Remove tokens that are only punctuation AND filter out tokens containing apostrophes
            # Also ensure tokens are actual words (contain at least one letter)
            tokens = [token for token in tokens 
                     if token not in string.punctuation 
                     and "'" not in token 
                     and any(c.isalpha() for c in token)]
            
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
            
        return tokens
    
    def sentence_tokenize(self, text):
        """
        Tokenize text into sentences
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        return sent_tokenize(text)
    
    def regexp_tokenize(self, text):
        """
        Tokenize using regular expressions (removes punctuation automatically)
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        # Use a more restrictive regex that only captures whole words (no apostrophes)
        # This will split on apostrophes and only keep alphabetic sequences
        import re
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        
        if self.lowercase:
            tokens = [token.lower() for token in tokens]
            
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
            
        return tokens
    
    def stem_tokens(self, tokens):
        """
        Apply Porter stemming to tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: List of stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """
        Apply lemmatization to tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_pos_tags(self, tokens):
        """
        Get part-of-speech tags for tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: List of (token, pos_tag) tuples
        """
        return nltk.pos_tag(tokens)
    
    def tokenize_and_clean(self, text, stem=False, lemmatize=False):
        """
        Complete tokenization pipeline with optional stemming/lemmatization
        
        Args:
            text (str): Input text
            stem (bool): Whether to apply stemming
            lemmatize (bool): Whether to apply lemmatization
            
        Returns:
            list: Processed tokens
        """
        # Basic tokenization
        tokens = self.basic_tokenize(text)
        
        # Apply stemming or lemmatization
        if stem:
            tokens = self.stem_tokens(tokens)
        elif lemmatize:
            tokens = self.lemmatize_tokens(tokens)
            
        return tokens
