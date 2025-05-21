import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re
import json
import csv
import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Text processing libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
import string

# For visualization
from wordcloud import WordCloud
from IPython.display import display, clear_output
from ipywidgets import widgets, HBox, VBox, Layout

# For feature extraction and modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Simple models for baselines
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Try to download necessary NLTK resources safely
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {e}")
    print("You may need to manually download them with nltk.download()")

# ==================== DOCUMENT LOADING AND BASIC INFO ====================

def load_text_documents(document_folder: str, 
                       extensions: List[str] = ['txt', 'csv', 'json', 'md'],
                       encoding: str = 'utf-8',
                       max_docs: Optional[int] = None) -> pd.DataFrame:
    """
    Load text documents from a folder with metadata
    
    Parameters:
    -----------
    document_folder : str
        Path to folder containing documents
    extensions : List[str]
        List of file extensions to include
    encoding : str
        File encoding to use when reading files
    max_docs : Optional[int]
        Maximum number of documents to load (None = all)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with document content and metadata
    """
    document_folder = Path(document_folder)
    doc_files = []
    
    for ext in extensions:
        doc_files.extend(document_folder.glob(f'*.{ext}'))
        doc_files.extend(document_folder.glob(f'*.{ext.upper()}'))
    
    # Limit number of documents if specified
    if max_docs:
        doc_files = doc_files[:max_docs]
    
    doc_info = []
    for doc_path in doc_files:
        try:
            # Extract creation/modification dates
            creation_time = os.path.getctime(doc_path)
            modified_time = os.path.getmtime(doc_path)
            
            # Read file based on extension
            extension = doc_path.suffix.lower()[1:]
            content = None
            
            if extension in ['txt', 'md']:
                with open(doc_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
            
            elif extension == 'json':
                with open(doc_path, 'r', encoding=encoding, errors='replace') as f:
                    json_data = json.load(f)
                    # Handle different JSON structures
                    if isinstance(json_data, dict):
                        # Try to find text content in common fields
                        for field in ['text', 'content', 'body', 'message']:
                            if field in json_data:
                                content = str(json_data[field])
                                break
                        # If no content field found, convert whole object to string
                        if content is None:
                            content = json.dumps(json_data)
                    else:
                        content = json.dumps(json_data)
            
            elif extension == 'csv':
                with open(doc_path, 'r', encoding=encoding, errors='replace') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if rows:
                        # Assume first row is header
                        headers = rows[0]
                        # Look for columns that might contain text
                        text_cols = [i for i, h in enumerate(headers) 
                                    if any(t in h.lower() for t in ['text', 'content', 'body', 'message'])]
                        if text_cols and len(rows) > 1:
                            # Get content from first text column
                            content = '\n'.join([row[text_cols[0]] for row in rows[1:] if len(row) > text_cols[0]])
                        else:
                            # Fallback: convert whole CSV to string
                            content = '\n'.join([','.join(row) for row in rows])
            
            # Create document info dictionary
            if content:
                # Extract document statistics
                word_count = len(content.split())
                char_count = len(content)
                line_count = content.count('\n') + 1
                
                info = {
                    'filename': doc_path.name,
                    'path': str(doc_path),
                    'extension': extension,
                    'size_kb': doc_path.stat().st_size / 1024,
                    'created_date': datetime.datetime.fromtimestamp(creation_time),
                    'modified_date': datetime.datetime.fromtimestamp(modified_time),
                    'word_count': word_count,
                    'char_count': char_count,
                    'line_count': line_count,
                    'content': content
                }
                doc_info.append(info)
        except Exception as e:
            print(f"Error loading {doc_path.name}: {e}")
    
    return pd.DataFrame(doc_info)

def parse_emails(content: str) -> Dict[str, Any]:
    """
    Parse email content to extract headers and body
    
    Parameters:
    -----------
    content : str
        Raw email content
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with email headers and body
    """
    # Simple regex-based email parser
    headers = {}
    
    # Find header section and body
    parts = re.split(r'\n\s*\n', content, 1)
    
    if len(parts) >= 1:
        header_section = parts[0]
        # Extract common headers
        for header in ['From', 'To', 'Cc', 'Subject', 'Date']:
            match = re.search(rf'{header}:\s*(.*?)(?:\n[A-Za-z-]+:|$)', header_section, re.DOTALL)
            if match:
                headers[header.lower()] = match.group(1).strip()
    
    # Extract body
    body = parts[1] if len(parts) > 1 else ""
    
    return {
        'headers': headers,
        'body': body
    }

def load_email_data(email_folder: str, 
                   extensions: List[str] = ['txt', 'eml'],
                   encoding: str = 'utf-8',
                   max_emails: Optional[int] = None) -> pd.DataFrame:
    """
    Load email data from files in a folder
    
    Parameters:
    -----------
    email_folder : str
        Path to folder containing email files
    extensions : List[str]
        List of file extensions to include
    encoding : str
        File encoding to use when reading files
    max_emails : Optional[int]
        Maximum number of emails to load (None = all)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with email content and metadata
    """
    email_folder = Path(email_folder)
    email_files = []
    
    for ext in extensions:
        email_files.extend(email_folder.glob(f'*.{ext}'))
        email_files.extend(email_folder.glob(f'*.{ext.upper()}'))
    
    # Limit number of emails if specified
    if max_emails:
        email_files = email_files[:max_emails]
    
    email_info = []
    for email_path in email_files:
        try:
            with open(email_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Parse email content
            email_data = parse_emails(content)
            
            # Extract metadata and statistics
            headers = email_data['headers']
            body = email_data['body']
            
            info = {
                'filename': email_path.name,
                'path': str(email_path),
                'from': headers.get('from', ''),
                'to': headers.get('to', ''),
                'cc': headers.get('cc', ''),
                'subject': headers.get('subject', ''),
                'date': headers.get('date', ''),
                'content': body,
                'raw_content': content,
                'word_count': len(body.split()),
                'char_count': len(body),
                'line_count': body.count('\n') + 1,
                'size_kb': email_path.stat().st_size / 1024,
            }
            
            # Try to parse the date
            try:
                if 'date' in headers:
                    # Handle different date formats
                    date_str = headers['date']
                    for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S']:
                        try:
                            info['parsed_date'] = datetime.datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
            except Exception:
                pass
            
            email_info.append(info)
            
        except Exception as e:
            print(f"Error loading {email_path.name}: {e}")
    
    return pd.DataFrame(email_info)

def quick_corpus_summary(doc_df: pd.DataFrame) -> None:
    """
    Print summary statistics about text document dataset
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document content and metadata
    """
    print("="*50)
    print("TEXT CORPUS SUMMARY")
    print("="*50)
    print(f"Total documents: {len(doc_df)}")
    
    if 'extension' in doc_df.columns:
        print(f"Document formats: {doc_df['extension'].value_counts().to_dict()}")
    
    print(f"\nSize statistics:")
    print(f"  Word count: {doc_df['word_count'].min()} - {doc_df['word_count'].max()} (avg: {doc_df['word_count'].mean():.0f})")
    print(f"  Character count: {doc_df['char_count'].min()} - {doc_df['char_count'].max()} (avg: {doc_df['char_count'].mean():.0f})")
    print(f"  Line count: {doc_df['line_count'].min()} - {doc_df['line_count'].max()} (avg: {doc_df['line_count'].mean():.0f})")
    print(f"  File size: {doc_df['size_kb'].min():.1f} - {doc_df['size_kb'].max():.1f} KB (avg: {doc_df['size_kb'].mean():.1f} KB)")
    
    if 'created_date' in doc_df.columns:
        print(f"\nDate range:")
        print(f"  Created: {doc_df['created_date'].min()} - {doc_df['created_date'].max()}")
    
    if 'from' in doc_df.columns:
        # This is likely email data
        n_senders = doc_df['from'].nunique()
        top_senders = doc_df['from'].value_counts().head(5).to_dict()
        
        print(f"\nEmail statistics:")
        print(f"  Unique senders: {n_senders}")
        print(f"  Top senders: {top_senders}")


# ==================== TEXT PREPROCESSING ====================

def preprocess_text(text: str, 
                   lowercase: bool = True,
                   remove_punctuation: bool = True,
                   remove_numbers: bool = False,
                   remove_stopwords: bool = True,
                   stem_words: bool = False,
                   lemmatize: bool = False) -> List[str]:
    """
    Preprocess text by applying various transformations
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
    lowercase : bool
        Convert text to lowercase
    remove_punctuation : bool
        Remove punctuation characters
    remove_numbers : bool
        Remove numeric characters
    remove_stopwords : bool
        Remove common stopwords
    stem_words : bool
        Apply Porter stemming
    lemmatize : bool
        Apply WordNet lemmatization
        
    Returns:
    --------
    List[str]
        List of preprocessed tokens
    """
    if not text or not isinstance(text, str):
        return []
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Remove punctuation if requested
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers if requested
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
    
    # Apply stemming if requested
    if stem_words:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    
    # Apply lemmatization if requested
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens

def add_preprocessed_text(doc_df: pd.DataFrame, 
                         content_col: str = 'content',
                         **kwargs) -> pd.DataFrame:
    """
    Add preprocessed text columns to document DataFrame
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document content
    content_col : str
        Column containing raw text content
    **kwargs : dict
        Preprocessing parameters to pass to preprocess_text()
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added preprocessing columns
    """
    df = doc_df.copy()
    
    # Add preprocessed tokens column
    df['tokens'] = df[content_col].apply(lambda x: preprocess_text(x, **kwargs))
    
    # Add preprocessed text column
    df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    
    # Add token count
    df['token_count'] = df['tokens'].apply(len)
    
    # Add unique token count
    df['unique_token_count'] = df['tokens'].apply(lambda x: len(set(x)))
    
    return df

def enhanced_extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text with improved handling of company names and products
    
    Parameters:
    -----------
    text : str
        Text to analyze
        
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary with entity types and extracted entities
    """
    # Basic entity types
    entities = {
        'emails': [],
        'urls': [],
        'phone_numbers': [],
        'dates': [],
        'people': [],
        'organizations': [],
        'products': [],
        'roles_titles': []
    }
    
    # Extract email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities['emails'] = re.findall(email_pattern, text)
    
    # Extract URLs
    url_pattern = r'https?://[^\s]+'
    entities['urls'] = re.findall(url_pattern, text)
    
    # Extract phone numbers (various formats)
    phone_pattern = r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'
    entities['phone_numbers'] = re.findall(phone_pattern, text)
    
    # Extract dates (various formats)
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
    ]
    
    all_dates = []
    for pattern in date_patterns:
        all_dates.extend(re.findall(pattern, text, re.IGNORECASE))
    
    entities['dates'] = all_dates
    
    # Try to identify people by common name patterns
    
    # Look for "Dear [Name]" pattern common in emails
    dear_pattern = r'Dear\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
    dear_matches = re.findall(dear_pattern, text)
    if dear_matches:
        entities['people'].extend(dear_matches)
    
    # Look for common name prefixes
    name_pattern = r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Miss|Sir|Madam)\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
    name_matches = re.findall(name_pattern, text)
    if name_matches:
        entities['people'].extend(name_matches)
    
    # Look for signature at end (simple heuristic)
    lines = text.split('\n')
    for i in range(len(lines)-1, max(0, len(lines)-5), -1):  # Check last few lines
        if re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}$', lines[i].strip()):
            entities['people'].append(lines[i].strip())
    
    # Extract common company name patterns
    company_patterns = [
        r'\b([A-Z][a-z]*(?:\s[A-Z][a-z]*)*(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co|GmbH))\b',
        r'\b([A-Z][A-Za-z]*(?:\.com|\.io|\.ai|\.co))\b',  # Tech companies with CamelCase names
        r'\b([A-Z][a-z]+[A-Z][a-z]+)\b'  # CamelCase company names
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text)
        if matches:
            entities['organizations'].extend(matches)
    
    # Look specifically for known common product/company names
    known_companies = ['Google', 'Microsoft', 'Apple', 'Amazon', 'Facebook', 'Twitter', 
                      'LinkedIn', 'HubSpot', 'Salesforce', 'Zoom', 'Slack', 'Adobe']
    
    for company in known_companies:
        if company in text:
            entities['organizations'].append(company)
    
    # Extract titles and roles
    role_patterns = [
        r'\b(CEO|CTO|CFO|COO|CIO|VP|Director|Manager|Supervisor|President|Admin|Administrator|Engineer|Developer|Analyst)\b',
        r'\b([A-Z][a-z]+ (?:Manager|Director|Engineer|Administrator|Specialist|Analyst))\b'
    ]
    
    for pattern in role_patterns:
        matches = re.findall(pattern, text)
        if matches:
            entities['roles_titles'].extend(matches)
    
    # Remove duplicates in all entity lists
    for entity_type in entities:
        entities[entity_type] = list(set(entities[entity_type]))
    
    return entities

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text using simple regex patterns
    
    Parameters:
    -----------
    text : str
        Text to analyze
        
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary with entity types and extracted entities
    """
    entities = {
        'emails': [],
        'urls': [],
        'phone_numbers': [],
        'dates': []
    }
    
    # Extract email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities['emails'] = re.findall(email_pattern, text)
    
    # Extract URLs
    url_pattern = r'https?://[^\s]+'
    entities['urls'] = re.findall(url_pattern, text)
    
    # Extract phone numbers (various formats)
    phone_pattern = r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'
    entities['phone_numbers'] = re.findall(phone_pattern, text)
    
    # Extract dates (various formats)
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
    ]
    
    all_dates = []
    for pattern in date_patterns:
        all_dates.extend(re.findall(pattern, text, re.IGNORECASE))
    
    entities['dates'] = all_dates
    
    return entities

def extract_ngrams(tokens: List[str], n: int = 2, min_freq: int = 2) -> Dict[str, int]:
    """
    Extract n-grams with minimum frequency from tokens
    
    Parameters:
    -----------
    tokens : List[str]
        List of tokens
    n : int
        Size of n-grams to extract (2 = bigrams, 3 = trigrams)
    min_freq : int
        Minimum frequency to include n-gram in results
        
    Returns:
    --------
    Dict[str, int]
        Dictionary of n-grams and their frequencies
    """
    # Generate n-grams
    n_grams = list(ngrams(tokens, n))
    
    # Count frequencies
    gram_freq = Counter(n_grams)
    
    # Filter by minimum frequency and convert tuples to strings
    filtered_grams = {' '.join(gram): freq for gram, freq in gram_freq.items() if freq >= min_freq}
    
    # Sort by frequency (descending)
    return dict(sorted(filtered_grams.items(), key=lambda x: x[1], reverse=True))

def custom_tokenize(text: str, 
                   keep_punct: bool = False,
                   keep_sent_boundaries: bool = False,
                   split_contractions: bool = False) -> List[str]:
    """
    Advanced tokenization with additional options
    
    Parameters:
    -----------
    text : str
        Text to tokenize
    keep_punct : bool
        Whether to keep punctuation as separate tokens
    keep_sent_boundaries : bool
        Whether to insert sentence boundary markers
    split_contractions : bool
        Whether to split contractions (e.g. "don't" -> "do n't")
        
    Returns:
    --------
    List[str]
        List of tokens with specified options
    """
    if not text or not isinstance(text, str):
        return []
    
    # Handle sentence boundaries
    if keep_sent_boundaries:
        sentences = sent_tokenize(text)
        tokens = []
        
        for i, sentence in enumerate(sentences):
            # Add sentence start marker
            if i > 0:
                tokens.append("<s>")
                
            # Tokenize the sentence
            if keep_punct:
                # Tokenize including punctuation
                sent_tokens = []
                for word in word_tokenize(sentence):
                    if any(p in word for p in string.punctuation) and len(word) > 1:
                        # Handle punctuation attached to words
                        start = 0
                        for i, char in enumerate(word):
                            if char in string.punctuation:
                                if i > start:
                                    sent_tokens.append(word[start:i])
                                sent_tokens.append(char)
                                start = i + 1
                        if start < len(word):
                            sent_tokens.append(word[start:])
                    else:
                        sent_tokens.append(word)
                tokens.extend(sent_tokens)
            else:
                # Standard tokenization
                tokens.extend(word_tokenize(sentence))
            
            # Add sentence end marker
            tokens.append("</s>")
    else:
        # Standard tokenization without sentence boundaries
        if keep_punct:
            # Tokenize including punctuation
            tokens = []
            for word in word_tokenize(text):
                if any(p in word for p in string.punctuation) and len(word) > 1:
                    # Handle punctuation attached to words
                    start = 0
                    for i, char in enumerate(word):
                        if char in string.punctuation:
                            if i > start:
                                tokens.append(word[start:i])
                            tokens.append(char)
                            start = i + 1
                    if start < len(word):
                        tokens.append(word[start:])
                else:
                    tokens.append(word)
        else:
            # Standard tokenization
            tokens = word_tokenize(text)
    
    # Handle contractions
    if split_contractions:
        expanded_tokens = []
        for token in tokens:
            if "'" in token and token.lower() not in ["'s", "'ll", "'ve", "'d", "'m", "'re"]:
                # Split contraction
                parts = token.split("'")
                if len(parts) == 2:
                    expanded_tokens.append(parts[0])
                    expanded_tokens.append("'" + parts[1])
            else:
                expanded_tokens.append(token)
        tokens = expanded_tokens
    
    return tokens

def add_advanced_text_features(doc_df: pd.DataFrame, 
                              text_col: str = 'content') -> pd.DataFrame:
    """
    Add advanced linguistic features to document DataFrame
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document content
    text_col : str
        Column containing text content
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added linguistic features
    """
    df = doc_df.copy()
    
    # Function to calculate readability metrics
    def calculate_readability(text):
        if not text or not isinstance(text, str):
            return {
                'avg_word_length': 0,
                'avg_sent_length': 0,
                'lexical_diversity': 0
            }
        
        # Tokenize
        tokens = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        # Skip if empty
        if not tokens or not sentences:
            return {
                'avg_word_length': 0,
                'avg_sent_length': 0,
                'lexical_diversity': 0
            }
        
        # Calculate metrics
        avg_word_length = sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        avg_sent_length = len(tokens) / len(sentences) if sentences else 0
        lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0
        
        return {
            'avg_word_length': avg_word_length,
            'avg_sent_length': avg_sent_length,
            'lexical_diversity': lexical_diversity
        }
    
    # Apply to each document
    readability_features = df[text_col].apply(calculate_readability)
    
    # Extract and add individual metrics to DataFrame
    df['avg_word_length'] = readability_features.apply(lambda x: x['avg_word_length'])
    df['avg_sent_length'] = readability_features.apply(lambda x: x['avg_sent_length'])
    df['lexical_diversity'] = readability_features.apply(lambda x: x['lexical_diversity'])
    
    # Function to count parts of speech
    def count_parts_of_speech(text):
        if not text or not isinstance(text, str):
            return {
                'noun_count': 0,
                'verb_count': 0, 
                'adj_count': 0,
                'adv_count': 0
            }
        
        try:
            # Download required NLTK resource if not present
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Tokenize and tag parts of speech
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            
            # Count by POS category
            noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
            verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
            adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
            adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
            
            return {
                'noun_count': noun_count,
                'verb_count': verb_count,
                'adj_count': adj_count,
                'adv_count': adv_count
            }
        except Exception as e:
            print(f"Error in POS tagging: {e}")
            return {
                'noun_count': 0,
                'verb_count': 0,
                'adj_count': 0,
                'adv_count': 0
            }
    
    # Apply to each document (sample for speed if many documents)
    if len(df) > 100:
        print("Many documents detected. Calculating POS tags for a sample...")
        sample_indices = np.random.choice(df.index, min(100, len(df)), replace=False)
        pos_features = pd.Series(index=df.index)
        pos_features.loc[sample_indices] = df.loc[sample_indices, text_col].apply(count_parts_of_speech)
    else:
        pos_features = df[text_col].apply(count_parts_of_speech)
    
    # Extract and add individual metrics to DataFrame
    df['noun_count'] = pos_features.apply(lambda x: x['noun_count'] if x is not None else 0)
    df['verb_count'] = pos_features.apply(lambda x: x['verb_count'] if x is not None else 0)
    df['adj_count'] = pos_features.apply(lambda x: x['adj_count'] if x is not None else 0)
    df['adv_count'] = pos_features.apply(lambda x: x['adv_count'] if x is not None else 0)
    
    # Calculate ratios
    total_pos = df['noun_count'] + df['verb_count'] + df['adj_count'] + df['adv_count']
    df['noun_ratio'] = df['noun_count'] / total_pos.replace(0, 1)  # Avoid division by zero
    df['verb_ratio'] = df['verb_count'] / total_pos.replace(0, 1)
    df['adj_ratio'] = df['adj_count'] / total_pos.replace(0, 1)
    df['adv_ratio'] = df['adv_count'] / total_pos.replace(0, 1)
    
    # Extract entities
    entities = df[text_col].apply(extract_named_entities)
    
    # Add entity counts
    df['email_count'] = entities.apply(lambda x: len(x['emails']))
    df['url_count'] = entities.apply(lambda x: len(x['urls']))
    df['phone_count'] = entities.apply(lambda x: len(x['phone_numbers']))
    df['date_count'] = entities.apply(lambda x: len(x['dates']))
    
    return df

# ==================== TEXT VISUALIZATION ====================

def visualize_document_lengths(doc_df: pd.DataFrame, 
                             count_col: str = 'word_count',
                             group_col: Optional[str] = None,
                             bins: int = 20,
                             figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualize document length distribution
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document data
    count_col : str
        Column containing count to visualize
    group_col : Optional[str]
        Column to group by (e.g., category, author)
    bins : int
        Number of histogram bins
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if group_col and group_col in doc_df.columns:
        # Create grouped histogram
        groups = doc_df[group_col].unique()
        
        # Use a maximum of 5 groups for clarity
        if len(groups) > 5:
            # Get top 5 groups by frequency
            top_groups = doc_df[group_col].value_counts().nlargest(5).index
            for group in top_groups:
                subset = doc_df[doc_df[group_col] == group]
                plt.hist(subset[count_col], alpha=0.5, label=f'{group} (n={len(subset)})', bins=bins)
            plt.legend()
        else:
            for group in groups:
                subset = doc_df[doc_df[group_col] == group]
                plt.hist(subset[count_col], alpha=0.5, label=f'{group} (n={len(subset)})', bins=bins)
            plt.legend()
    else:
        # Create simple histogram
        plt.hist(doc_df[count_col], bins=bins)
    
    plt.xlabel(count_col.replace('_', ' ').title())
    plt.ylabel('Number of Documents')
    plt.title(f'Distribution of {count_col.replace("_", " ").title()}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_top_terms(doc_df: pd.DataFrame, 
                 n_terms: int = 20,
                 use_tokens: bool = True,
                 ngram_range: Tuple[int, int] = (1, 1),
                 figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot top terms across all documents
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document data
    n_terms : int
        Number of top terms to show
    use_tokens : bool
        Whether to use pre-tokenized text
    ngram_range : Tuple[int, int]
        Range of n-gram sizes
    figsize : Tuple[int, int]
        Figure size
    """
    if use_tokens and 'tokens' in doc_df.columns:
        # Flatten all token lists
        all_tokens = [token for token_list in doc_df['tokens'] for token in token_list]
        term_counts = Counter(all_tokens).most_common(n_terms)
        
        terms = [term for term, _ in term_counts]
        counts = [count for _, count in term_counts]
    else:
        # Use vectorizer for n-grams
        if 'processed_text' in doc_df.columns:
            text_column = 'processed_text'
        else:
            text_column = 'content'
        
        # Create and fit vectorizer
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=n_terms)
        term_counts = vectorizer.fit_transform(doc_df[text_column].fillna(''))
        
        # Get feature names and sum counts
        terms = vectorizer.get_feature_names_out()
        counts = term_counts.sum(axis=0).A1
        
        # Sort by frequency
        idx = counts.argsort()[::-1]
        terms = terms[idx]
        counts = counts[idx]
    
    # Create horizontal bar chart
    plt.figure(figsize=figsize)
    plt.barh(range(len(terms)), counts, align='center')
    plt.yticks(range(len(terms)), terms)
    plt.xlabel('Frequency')
    plt.ylabel('Terms')
    
    if ngram_range[1] > 1:
        plt.title(f'Top {n_terms} N-grams ({ngram_range[0]}-{ngram_range[1]})')
    else:
        plt.title(f'Top {n_terms} Terms')
        
    plt.gca().invert_yaxis()  # Display terms from top to bottom
    plt.tight_layout()
    plt.show()

def generate_wordcloud(doc_df: pd.DataFrame,
                     column: str = 'processed_text',
                     figsize: Tuple[int, int] = (12, 8),
                     max_words: int = 200,
                     background_color: str = 'white') -> None:
    """
    Generate and display word cloud for text data
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document data
    column : str
        Column containing text to visualize
    figsize : Tuple[int, int]
        Figure size
    max_words : int
        Maximum number of words to include
    background_color : str
        Background color for word cloud
    """
    # Combine all text
    if column == 'tokens':
        # Join tokens from all documents
        all_text = ' '.join([' '.join(tokens) for tokens in doc_df[column] if tokens])
    else:
        # Join text from all documents
        all_text = ' '.join(doc_df[column].fillna('').astype(str))
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        max_words=max_words, 
        background_color=background_color,
        collocations=False
    ).generate(all_text)
    
    # Display word cloud
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_term_frequency_by_group(doc_df: pd.DataFrame,
                               term_col: str = 'processed_text',
                               group_col: str = 'category',
                               n_terms: int = 10,
                               ngram_range: Tuple[int, int] = (1, 1),
                               figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot top terms for each group in a faceted plot
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document data
    term_col : str
        Column containing terms to analyze
    group_col : str
        Column to group by
    n_terms : int
        Number of top terms per group
    ngram_range : Tuple[int, int]
        Range of n-gram sizes
    figsize : Tuple[int, int]
        Figure size
    """
    if group_col not in doc_df.columns:
        print(f"Error: Group column '{group_col}' not found in DataFrame")
        return
    
    groups = doc_df[group_col].unique()
    n_groups = len(groups)
    
    if n_groups > 6:
        # Limit to top 6 groups by frequency
        top_groups = doc_df[group_col].value_counts().nlargest(6).index
        groups = top_groups
        n_groups = len(groups)
        print(f"Limiting visualization to top {n_groups} groups")
    
    # Calculate rows and columns for subplots
    n_cols = min(3, n_groups)
    n_rows = (n_groups - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'Top {n_terms} Terms by {group_col.title()}', fontsize=16)
    
    # Flatten axes for easy indexing
    if n_groups > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Create vectorizer for consistent feature extraction
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=100)
    
    # Fit vectorizer on all text
    vectorizer.fit(doc_df[term_col].fillna(''))
    
    for i, group in enumerate(groups):
        if i < len(axes):
            ax = axes[i]
            
            # Get documents for this group
            group_docs = doc_df[doc_df[group_col] == group]
            
            if len(group_docs) > 0:
                # Transform group documents
                group_counts = vectorizer.transform(group_docs[term_col].fillna(''))
                
                # Get feature names and sum counts
                terms = vectorizer.get_feature_names_out()
                counts = group_counts.sum(axis=0).A1
                
                # Sort by frequency and get top terms
                idx = counts.argsort()[::-1][:n_terms]
                top_terms = terms[idx]
                top_counts = counts[idx]
                
                # Horizontal bar chart
                ax.barh(range(len(top_terms)), top_counts, align='center')
                ax.set_yticks(range(len(top_terms)))
                ax.set_yticklabels(top_terms)
                ax.invert_yaxis()  # Display terms from top to bottom
                ax.set_title(f'{group} (n={len(group_docs)})')
                ax.set_xlabel('Frequency')
                
                if i % n_cols == 0:
                    ax.set_ylabel('Terms')
            else:
                ax.text(0.5, 0.5, f"No documents for {group}", 
                       ha='center', va='center')
                ax.axis('off')
    
    # Hide any unused subplots
    for i in range(n_groups, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_text_over_time(doc_df: pd.DataFrame,
                       date_col: str = 'created_date',
                       count_col: str = 'word_count',
                       group_col: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot text metrics over time
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document data
    date_col : str
        Column containing date information
    count_col : str
        Column containing count to visualize
    group_col : Optional[str]
        Column to group by
    figsize : Tuple[int, int]
        Figure size
    """
    if date_col not in doc_df.columns:
        print(f"Error: Date column '{date_col}' not found in DataFrame")
        return
    
    # Create copy with datetime index
    df = doc_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Set up plot
    plt.figure(figsize=figsize)
    
    if group_col and group_col in df.columns:
        # Calculate time-based stats by group
        groups = df[group_col].unique()
        
        # Limit to top 5 groups for clarity
        if len(groups) > 5:
            top_groups = df[group_col].value_counts().nlargest(5).index
            groups = top_groups
        
        for group in groups:
            group_df = df[df[group_col] == group]
            
            # Resample by month and calculate mean
            group_df = group_df.set_index(date_col)
            monthly = group_df[count_col].resample('M').mean()
            
            # Plot
            plt.plot(monthly.index, monthly.values, label=f'{group} (n={len(group_df)})')
    else:
        # Calculate overall time-based stats
        df = df.set_index(date_col)
        monthly = df[count_col].resample('M').mean()
        
        # Plot
        plt.plot(monthly.index, monthly.values, label=f'All Documents (n={len(df)})')
    
    plt.xlabel('Date')
    plt.ylabel(count_col.replace('_', ' ').title())
    plt.title(f'{count_col.replace("_", " ").title()} Over Time')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_sentiment_over_time(doc_df: pd.DataFrame,
                           date_col: str = 'created_date',
                           text_col: str = 'content',
                           window: str = 'M',
                           figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot sentiment analysis over time
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document data
    date_col : str
        Column containing date information
    text_col : str
        Column containing text to analyze
    window : str
        Resampling window ('D' for day, 'W' for week, 'M' for month)
    figsize : Tuple[int, int]
        Figure size
    """
    try:
        from textblob import TextBlob
    except ImportError:
        print("TextBlob is required for sentiment analysis. Install with pip install textblob")
        return
    
    if date_col not in doc_df.columns:
        print(f"Error: Date column '{date_col}' not found in DataFrame")
        return
    
    # Create copy with datetime index
    df = doc_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Calculate sentiment 
    df['sentiment'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0)
    df['subjectivity'] = df[text_col].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0)
    
    # Set up plot
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Set datetime index
    df = df.set_index(date_col)
    
    # Resample by specified window and calculate mean
    sentiment = df['sentiment'].resample(window).mean()
    subjectivity = df['subjectivity'].resample(window).mean()
    
    # Plot sentiment on first y-axis
    ax1.plot(sentiment.index, sentiment.values, 'b-', label='Sentiment')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment (Polarity)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Create second y-axis for subjectivity
    ax2 = ax1.twinx()
    ax2.plot(subjectivity.index, subjectivity.values, 'r-', label='Subjectivity')
    ax2.set_ylabel('Subjectivity', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('Sentiment and Subjectivity Over Time')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_interactive_embedding_plot(doc_df: pd.DataFrame,
                                    embeddings: np.ndarray,
                                    labels: Optional[List[str]] = None,
                                    title: str = 'Document Embeddings',
                                    hover_data: Optional[List[str]] = None) -> None:
    """
    Create interactive visualization of document embeddings
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document data
    embeddings : np.ndarray
        2D or 3D embedding coordinates
    labels : Optional[List[str]]
        Labels for coloring points (e.g., document categories)
    title : str
        Plot title
    hover_data : Optional[List[str]]
        Additional columns to include in hover information
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is required for interactive plots. Install with pip install plotly")
        return
    
    # Check embeddings dimensions
    if embeddings.shape[1] < 2 or embeddings.shape[1] > 3:
        print("Error: Embeddings must be 2D or 3D")
        return
    
    # Set up hover data
    if hover_data is None:
        if 'filename' in doc_df.columns:
            hover_data = ['filename']
        else:
            hover_data = []
    
    # Create DataFrame for plotting
    plot_df = doc_df.copy()
    
    # Add embedding coordinates
    plot_df['x'] = embeddings[:, 0]
    plot_df['y'] = embeddings[:, 1]
    if embeddings.shape[1] == 3:
        plot_df['z'] = embeddings[:, 2]
    
    # Create plot
    if embeddings.shape[1] == 2:
        # 2D scatter plot
        if labels is not None and labels in doc_df.columns:
            fig = px.scatter(
                plot_df, x='x', y='y', 
                color=labels,
                hover_data=hover_data,
                title=title
            )
        else:
            fig = px.scatter(
                plot_df, x='x', y='y',
                hover_data=hover_data,
                title=title
            )
    else:
        # 3D scatter plot
        if labels is not None and labels in doc_df.columns:
            fig = px.scatter_3d(
                plot_df, x='x', y='y', z='z',
                color=labels,
                hover_data=hover_data,
                title=title
            )
        else:
            fig = px.scatter_3d(
                plot_df, x='x', y='y', z='z',
                hover_data=hover_data,
                title=title
            )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(family="sans-serif", size=12),
        )
    )
    
    # Show plot
    fig.show()

# ==================== FEATURE EXTRACTION ====================

def extract_document_features(doc_df: pd.DataFrame, 
                            text_col: str = 'processed_text',
                            vectorizer_type: str = 'tfidf',
                            ngram_range: tuple = (1, 2),
                            max_features: int = 1000,
                            min_df: int = 2) -> tuple:
    """
    Extract features from document text using CountVectorizer or TfidfVectorizer
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document content
    text_col : str
        Column containing text to vectorize
    vectorizer_type : str
        Type of vectorizer ('count' or 'tfidf')
    ngram_range : tuple
        Range of n-gram sizes
    max_features : int
        Maximum number of features to extract
    min_df : int
        Minimum document frequency for terms
        
    Returns:
    --------
    tuple
        DataFrame with document features and fitted vectorizer
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    
    # Create vectorizer
    if vectorizer_type.lower() == 'count':
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df
        )
    else:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df
        )
    
    # Extract features
    X = vectorizer.fit_transform(doc_df[text_col].fillna(''))
    
    # Convert to DataFrame with feature names
    feature_names = vectorizer.get_feature_names_out()
    feature_df = pd.DataFrame(X.toarray(), columns=feature_names)
    
    # Add index from original DataFrame
    feature_df.index = doc_df.index
    
    return feature_df, vectorizer

def dimensionality_reduction(feature_matrix, 
                           method: str = 'svd',
                           n_components: int = 2) -> np.ndarray:
    """
    Reduce dimensionality of feature matrix
    
    Parameters:
    -----------
    feature_matrix : np.ndarray or scipy.sparse matrix
        Document feature matrix
    method : str
        Reduction method ('svd', 'pca', or 'lda')
    n_components : int
        Number of components to extract
        
    Returns:
    --------
    np.ndarray
        Reduced feature matrix
    """
    from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation
    
    if method.lower() == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    elif method.lower() == 'pca':
        # Convert sparse matrix to dense if necessary
        if hasattr(feature_matrix, 'toarray'):
            feature_matrix = feature_matrix.toarray()
        reducer = PCA(n_components=n_components, random_state=42)
    elif method.lower() == 'lda':
        reducer = LatentDirichletAllocation(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit and transform
    reduced_features = reducer.fit_transform(feature_matrix)
    
    return reduced_features

def train_document_classifier(X_train, y_train, X_test, y_test,
                            classifier_type: str = 'lr',
                            **kwargs) -> dict:
    """
    Train a document classifier
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    classifier_type : str
        Type of classifier ('lr', 'nb', 'rf', 'svm')
    **kwargs : dict
        Additional parameters for the classifier
        
    Returns:
    --------
    dict
        Dictionary with trained model and evaluation results
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # Create classifier
    if classifier_type.lower() == 'lr':
        classifier = LogisticRegression(max_iter=1000, **kwargs)
    elif classifier_type.lower() == 'nb':
        classifier = MultinomialNB(**kwargs)
    elif classifier_type.lower() == 'rf':
        classifier = RandomForestClassifier(**kwargs)
    elif classifier_type.lower() == 'svm':
        classifier = LinearSVC(max_iter=1000, **kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Train classifier
    classifier.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Return results
    return {
        'classifier': classifier,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

def cross_validate_classifier(doc_df, 
                            feature_df,
                            label_col,
                            classifier_type='lr',
                            n_folds=5,
                            random_state=42) -> dict:
    """
    Perform cross-validation for document classification
    
    Parameters:
    -----------
    doc_df : pd.DataFrame
        DataFrame with document data
    feature_df : pd.DataFrame
        DataFrame with document features
    label_col : str
        Column in doc_df containing labels
    classifier_type : str
        Type of classifier ('lr', 'nb', 'rf', 'svm')
    n_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    import numpy as np
    
    # Create classifier
    if classifier_type.lower() == 'lr':
        classifier = LogisticRegression(max_iter=1000, random_state=random_state)
    elif classifier_type.lower() == 'nb':
        classifier = MultinomialNB()
    elif classifier_type.lower() == 'rf':
        classifier = RandomForestClassifier(random_state=random_state)
    elif classifier_type.lower() == 'svm':
        classifier = LinearSVC(max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Get features and labels
    X = feature_df.values
    y = doc_df[label_col].values
    
    # Create cross-validation folds
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Perform cross-validation
    cv_scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
    
    # Return results
    results = {
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'min_accuracy': cv_scores.min(),
        'max_accuracy': cv_scores.max(),
        'all_scores': cv_scores,
        'classifier_type': classifier_type,
        'n_folds': n_folds
    }
    
    return results

def explain_model_features(model, feature_names, class_names=None, n_features=10) -> dict:
    """
    Explain the most important features for a classification model
    
    Parameters:
    -----------
    model : estimator
        Trained classification model
    feature_names : list
        List of feature names
    class_names : list, optional
        List of class names
    n_features : int
        Number of top features to show per class
        
    Returns:
    --------
    dict
        Dictionary with important features by class
    """
    # Initialize results dictionary
    results = {}
    
    # Different classifiers have different ways to get feature importance
    if hasattr(model, 'coef_'):
        # Linear models (LR, SVM)
        coefficients = model.coef_
        
        # For binary classification with only one coefficient array
        if len(coefficients.shape) == 1 or coefficients.shape[0] == 1:
            coefficients = coefficients.reshape(1, -1)
            if class_names is None:
                class_names = ['Negative', 'Positive']
            elif len(class_names) == 2:
                # For binary classification, we have coefficients for positive class
                class_names = [class_names[1]]
        elif class_names is None:
            class_names = [f'Class {i}' for i in range(coefficients.shape[0])]
        
        # Get top features for each class
        for i, class_name in enumerate(class_names):
            if i < len(coefficients):
                # Get coefficients for this class
                class_coef = coefficients[i]
                
                # Get top positive and negative features
                top_pos_idx = np.argsort(class_coef)[-n_features:][::-1]
                top_neg_idx = np.argsort(class_coef)[:n_features]
                
                top_pos_features = [(feature_names[j], class_coef[j]) for j in top_pos_idx]
                top_neg_features = [(feature_names[j], class_coef[j]) for j in top_neg_idx]
                
                results[class_name] = {
                    'positive': top_pos_features,
                    'negative': top_neg_features
                }
    
    elif hasattr(model, 'feature_importances_'):
        # Tree-based models (RF)
        importances = model.feature_importances_
        
        # Get top important features overall
        top_idx = np.argsort(importances)[-n_features:][::-1]
        top_features = [(feature_names[i], importances[i]) for i in top_idx]
        
        results['overall'] = top_features
    
    return results

def visualize_classification_results(model_results, class_names=None, figsize=(14, 10)) -> None:
    """
    Visualize document classification results
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model and evaluation results from train_document_classifier
    class_names : list, optional
        List of class names
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Get confusion matrix and report
    cm = model_results['confusion_matrix']
    report = model_results['classification_report']
    
    # Use provided class names or extract from report
    if class_names is None:
        class_names = list(report.keys())
        class_names = [c for c in class_names if c not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot confusion matrix
    ax = axes[0]
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Plot classification metrics
    ax = axes[1]
    
    # Extract metrics
    metrics = ['precision', 'recall', 'f1-score']
    metrics_data = []
    
    for cls in class_names:
        for metric in metrics:
            metrics_data.append({
                'class': cls,
                'metric': metric,
                'value': report[cls][metric]
            })
    
    # Create DataFrame and pivot
    import pandas as pd
    metrics_df = pd.DataFrame(metrics_data)
    pivot_df = metrics_df.pivot(index='class', columns='metric', values='value')
    
    # Plot metrics as grouped bar chart
    pivot_df.plot(kind='bar', ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title('Classification Metrics by Class')
    ax.set_ylabel('Score')
    ax.legend(title='Metric')
    ax.grid(axis='y', alpha=0.3)
    
    # Display overall accuracy
    plt.figtext(
        0.5, 0.01, 
        f"Overall Accuracy: {model_results['accuracy']:.4f}", 
        ha='center', 
        bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 5}
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

