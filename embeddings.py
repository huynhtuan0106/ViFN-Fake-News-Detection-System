import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from config import TEXT_CONFIG, DEVICE
import warnings
warnings.filterwarnings('ignore')

class TFIDFEmbedder:
    """TF-IDF embeddings for text fields"""
    
    def __init__(self, max_features=3000):
        self.vectorizers = {}
        self.max_features = max_features
    
    def fit_transform(self, texts, field_name):
        """Fit and transform texts to TF-IDF vectors"""
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=TEXT_CONFIG['ngram_range'],
            min_df=TEXT_CONFIG['min_df'],
            max_df=TEXT_CONFIG['max_df']
        )
        embeddings = vectorizer.fit_transform(texts).toarray()
        self.vectorizers[field_name] = vectorizer
        return embeddings
    
    def transform(self, texts, field_name):
        """Transform texts using fitted vectorizer"""
        if field_name not in self.vectorizers:
            raise ValueError(f"Vectorizer for {field_name} not fitted")
        return self.vectorizers[field_name].transform(texts).toarray()

class BERTEmbedder:
    """BERT-based embeddings for Vietnamese text"""
    
    def __init__(self, model_name='google-bert/bert-base-uncased', max_length=256):
        print("Loading BERT model for Vietnamese...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()
        self.max_length = max_length
        print(f"BERT model loaded on {DEVICE}")
    
    def encode_texts(self, texts, batch_size=16):
        """Encode texts to BERT embeddings"""
        embeddings = []
        
        print(f"Encoding {len(texts)} texts with BERT...")
        for i in range(0, len(texts), batch_size):
            if i % (batch_size * 10) == 0:
                print(f"Progress: {i}/{len(texts)}")
                
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(DEVICE)
            attention_mask = encoded['attention_mask'].to(DEVICE)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Use CLS token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        print("BERT encoding completed!")
        return np.array(embeddings)
    
class PhoBERTEmbedder:
    """BERT-based embeddings for Vietnamese text"""
    
    def __init__(self, model_name='vinai/phobert-base', max_length=256):
        print("Loading BERT model for Vietnamese...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()
        self.max_length = max_length
        print(f"BERT model loaded on {DEVICE}")
    
    def encode_texts(self, texts, batch_size=16):
        """Encode texts to BERT embeddings"""
        embeddings = []
        
        print(f"Encoding {len(texts)} texts with BERT...")
        for i in range(0, len(texts), batch_size):
            if i % (batch_size * 10) == 0:
                print(f"Progress: {i}/{len(texts)}")
                
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(DEVICE)
            attention_mask = encoded['attention_mask'].to(DEVICE)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Use CLS token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        print("BERT encoding completed!")
        return np.array(embeddings)


def simulate_bert_embeddings(texts, dim=768):
    """Simulate BERT embeddings for demonstration purposes"""
    print(f"Creating simulated BERT embeddings ({dim}D) for {len(texts)} texts...")
    np.random.seed(42)
    return np.random.normal(0, 1, (len(texts), dim))

def create_all_embeddings(df, use_real_bert=False):
    """
    Create embeddings for all text fields from processed dataframe
    
    Args:
        df: DataFrame from data_loader with processed text columns
        use_real_bert: Whether to use real BERT or simulated embeddings
    
    Returns:
        embeddings: Dictionary with TF-IDF and BERT embeddings
        tfidf_embedder: Fitted TF-IDF embedder for future use
    """
    
    print("\n" + "="*60)
    print("CREATING TEXT EMBEDDINGS")
    print("="*60)
    
    # Check for processed text columns
    required_columns = ['title_processed', 'summary_processed', 'content_processed']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing processed text columns: {missing_columns}. "
                        f"Please run data preprocessing first.")
    
    # TF-IDF embeddings
    print("Creating TF-IDF embeddings...")
    tfidf_embedder = TFIDFEmbedder(max_features=TEXT_CONFIG['max_features_tfidf'])
    
    title_tfidf = tfidf_embedder.fit_transform(df['title_processed'], 'title')
    summary_tfidf = tfidf_embedder.fit_transform(df['summary_processed'], 'summary')
    content_tfidf = tfidf_embedder.fit_transform(df['content_processed'], 'content')
    
    print(f"TF-IDF embeddings created:")
    print(f"  Title: {title_tfidf.shape}")
    print(f"  Summary: {summary_tfidf.shape}")
    print(f"  Content: {content_tfidf.shape}")
    
    # BERT embeddings
    print("\nCreating BERT embeddings...")
    if use_real_bert:
        try:
            bert_embedder = BERTEmbedder()
            title_bert = bert_embedder.encode_texts(df['title_processed'].tolist())
            summary_bert = bert_embedder.encode_texts(df['summary_processed'].tolist())
            content_bert = bert_embedder.encode_texts(df['content_processed'].tolist())
            print("Real BERT embeddings created successfully!")
        except Exception as e:
            print(f"Error creating BERT embeddings: {e}")
            print("Falling back to simulated BERT embeddings...")
            title_bert = simulate_bert_embeddings(df['title_processed'])
            summary_bert = simulate_bert_embeddings(df['summary_processed'])
            content_bert = simulate_bert_embeddings(df['content_processed'])
    else:
        # Simulated BERT embeddings for demonstration
        title_bert = simulate_bert_embeddings(df['title_processed'])
        summary_bert = simulate_bert_embeddings(df['summary_processed'])
        content_bert = simulate_bert_embeddings(df['content_processed'])
    
    print(f"BERT embeddings created:")
    print(f"  Title: {title_bert.shape}")
    print(f"  Summary: {summary_bert.shape}")
    print(f"  Content: {content_bert.shape}")
    
    embeddings = {
        'tfidf': {
            'title': title_tfidf,
            'summary': summary_tfidf,
            'content': content_tfidf
        },
        'bert': {
            'title': title_bert,
            'summary': summary_bert,
            'content': content_bert
        }
    }
    
    print("\nAll embeddings created successfully!")
    return embeddings, tfidf_embedder

if __name__ == "__main__":
    print("Testing embeddings creation...")
    print("Please run this through main_real_data.py with actual data.")
    print("This module requires processed dataframe from data_loader.py")