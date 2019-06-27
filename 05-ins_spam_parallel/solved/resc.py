from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.base import TransformerMixin
from joblib import Parallel, delayed
from scipy.sparse import vstack
import numpy as np

class ParallelHashingVectorizer(HashingVectorizer):
    
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                 binary=False, norm='l2', alternate_sign=True,
                 dtype=np.float64, n_jobs=1):
        
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.alternate_sign = alternate_sign
        self.dtype = dtype
        self.n_jobs = n_jobs
    
    def transform(self, X, y=None, **fit_params):
        
        delayed_hashing_vectorizer = delayed(HashingVectorizer().transform)
        
        X_parts = np.array_split(X, self.n_jobs)
        
        X_parts_transformed = Parallel(n_jobs=self.n_jobs)(delayed_hashing_vectorizer(X_part) for X_part in X_parts)
        
        X_transformed = vstack(X_parts_transformed)
        
        return X_transformed
    
