from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import pickle
import pdb
import nltk


class Tfidf_Vectorizer:
    def __init__(self):
        self.stemmer = PorterStemmer()


    def stem_tokens(self, tokens, stemmer):
            stemmed = []
            for item in tokens:
                stemmed.append(stemmer.stem(item))
            return stemmed

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        stems = self.stem_tokens(tokens, self.stemmer)
        return stems

    def get_tfidf_vectors(self, essays):
        tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words='english')#, decode_error="ignore") 
        tfs = tfidf.fit_transform(essays)
        return (tfidf, tfs)

    def get_normalized_tfidf_vectors(self, essays):
        tfidf = TfidfVectorizer(tokenizer=self.tokenize, norm='l1' ,stop_words='english')#, decode_error="ignore") 
        tfs = tfidf.fit_transform(essays)
        return (tfidf, tfs)

