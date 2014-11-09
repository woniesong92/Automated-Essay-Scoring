import parser
from tfidf_vectorizer import Tfidf_Vectorizer
import re
import pdb
import numpy as np



# Given that we already have "essays.pkl"
# extract tfidf, word_count, number of different words, words average length
# More ideas: complexity of the parse trees of the essay?
class FeatureExtractor:
    def extract_essay_and_scores(self, train_examples):
        essays= []
        scores = []
        for train_example in train_examples:
            essay = self.remove_symbols(train_example["essay"])
            essays.append(essay)
            scores.append(train_example["domain1_score"])
        return (essays, scores)

    # List of features for each essay
    # 1. tfidf
    # 2. word count
    # 3. number of different words
    # 4. words' average length 
    # TODO: parse trees' complexity?
    def transform_training_examples(self, train_examples):
        essays, scores = self.extract_essay_and_scores(train_examples)
        # 1. tfidf 
        tfidf_vectorizer = Tfidf_Vectorizer()
        (tfidf, tfs) = tfidf_vectorizer.get_tfidf_vectors(essays)
        matrix = np.array(tfs.todense())

        # 2. word count
        word_counts_features = [len(essay.split()) for essay in essays]
        matrix = self.update_matrix(matrix, word_counts_features)

        # 3. number of different words
        num_diff_words_features = [self.num_diff_words(essay) for essay in essays]
        matrix = self.update_matrix(matrix, num_diff_words_features)

        # 4. average length of words
        words_avg_length_features = [self.words_avg_length(essay) for essay in essays]
        matrix = self.update_matrix(matrix, words_avg_length_features)



        # Add the scores 
        matrix = self.update_matrix(matrix, scores)
        pdb.set_trace()
        return matrix



    ####### HELPER METHODS ######
    def update_matrix(self, matrix, new_features):
        return np.insert(matrix, 0, new_features, 1)

    def remove_symbols(self, essay):
        return re.sub(r'[^\w]', ' ', essay)

    def words_avg_length(self, essay):
        words = essay.split(" ")
        lengths = [len(w) for w in words]
        avg_word_length = sum(lengths) / len(lengths)
        return avg_word_length

    def num_diff_words(self, essay):
        unique_words = set()
        words = essay.split(" ")
        for word in words:
            if word not in unique_words:
                unique_words.add(word)
        return len(unique_words)




myParser = parser.Parser()
training_examples = myParser.parse("data/training_set_rel3.tsv")
myFeatureExtractor = FeatureExtractor()
features = myFeatureExtractor.transform_training_examples(training_examples[:5])


