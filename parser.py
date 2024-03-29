import csv
import pdb
import re
import pickle


class Parser:
    def parse(self, tsv_file):
        with open(tsv_file) as tsv:
            lines = []
            for line in csv.reader(tsv, delimiter="\t"):
                lines.append(line)
            # columns = lines[0]

            columns = ['essay_id', 'essay_set', 'essay', 
                       'rater1_domain1',  'rater2_domain1',  'rater3_domain1',
                       'domain1_score',   'rater1_domain2',  'rater2_domain2',  'domain2_score',
                       'rater1_trait1',   'rater1_trait2',   'rater1_trait3',   'rater1_trait4',   'rater1_trait5',   'rater1_trait6',   
                       'rater2_trait1',   'rater2_trait2',   'rater2_trait3',   'rater2_trait4',   'rater2_trait5',   'rater2_trait6',   
                       'rater3_trait1',   'rater3_trait2',   'rater3_trait3',   'rater3_trait4',   'rater3_trait5',   'rater3_trait6']

            data = lines
            # each essay will be represented as a dictionary
            # example: train_example["essay_id"] will return the essay_id of the train example
            train_examples = []
            for datum in data:
                train_example = {}
                i = 0  
                for column in columns:
                    train_example[column] = datum[i]
                    i += 1
                train_examples.append(train_example)
            return train_examples