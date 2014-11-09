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
            columns = lines[0]
            data = lines[1:]
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