# Parses wordcount, number of different words used (lexical richness) and
# average length of words from given essay data
class FeatureParser:
  def __init__(self, filename):
    self.filename = filename
    self.wordcount_dict = {} #{essay_id: wordcount}
    self.num_diff_words_dict = {} #{essay_id: num_diff_words}
    self.words_avg_length_dict = {} #{essay_id: avg_word_length}

  def parse(self):
    def wordcount(essay, essay_id, wordcount_dict):
      word_count = len(essay.split(" "))
      wordcount_dict[essay_id] = word_count

    def num_diff_words(essay, essay_id, num_diff_words_dict):
      unique_words = set()
      words = essay.split(" ")
      for word in words:
        if word not in unique_words:
          unique_words.add(word)
      num_diff_words_dict[essay_id] = len(unique_words)

    def words_avg_length(essay, essay_id, words_avg_length_dict):
      words = essay.split(" ")
      lengths = [len(w) for w in words]
      avg_word_length = sum(lengths) / len(lengths)
      words_avg_length_dict[essay_id] = avg_word_length

    with open(self.filename, 'r') as f:
      f.next() #skip the first line
      for line in f:
        [essay_id,essay_set,essay,rater1_domain1,rater2_domain1,rater3_domain1,
        domain1_score,rater1_domain2,rater2_domain2,domain2_score,rater1_trait1,
        rater1_trait2,rater1_trait3,rater1_trait4,rater1_trait5,rater1_trait6,
        rater2_trait1,rater2_trait2,rater2_trait3,rater2_trait4,rater2_trait5,
        rater2_trait6,rater3_trait1,rater3_trait2,rater3_trait3,rater3_trait4,
        rater3_trait5,rater3_trait6] = line.split('\t')

        # get wordcount
        wordcount(essay, essay_id, self.wordcount_dict)

        # get number of different words
        num_diff_words(essay, essay_id, self.num_diff_words_dict)

        # get average length of words
        words_avg_length(essay, essay_id, self.words_avg_length_dict)

      return {"wordcount": self.wordcount_dict,
              "num_diff_words": self.num_diff_words_dict,
              "words_avg_length": self.words_avg_length_dict}

  def write_to_file(self):
    def write_wordcounts(f):
      f.write("wordcount\n")
      f.write("essay_id\twordcount\n")
      for essay_id in self.wordcount_dict:
        line = essay_id + "\t" + str(self.wordcount_dict[essay_id]) + "\n"
        f.write(line)

    def write_num_diff_words(f):
      f.write("num_diff_words\n")
      f.write("essay_id\tnum_diff_words\n")
      for essay_id in self.num_diff_words_dict:
        line = essay_id + "\t" + str(self.num_diff_words_dict[essay_id]) + "\n"
        f.write(line)

    def write_words_avg_length(f):
      f.write("words_avg_length\n")
      f.write("essay_id\twords_avg_length\n")
      for essay_id in self.words_avg_length_dict:
        line = essay_id + "\t" + str(self.words_avg_length_dict[essay_id]) + "\n"
        f.write(line)

    filename = self.filename + ".results"
    with open(filename, 'wb') as f:
      write_wordcounts(f)
      write_num_diff_words(f)
      write_words_avg_length(f)
    return 0
