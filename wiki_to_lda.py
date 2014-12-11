import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# save wikipedia dump 
id2word = gensim.corpora.Dictionary.load_from_text('data/wiki_en_wordids.txt')
mm = gensim.corpora.MmCorpus('data/wiki_en_tfidf.mm')

lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)
lda.save('data/wiki_en_lda.model')