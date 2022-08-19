# creating an lda class with customized attributes
import pickle
from gensim.models import LdaMulticore
from gensim import corpora

class LDA(LdaMulticore):
    def __init__(self, train_data,n_topics,passes=2, workers=4):
        self.train_data = train_data
        self.n_topics = n_topics
        self.passes = passes
        self.workers = workers
        self.dictionary = corpora.Dictionary(self.train_data)
        self.dictionary.filter_extremes(no_below=30, no_above=0.5, keep_n=100000)
        self.bow_corpus = [self.dictionary.doc2bow(text) for text in self.train_data]
        self.lda_bow_model = self.create_bow_lda_model()


    def create_bow_lda_model(self):
        return LdaMulticore(self.bow_corpus,num_topics=self.n_topics,
                        id2word=self.dictionary, passes=self.passes,workers=self.workers)


    #save the trained object into a file
    def pickle(self,file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
