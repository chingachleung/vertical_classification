from sklearn.svm import SVC

import pandas as pd
import pickle
import argparse

from utils import *
from LDA import LDA

args = argparse.ArgumentParser(description='Train a SVM model with LDA and Doc2vec feats')
args.add_argument('-a', '--train_file', type=str, help='train file', required=True)
args = args.parse_args()


TRAIN_FILE = args.train_file


def svm_model(X_train, y_train):
    model = SVC()  # default is RBF, degree is 3
    model.fit(X_train, y_train)
    return model


def main():

    train = pd.read_csv(TRAIN_FILE, encoding='utf-8-sig')
    # for testing:
    #train = train[:10]

    train_labels = train['categories']
    train_docs = train['sources']

    processed_docs = preprocess_texts(train_docs)

    train_lda = LDA(processed_docs, n_topics=9)
    train_lda.pickle('lda_model')
     
    documents = [TaggedDocument(doc,[i]) for i, doc in enumerate(processed_docs)]
    doc2vec_model = Doc2Vec(documents, vector_size=40, window=5,min_count=3, workers=2)
    with open("doc2vec_model", 'wb') as newf:
        pickle.dump(doc2vec_model, newf)


    # get LDA features
    train_bow_corpus = train_lda.bow_corpus
    lda_bow_model = train_lda.lda_bow_model
    bow_topic_dist = list(lda_bow_model[train_bow_corpus])
    topic_num = train_lda.n_topics
    lda_bow_features = covert_to_sparse_array(bow_topic_dist, topic_num)


    # get doc2vec features
    train_doc2vec_vectors = [doc2vec_model.infer_vector(x) for x in processed_docs]
    # combine the features
    train_combined_feats = np.array(list(map(lambda x, y: list(x) + list(y), lda_bow_features, train_doc2vec_vectors)))

    model = svm_model(train_combined_feats, train_labels)
    filename='svm_model.sav'
    pickle.dump(model, open(filename,'wb'))

main()
