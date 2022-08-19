from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pickle
from utils import *

args = argparse.ArgumentParser(description='test SVM model on test data')
args.add_argument('-t', '--test_file', type=str, help='test file', required=True)
args.add_argument('-m', '--model_file', type=str, help='model file', required=True)
args = args.parse_args()


TEST_FILE = args.test_file
MODEL_FILE = args.model_file
TOPIC_NUM = 9

def svm_validate(model, test_data, test_labels):
    fig, ax2 = plt.subplots(figsize=(15, 12))
    preds = model.predict(test_data)
    score = classification_report(test_labels, preds, target_names=model.classes_)
    print(score)
    cm = confusion_matrix(test_labels, preds, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax2)
    plt.show()
    plt.savefig('svm_challenge_cm.png')

def main():
    #load the svm model

    model = pickle.load(open(MODEL_FILE,'rb'))
    test = pd.read_csv(TEST_FILE, encoding='utf-8-sig')
    # for testing:
    #test = test[:500]

    test_labels = test['categories']
    test_docs = test['sources']

    processed_docs = preprocess_texts(test_docs)
    with open("preprocessed_test_docs", 'wb') as newf0:
        pickle.dump(processed_docs, newf0)

    print('finished processing test docs')

    # use the trained lda and doc2vec models
    with open('lda_model', 'rb') as f1:
        loaded_lda = pickle.load(f1)
    with open('doc2vec_model', 'rb') as f2:
        doc2vec_model = pickle.load(f2)

    # get the test feats
    lda_bow_model = loaded_lda.lda_bow_model
    test_bow_corpus = [loaded_lda.dictionary.doc2bow(text) for text in processed_docs]
    test_bow_topic_dist = list(lda_bow_model[test_bow_corpus])
    test_lda_bow_features = covert_to_sparse_array(test_bow_topic_dist, TOPIC_NUM)

    test_doc2vec_vectors = [doc2vec_model.infer_vector(x) for x in processed_docs]

    # combine the features
    test_combined_feats = np.array(list(map(lambda x, y: list(x) + list(y), test_lda_bow_features, test_doc2vec_vectors)))

    #validation

    svm_validate(model, test_combined_feats,test_labels)

main()
