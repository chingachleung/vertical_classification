
"""
helper functions to preprocess texts
"""
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")
import csv
import sys
maxInt = sys.maxsize

MAX_WORD = 50

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def limit_char(text):
    #spacy char limit = 1000,000
    return text[:1000000]

def preprocess_texts(text_data):
    train_texts = text_data.map(limit_char)
    processed_docs = []
    for doc in nlp.pipe(train_texts, disable=['tok2vec', 'parser', 'ner','textcat']):
        toks = []
        count = 0
        for tok in doc:
            if count < MAX_WORD:
                if tok.is_stop == False:
                    if tok.is_digit:
                        toks.append("NUM")
                        count += 1
                    elif tok.is_alpha:
                        toks.append(tok.lemma_.lower())
                        count += 1
            else:
                break
        #new_doc = " ".join(toks)
        processed_docs.append(toks)
    return processed_docs

def covert_to_sparse_array(predictions,topic_num):
    """
    helper function to convert LDA predictions into features
    """
    topic_arr = np.zeros([len(predictions), topic_num])
    for i, result in enumerate(predictions):
        for topic, score in result:
            topic_arr[i,topic] = score
        return topic_arr
