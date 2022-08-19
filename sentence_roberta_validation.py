"""validating a fine-tuned roberta model
"""
import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer
from RobertaClass import RobertaClass
from torch.utils.data import DataLoader
from sentence_VerticalData import VerticalData
import torch.nn as nn
from statistics import mode
#from nltk.tokenize import sent_tokenize

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

args = argparse.ArgumentParser(description='validating the sentence based Roberta model')
args.add_argument('-a', '--testing_file', type=str, help='testing_file', required=True)
args.add_argument('-m', '--model_file', type=str, help='saved model', required=True)
args = args.parse_args()


model_file = args.model_file
testing_file = args.testing_file

LEARNING_RATE = 1e-05
MAX_LEN = 50
BATCH_SIZE = 1

model = RobertaClass()
#model = nn.DataParallel(model, device_ids=[0,1,2,3])
model = nn.DataParallel(model, device_ids=[0])
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
onehot_encoder= checkpoint['encoder']

loss_function = torch.nn.CrossEntropyLoss()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True)



def make_confusion_matrix(labels, predictions):
    fig, ax2 = plt.subplots(figsize=(14, 12))
    # why it is showing just number labels in the test set?
    label_names = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label_names)
    disp.plot(ax=ax2)
    plt.savefig('sentence_roberta_challenge_cm.png')
    plt.show()



def valid(model, testing_file):
    model.eval()
    predictions = np.array([])
    labels = np.array([])
    #onehot_encoder = OneHotEncoder(handle_unknown="ignore")
    test_data = pd.read_csv(testing_file, encoding="utf-8-sig")
    test_data = test_data[['sources','categories']]
    #for testing
    #test_data = test_data[:5]
    #print(test_data['categories'])
    #print(test_data['sources'][3])
    #sent_list = sent_tokenize(test_data['sources'][1])
    #print('sent length', len(sent_list))


    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': False, # should not shuffle
                   'num_workers': 0
                   }


    testing_onehot_targets = onehot_encoder.transform(test_data['categories'].values.reshape(-1, 1)).toarray()
    testing_set = VerticalData(test_data, testing_onehot_targets, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set, **test_params)
    with torch.no_grad():
        print('validation started')
        for _, data in tqdm(enumerate(testing_loader, 0)):
            print(_)
            id_list = data['ids'].to(device, dtype=torch.long)[0]
            mask_list = data['mask'].to(device, dtype=torch.long)[0]
            token_type_id_list =data['token_type_ids'].to(device, dtype=torch.long)[0]
            targets = data['targets'].to(device, dtype=torch.long)
            #target_big_val, target_big_idx = torch.max(targets, dim=1)
            #numpy cannoy handle gpu
            targets = targets.cpu()
            inversed_targets = onehot_encoder.inverse_transform(targets)
            labels = np.append(labels, inversed_targets)
            print('id_list', id_list)
            outputs = model(id_list, mask_list,token_type_id_list)
            inversed_predictions = onehot_encoder.inverse_transform(outputs.data.tolist())
            inversed_predictions = list(np.concatenate(inversed_predictions).flat)
            doc_prediction = mode(inversed_predictions)
            predictions = np.append(predictions, doc_prediction)

    scores = classification_report(labels, predictions)
    print(scores)
    make_confusion_matrix(labels, predictions)

valid(model, testing_file)
