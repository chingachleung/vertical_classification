"""validating a fine-tuned roberta model
"""
import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import LongformerTokenizer
from LongformerClass import LongformerClass
from torch.utils.data import DataLoader
from VerticalData import VerticalData
import torch.nn as nn

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

args = argparse.ArgumentParser(description='validating the lonformer model')
args.add_argument('-a', '--testing_file', type=str, help='testing_file', required=True)
args.add_argument('-m', '--model_file', type=str, help='saved model', required=True)
args = args.parse_args()


model_file = args.model_file
testing_file = args.testing_file

LEARNING_RATE = 1e-05
MAX_LEN = 1000
BATCH_SIZE = 1

model = LongformerClass()
model = nn.DataParallel(model, device_ids=[0,1,2,3])
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
onehot_encoder= checkpoint['encoder']
#label_names = ['corporatecomm', 'elearning', 'banking', 'lspfa', 'legal','patents', 'retail', 'medicaldevice', 'tourism']

loss_function = torch.nn.CrossEntropyLoss()
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", truncation=True)

def make_confusion_matrix(labels, predictions):
    fig, ax2 = plt.subplots(figsize=(14,12))
    label_names = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label_names)
    disp.plot(ax=ax2)
    plt.savefig('longformer_cm.png')
    plt.show()

def valid(model, testing_file):
    model.eval()
    predictions = np.array([])
    labels = np.array([])
    #onehot_encoder = OneHotEncoder(handle_unknown="ignore")
    test_data = pd.read_csv(testing_file, encoding="utf-8-sig")
    test_data = test_data[['sources','categories']]
    #for testing
    #test_data = test_data[:500]

    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }
    tr_loss = 0
    nb_tr_steps = 0

    testing_onehot_targets = onehot_encoder.transform(test_data['categories'].values.reshape(-1, 1)).toarray()
    testing_set = VerticalData(test_data, testing_onehot_targets, tokenizer, MAX_LEN)
    testing_loader = DataLoader(testing_set, **test_params)
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            target_big_val, target_big_idx = torch.max(targets, dim=1)
            targets = targets.cpu()
            inversed_targets = onehot_encoder.inverse_transform(targets)
            labels = np.append(labels, inversed_targets)
            outputs = model(ids, mask, token_type_ids)
            inversed_predictions = onehot_encoder.inverse_transform(outputs.data.tolist())
            predictions = np.append(predictions, inversed_predictions)
            loss = loss_function(outputs, target_big_idx)
            tr_loss += loss.item()

            nb_tr_steps += 1
    epoch_loss = tr_loss / nb_tr_steps
    print(f"Validation Loss: {epoch_loss}")
    print('labels', labels)
    print('predictions', predictions)
    scores = classification_report(labels, predictions)
    print(scores)
    make_confusion_matrix(labels, predictions)

valid(model, testing_file)
