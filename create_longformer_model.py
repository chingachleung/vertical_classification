# Importing the libraries needed
import pandas as pd
import torch.nn as nn
from model_train import *
from transformers import LongformerTokenizer
from torch.utils.data import DataLoader
from VerticalData import VerticalData
from LongformerClass import LongformerClass
import argparse
import logging
logging.basicConfig(level=logging.ERROR)

from sklearn.preprocessing import OneHotEncoder

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

model = LongformerClass()
model = nn.DataParallel(model, device_ids=[0,1,2,3])
model.to(device)

args = argparse.ArgumentParser(description='create a longformer model ')
args.add_argument('-a', '--train_file', type=str, help='train file', required=True)
args = args.parse_args()

train_file = args.train_file



#global variables
# Defining some key variables that will be used later on in the training
MAX_LEN = 1000
BATCH_SIZE = 15
EPOCHS = 2
LEARNING_RATE = 1e-05

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", truncation=True)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

#load the data
train_data = pd.read_csv(train_file, encoding='utf-8-sig')
train_data = train_data[['sources','categories']]
#categories = train_data['categories']
#cancel out for real training
#train_data = train_data[:5]
#label_names = ['corporatecomm', 'elearning', 'banking', 'lspfa', 'legal','patents', 'retail', 'medicaldevice', 'tourism']
train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }




#data preparation
onehot_encoder = OneHotEncoder(handle_unknown="ignore")  # set to zeros if new categories in test set occur

#training_onehot_targets = onehot_encoder.fit_transform(train_data['categories'].values.reshape(-1, 1)).toarray()
#onehot_encoder = onehot_encoder.fit(train_data['categories'].values.reshape(-1, 1))
training_onehot_targets = onehot_encoder.fit_transform(train_data['categories'].values.reshape(-1, 1)).toarray()
training_set = VerticalData(train_data, training_onehot_targets, tokenizer, MAX_LEN)
training_loader = DataLoader(training_set, **train_params)

#start training
model, loss, optimizer = train(model, optimizer, EPOCHS, training_loader)

torch.save({'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'encoder': onehot_encoder},'longformer_model.pt')

print('All files saved')
