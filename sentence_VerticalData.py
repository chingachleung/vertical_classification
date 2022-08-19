import torch
from torch.utils.data import Dataset
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
class VerticalData(Dataset):
    def __init__(self, dataframe, targets, tokenizer, max_len):
        self.tokenizer = tokenizer
        #self.data = dataframe
        self.text = dataframe.sources
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        sentences = sent_tokenize((text))
        curr_ids = []
        curr_attention_masks= []
        curr_token_type_ids = []
        for sent in sentences:
            input = self.tokenizer.encode_plus(
                sent,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                #pad to the max length
                padding='max_length',
                truncation=True,
                return_token_type_ids=True
            )
            curr_ids.append(input['input_ids'])
            curr_attention_masks.append(input['attention_mask'])
            curr_token_type_ids.append(input['token_type_ids'])
        ids = curr_ids
        mask = curr_attention_masks
        token_type_ids = curr_token_type_ids


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
        }
