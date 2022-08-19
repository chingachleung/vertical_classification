"""
creating a Longformer class
"""

import torch
from transformers import LongformerModel

class LongformerClass(torch.nn.Module):
    def __init__(self):
        super(LongformerClass, self).__init__()
        self.l1 = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 9)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0] # returning sequence outputs, if it's -1, it's all the hidden states
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
