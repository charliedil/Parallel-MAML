#this code is code Noor and I developed in our Emerging Trends in NLP Class, I'll be parallelizing it such that each task is learned on a different process
import torch
import torch.nn as nn
from transformers import AutoModel
class NERModel(nn.Module):
    def __init__(self):
        super(NERModel, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased") #bet
        #for param in self.bert.parameters():
        #    param.requires_grad = False  # let's try to unfreeze bert
        self.dropout = nn.Dropout(0.1)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True) #bilstn
        self.linear_layer = nn.Linear(200, 13)  # 100 * 2 (bidirectional) linear

    def forward(self, input_ids, attention_mask):
        #print(input_ids.device)
        #print(attention_mask.device)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Pass the output through the BiLSTM layer
        lstm_output, _ = self.bilstm(sequence_output)

        # Apply the linear layer to get NER label logits
        logits = self.linear_layer(lstm_output)

        return logits
