import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchquantum.nn import QuantumLayer  # 假設使用 torchquantum 的 QNN 套件

class DistilBERT_QNN_Model(nn.Module):
    def __init__(self):
        super(DistilBERT_QNN_Model, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.qnn = QuantumLayer(n_qubits=4, n_layers=1)  # 量子層設計
        self.fc = nn.Linear(4, 2)  # 二分類

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        quantum_output = self.qnn(pooled_output[:, :4])
        return self.fc(quantum_output)