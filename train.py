import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import DistilBertTokenizer
from sklearn.metrics import accuracy_score
from data_loader import get_imdb_dataloaders
from model import DistilBERT_QNN_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_imdb_dataloaders(batch_size=8)
model = DistilBERT_QNN_Model().to(device)
optimizer = Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_preds = []
    total_labels = []

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_preds.extend(preds.detach().cpu().numpy())
        total_labels.extend(labels.detach().cpu().numpy())

    acc = accuracy_score(total_labels, total_preds)
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Train Accuracy: {acc:.4f}")

torch.save(model.state_dict(), "quantum_sentiment_model.pt")