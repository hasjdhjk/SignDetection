#
#
# LSTM model is defined here. Can be tweaked and altered. 
#
#


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import LandmarkDataset

# Load dataset
dataset = LandmarkDataset("landmark_data/")
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define LSTM model
class SignLSTM(nn.Module):
    def __init__(self, input_size=129, hidden_size=64, num_layers=2, num_classes=3):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLSTM(num_classes=len(dataset.label_map)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")
