import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import LandmarkDataset

#---------------------------------------------#
# Load dataset and extract label map          #
#---------------------------------------------#
dataset = LandmarkDataset("landmark_data/")
label_map = dataset.label_map  # Access from dataset
num_classes = len(label_map)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#---------------------------------------------#
# Define LSTM model                           #
#                                             #
#---------------------------------------------#
class SignLSTM(nn.Module):
    def __init__(self, input_size=135, hidden_size=64, num_layers=2, num_classes=3):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

#---------------------------------------------#
# Initialize                                  #
#                                             #
#---------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLSTM(input_size=135, hidden_size=64, num_layers=2, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#---------------------------------------------#
# Training loop                               #
#                                             #
#---------------------------------------------#
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

#---------------------------------------------#
# Optional: Save model and label map          # 
#                                             #
#---------------------------------------------#
torch.save(model.state_dict(), "sign_lstm.pth")
print("✅ Model saved to sign_lstm.pth")
