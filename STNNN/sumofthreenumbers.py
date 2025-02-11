import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, dataset
import numpy as np
import math
import matplotlib.pyplot as plt

# a, b, c, d [a + b + c = d]
epochs = 2
lr = .001
n_samples = 2500000
range_data = range(0,100)
batch_size = 64

Data = [[(sample := random.choices(range_data, k=3)), sum(sample)] for _ in range(n_samples)]

Train, Test = train_test_split(Data, test_size=.2)

train_loader = DataLoader(Train, batch_size=batch_size, shuffle=True)
test_load = DataLoader(Test, batch_size=batch_size, shuffle=False)



class STNNN(nn.Module):
  def __init__(self):
    super(STNNN, self).__init__()
    self.fc1 = nn.Linear(3,32)
    #Shape: (1, 32, 1) → (batch_size, channels, sequence_length)
    #Output_sequence_Size=⌊(input_sequence_Size+2×Padding−Dilation×(Kernel_Size−1)−1)/Stride⌋+1
    self.conv1 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
    self.fc2 = nn.Linear(32,1)
  def forward(self, input):
    x = F.relu(self.fc1(input))
    x = x.unsqueeze(2)
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.squeeze(2)
    x = self.fc2(x)
    return x

data = torch.rand(4,3)
res = 6
model = STNNN()
results = model(data)

Loss_F = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
def train():
  for i in range(epochs):
    batch_loss = []
    for ib, data in enumerate(train_loader):
      # Stack the first three tensors along the second dimension
      stacked_data = torch.stack([data[0][0], data[0][1], data[0][2]], dim=1)  # Shape: (8, 3)

      # Combine with the last tensor
      reshaped_data = torch.cat([stacked_data, data[1].unsqueeze(1)], dim=1).float()  # Shape: (8, 4)

      # Display the reshaped data
      input_data = reshaped_data[:,:3]
      y_results = reshaped_data[:,3].unsqueeze(1)
      y_hat = model(input_data)
      loss = Loss_F(y_hat, y_results)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      batch_loss.append(loss.item())

        # Display real-time loss updates
      print(f"\r[!] Epoch [{i+1}/{epochs}] Batch [{ib+1}/{len(train_loader)}] "
              f"Loss: {loss.item():.4f} | Avg Loss: {sum(batch_loss)/len(batch_loss):.4f}", end="")

train()

def test():
  model.eval()
  fresults = []
  for idx, data in enumerate(test_load):
      stacked_data = torch.stack([data[0][0], data[0][1], data[0][2]], dim=1)
      reshaped_data = torch.cat([stacked_data, data[1].unsqueeze(1)], dim=1).float()
      # Display the reshaped data
      input_data = reshaped_data[:,:3]
      y_results = reshaped_data[:,3].unsqueeze(1)
      y_hat = model(input_data)
      for res in zip(y_hat, y_results):
        yp = res[0].item()
        yr = res[1].item()
        if yp - math.floor(yp) < 0.5:
          yp = int(yp)
        else:
          yp = int(math.ceil(yp))
        fresults.append([yp, int(yr)])
  return fresults
test_results = test()

P = [res for res in test_results if res[0]==res[1]]
N = [res for res in test_results if res[0]!=res[1]]

import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def calculate_accuracy(results):
    correct = sum([1 for pred, true in results if pred == true])
    total = len(results)
    accuracy = (correct / total) * 100
    return accuracy
def plot_results(results):
    predictions = [res[0] for res in results]
    actuals = [res[1] for res in results]

    # Scatter Plot: Predictions vs Actual
    plt.figure(figsize=(10, 5))
    plt.scatter(actuals, predictions, alpha=0.7)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    plt.grid()
    plt.show()

    # Line Plot: Showing trends over index
    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual', marker='o')
    plt.plot(predictions, label='Predicted', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Prediction Trends')
    plt.legend()
    plt.grid()
    plt.show()

    # Optional: Confusion Matrix for Classification Tasks
    if len(set(actuals)) < 20:  # Only plot if reasonable number of classes
        cm = confusion_matrix(actuals, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_perfect_accuracy(results):
    predictions = [res[0] for res in results]
    actuals = [res[1] for res in results]

    # Scatter Plot with Perfect Diagonal
    plt.figure(figsize=(10, 6))
    # Residual Plot (Should be zero for all points)
    residuals = np.array(predictions) - np.array(actuals)
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(residuals)), residuals, color='purple', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Residuals (Prediction - Actual)')
    plt.title('Residual Plot (Zero Residuals = Perfect Accuracy)', fontsize=14)
    plt.grid(True)
    plt.show()

    # Bar Plot of Actual vs Predicted
    indices = np.arange(len(actuals))
    bar_width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(indices, actuals, bar_width, label='Actual', alpha=0.7)
    plt.bar(indices + bar_width, predictions, bar_width, label='Predicted', alpha=0.7, color='orange')

    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Actual vs Predicted Bar Plot', fontsize=14)
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    # Heatmap of Confusion Matrix (Diagonal Dominance)
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', linewidths=0.5, square=True, cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix (100% Accuracy)', fontsize=14)
    plt.show()

accuracy = calculate_accuracy(test_results)
print(f"Model Accuracy: {accuracy:.2f}%")

# Plot Results
plot_results(test_results)
plot_perfect_accuracy(test_results)

data = torch.tensor([[11,22,33]], dtype=torch.float32)
print(model(data))

def save_model(model, optimizer, epoch, loss, path='pretrained/model_checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"Model saved to {path}")
save_model(model=model, optimizer=optimizer, epoch=epochs, loss=0.04406)

def load_model(model, optimizer, path='pretrained/model_checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {path}, resuming from Epoch {epoch+1} with Loss {loss:.4f}")
    return epoch, loss
load_model(model=model, optimizer=optimizer)

