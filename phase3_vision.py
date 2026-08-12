import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
 
print("Starting Phase 3b: Training Computer Vision Neural Network (CNN)...\n")
 
# 1. Load the MNIST Dataset
# Same as before, but we also normalize using MNIST's known mean/std,
# which tends to help CNNs converge a bit faster/more stably.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
 
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
 
print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} testing images.")
 
# 2. Build the CNN Architecture
# Key idea: Conv2d layers slide small filters over the image to detect
# local patterns (edges, curves, loops) BEFORE we flatten anything.
# This preserves spatial relationships that a plain MLP throws away immediately.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv block 1: 1 input channel (grayscale) -> 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Conv block 2: 32 -> 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves the spatial size each time
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
 
        # After two 2x2 pools: 28 -> 14 -> 7, so we have 64 channels of 7x7 maps
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)                # flatten only at the end
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
 
model = CNN()
 
# 3. Define the Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 4. Train the Model!
epochs = 5
print("\n--- Training the CNN ---")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
 
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
 
# 5. Evaluate the Model on Unseen Data
print("\n--- Testing the Model ---")
correct = 0
total = 0
model.eval()
 
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
accuracy = 100 * correct / total
print(f"\nComputer Vision Accuracy: {accuracy:.2f}%")
 
# 6. Live Demonstration
image_index = np.random.randint(0, len(test_dataset))
unseen_image, true_label = test_dataset[image_index]
 
output = model(unseen_image.unsqueeze(0))
_, predicted_tensor = torch.max(output.data, 1)
predicted_digit = predicted_tensor.item()
 
print("\n--- Live Image Scan ---")
print(f"The actual handwritten digit is: {true_label}")
print(f"The Neural Network predicts:     {predicted_digit}")
 
# Note: unseen_image has been normalized, so unnormalize it for a clean display
img_display = unseen_image.squeeze() * 0.3081 + 0.1307
plt.imshow(img_display, cmap='gray')
plt.title(f"Network Guessed: {predicted_digit} | Actual: {true_label}")
plt.axis('off')
plt.show()
 
print("\nPhase 3b Complete!")
 
