import os
import librosa
import numpy as np
import torch
import timm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

# Assuming `data/audio` is the path where your audio files are stored
audio_path = 'data/audios'
audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.wav')]

def extract_label(filename):
    # Extracts condition (HC or PD) from filename
    # Returns 0 for HC and 1 for PD
    return 0 if filename.split('_')[0] == 'HC' else 1

def audio_to_melspectrogram(audio_file):
    y, sr = librosa.load(audio_file)
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    return mels_db

print("Reading audio files and extracting labels...")
# Process all audio files and extract labels
data = []
labels = []
for file in audio_files:
    label = extract_label(file)
    spectrogram = audio_to_melspectrogram(file)
    data.append(spectrogram)
    labels.append(label)

# Split data into train and test sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define a custom dataset
class AudioDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spectrogram = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            spectrogram = self.transform(spectrogram)
        return spectrogram, label

# Convert data to PyTorch tensors and create dataloaders
transform = ToTensor()
train_dataset = AudioDataset(X_train, y_train, transform=transform)
test_dataset = AudioDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load a pre-trained model from timm
print("Loading pre-trained model...")
model = timm.create_model('resnet18', pretrained=True, num_classes=2)

# Adapt the model for 1-channel input if necessary
model.conv1 = torch.nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=False)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training the model...")
# Training loop (simplified for demonstration)
for epoch in range(3):  # loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.tensor(labels, dtype=torch.long))
        loss.backward()
        optimizer.step()

# Prediction on the test dataset (simplified)
print("Making predictions...")
predictions = []
model.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.numpy())

# Evaluate the predictions
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

print(accuracy_score(y_test, predictions))
print(balanced_accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


