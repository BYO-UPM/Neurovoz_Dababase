import os
import librosa
import numpy as np
import torch
import timm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

# Assuming `data/audio` is the path where your audio files are stored
audio_path = "data/audios"
audio_files = [
    os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith(".wav")
]


def extract_label(filename):
    # Extracts condition (HC or PD) from filename
    # Returns 0 for HC and 1 for PD
    return 0 if filename.split("/")[-1].split("_")[0] == "HC" else 1


def extract_id(filename):
    # Extracts the patient ID from filename
    return filename.split("/")[-1].split("_")[2]


def audio_to_melspectrograms(audio_file):
    y, sr = librosa.load(audio_file)
    # Frame the signal into short windows of 400ms
    # and hop by 200ms
    if len(y) > int(sr * 0.4):
        y_framed = librosa.util.frame(
            y, frame_length=int(sr * 0.4), hop_length=int(sr * 0.2)
        )
    else:
        # If the signal is too short, pad it with zeros
        y_padded = np.concatenate([y, np.zeros(int(sr * 0.4 - len(y)))])
        y_framed = librosa.util.frame(
            y_padded, frame_length=int(sr * 0.4), hop_length=int(sr * 0.2)
        )
    # Compute the power of the signal in each frame
    mels = librosa.feature.melspectrogram(
        y=y_framed.T, sr=sr, n_fft=512, hop_length=int(sr * 0.03), n_mels=65
    )
    mels_db = librosa.power_to_db(mels, ref=np.max)
    # Normalize the melspectrogram
    mels_db = (mels_db - mels_db.mean()) / mels_db.std()
    return mels_db


print("Reading audio files and extracting labels...")
# Process all audio files and extract labels
data = []
labels = []
ids = []
for file in audio_files:
    label = extract_label(file)
    spectrograms = audio_to_melspectrograms(file)
    # Repeat the label for each spectrogram
    label_exploded = [label] * spectrograms.shape[0]
    ids_exploded = [extract_id(file)] * spectrograms.shape[0]
    data.append(spectrograms)
    labels.append(label_exploded)
    ids.append(ids_exploded)


# Concatenate the lists
data = np.vstack(data)
labels = np.concatenate(labels)
ids = np.concatenate(ids)

# Split data into train and test sets using ID, no id should be in both sets
unique_ids = np.unique(ids)
ids_train, ids_test = train_test_split(unique_ids, test_size=0.2, random_state=42)
X_train = data[np.isin(ids, ids_train)]
y_train = labels[np.isin(ids, ids_train)]
X_test = data[np.isin(ids, ids_test)]
y_test = labels[np.isin(ids, ids_test)]


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
model = timm.create_model("resnet18", pretrained=True, num_classes=2)

# Move to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Adapt the model for 1-channel input if necessary
model.conv1 = torch.nn.Conv2d(
    1,
    model.conv1.out_channels,
    kernel_size=model.conv1.kernel_size,
    stride=model.conv1.stride,
    padding=model.conv1.padding,
    bias=False,
)

model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training the model...")
# Training loop (simplified for demonstration)
for epoch in range(10):  # loop over the dataset multiple times
    # use tqdm
    for data in tqdm(train_loader):
        inputs, labels = data
        inputs = inputs.to(torch.float32).to(device)
        labels = labels.to(torch.long).to(device)
        optimizer.zero_grad()
        # convert to double
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Prediction on the test dataset (simplified)
print("Making predictions...")
predictions = []
model.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(torch.float32).to(device)
        labels = labels.to(torch.long).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().detach().numpy())

# Evaluate the predictions
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)

print(accuracy_score(y_test, predictions))
print(balanced_accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
