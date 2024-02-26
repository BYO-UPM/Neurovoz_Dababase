# NeuroVoz: a Castillian Spanish dataset of parkinosnian speech
A complete corpus comprising data from 108 Spanish Castilian speakers. This corpus encompasses speech recordings from both control speakers (55) and parkinsonian speakers (53). All speakers with Parkinson's disease were receiving pharmacological treatment at the time of recording, and they took their medication from 2h to 4h prior to the speech collection session (i.e., the patients were in the ON state). This corpus includes tasks such as sustained phonations of the vowels, diadochokinetic evaluation, 16 various listen-and-repeat utterances and a monologue. This dataset is the first public dataset in Castilian Spanish of parkinsonian speech.

The dataset comprises 2,903 audio files, with an average of $26.88 \pm 3.35$ audio recordings per user. It provides valuable resources to the scientific community to systematically investigate the objective parkinson's effect on speech patterns.


## Audio Classification and Analysis Repository

This repository contains a collection of scripts for audio classification using deep learning and traditional machine learning methods, as well as utilities for audio data preprocessing and analysis. The goal of this project is to demonstrate different approaches to classify audio samples into predefined categories, such as distinguishing between different conditions based on audio features.

## Overview

The repository is structured to include scripts for audio classification using ResNet and traditional machine learning algorithms like RandomForest (RF) and Logistic Regression (LR). Additionally, it provides utility scripts for calculating audio statistics, cleaning datasets, and analyzing general statistics of the datasets.

### Classification Scripts

1. **simple_predictor_audio.py**: Utilizes a pre-trained ResNet model to classify audio samples based on their mel frequency spectrogram representations. This approach leverages the power of deep learning to understand complex patterns in audio data.

2. **simple_predictor_af.py**: Applies RandomForest and Logistic Regression algorithms on audio quality features extracted from the samples. This method showcases traditional machine learning techniques for audio classification.

### Utility Scripts

1. **audio_statistics.py**: Calculates and outputs general statistics of the audio files, such as mean, median, and standard deviation of audio lengths, to understand the dataset better.

2. **clean_dataset.py**: Preprocesses the audio dataset by removing noise and other unwanted artifacts, providing a cleaner dataset for more accurate classification.

3. **general_statistics.py**: Generates general statistics of the dataset, including distributions of classes and features, to give insights into the dataset's composition and potential biases.

## Getting Started

The requirements of the project are listed in the `requirements.txt` file. To install the necessary packages, run the following command:

```bash
conda create --name <env> --file requirements.txt
```

## Usage

To use the classification scripts, run the following command:

```bash
python simple_predictor_audio.py
```

```bash
python simple_predictor_af.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you encounter any problems.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
