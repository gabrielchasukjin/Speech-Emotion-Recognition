# Speech Emotion Recognition

This project aims to classify human emotions from speech using machine learning models. Developed as part of the Halıcıoğlu Data Science Institute Undergraduate Research Scholarship, this system processes audio clips, extracts meaningful features, and applies advanced machine learning techniques to achieve accurate emotion classification.

![Framework](SER.pdf)

## Overview
The SER system processes speech data to classify emotions such as happiness, sadness, anger, and neutrality. By leveraging state-of-the-art machine learning techniques and feature engineering, the project achieved **86% accuracy** on the test dataset.

## Features
- Aggregates and processes 12,000 audio clips.
- Data augmentation techniques (noise injection, time-stretching) to simulate real-world variability.
- Feature extraction through spectrogram analysis for visual representation of audio signals.
- Supports multiple machine learning models, including:
  - **Convolutional Neural Networks (CNN)**
  - **Support Vector Machines (SVM)**
  - **Vision Transformers (ViT)**
  - **Decision Trees**

## Technologies Used
- **Languages**: Python
- **Libraries**: 
  - `librosa` for audio processing
  - `matplotlib` for spectrogram generation
  - `scikit-learn`, `TensorFlow` for machine learning
- **Development Tools**: Jupyter Notebook, Git

## Project Workflow
1. **Data Collection**:
   - Aggregated audio clips from Kaggle datasets: (1) RAVDESS Emotional speech audio, (2) Toronto emotional speech set (TESS), (3) CREMA-D, and (4) Surrey Audio-Visual Expressed Emotion (SAVEE)
2. **Data Preprocessing**:
   - Applied augmentations such as noise injection and time-stretching to enhance dataset diversity.
   - Extracted features like spectrograms for better representation of audio signals.
3. **Model Training**:
   - Optimized and trained four models (CNN, SVM, ViT, Decision Trees) for classification.
4. **Evaluation**:
   - Tested models on unseen data, achieving **86% accuracy** with the CNN model.

## Results
- **Best Model**: CNN
- **Accuracy**: 86%
- **Challenges**: 
  - Overconfidence in misclassifications (e.g., anger classified as sadness).
  - Motivated further exploration into interpretability and fairness in machine learning models.


Authors: Gabriel Cha, JunHee Hwang, Arman Rahman, Mentor Justin Eldridge