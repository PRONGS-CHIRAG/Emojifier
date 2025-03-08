# Emojifier

## Overview
**Emojifier** is a deep learning-based Natural Language Processing (NLP) model designed to add relevant emoji representations to text input. It utilizes word embeddings and Long Short-Term Memory (LSTM) networks to classify sentences into predefined emoji categories.

## Features
- Implements **GloVe word embeddings** for sentence representation.
- Uses **LSTM layers** for sequence modeling.
- Supports **emoji classification** based on textual input.
- Leverages **pre-trained word embeddings** to improve model accuracy.
- Employs **softmax activation** for classification.

## Project Structure
```
Emojifier/
│── Emoji_v3a.ipynb      # Jupyter Notebook containing model training and evaluation
│── utils.py             # Utility functions for processing and modeling
│── emo_utils.py         # Additional helper functions 
│── test_utils.py        # Test functions for validation 
```

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install numpy tensorflow emoji matplotlib
```

## Model Architecture
The Emojifier model follows this architecture:
1. **Embedding Layer** - Pre-trained GloVe embeddings for word representation.
2. **LSTM Layer (128 units)** - Captures contextual dependencies in text.
3. **Dropout Layer (0.5)** - Reduces overfitting.
4. **Second LSTM Layer (128 units)** - Further learns sentence patterns.
5. **Dropout Layer (0.5)** - Ensures better generalization.
6. **Dense Layer (5 units)** - Outputs class scores.
7. **Softmax Activation** - Converts scores to probabilities.

## Training
The model is trained using:
- **Cross-entropy loss** for classification.
- **Stochastic Gradient Descent (SGD)** optimizer.
- **One-hot encoding** for labels.
- **400 iterations** for optimization.

## Usage
To run the model, follow these steps:
1. Load the dataset and word embeddings.
2. Train the model using the `model()` function.
3. Test the model using `predict()`.
4. Use `Emojify_V2()` to classify new sentences.

## Future Enhancements
- Extend the emoji classification to a larger set.
- Implement attention mechanisms for better context understanding.
- Deploy as a web API for integration into messaging applications.

## License
This project is open-source and available under the MIT License.


