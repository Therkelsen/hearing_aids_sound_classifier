# Hearing Aid Sound Scene Classifier

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

An accurate, computationally efficient convolutional neural network (CNN) designed for hearing aid sound scene classification. Developed as a collaborative university project in partnership with Oticon.

---

## Abstract & Overview

Modern hearing aids use sophisticated algorithms to enhance speech comprehension and suppress background noise, but performance in dynamic acoustic environments can still be optimized. This project designs a compact sound scene classifier capable of categorizing audio snippets into five distinct classes while respecting the strict memory and computational limitations of hearing aid hardware (under $500,000$ parameters).

The model evaluates spectrogram data ($96$ time frames $\times$ $32$ frequency bands) and classifies audio into:
1. **Other**
2. **Music**
3. **Human Voice**
4. **Engine Sounds**
5. **Alarm**

---

## Project Documentation & Report

The full technical report detailing the design choices, architecture rationale, hyperparameter tuning, and performance analysis is available in the repository:
* **Report:** [`hearing_aids_sound_classifier report.docx`](hearing_aids_sound_classifier%20report.docx)

---

## Neural Network Architecture

The model is structured as a sequential convolutional neural network focused on depth rather than width:

* **Input Layer:** Shape $(96, 32, 1)$ representing 2D spectrogram representations of audio snippets.
* **Feature Extraction:** 
  - `Conv2D` (filters dynamically tuned, kernel $5 \times 3$ / $3 \times 3$) + `LeakyReLU` activation + `BatchNormalization` + `MaxPooling2D` + `Dropout`
  - Secondary and tertiary `Conv2D` blocks with max pooling and dropout regularizations.
* **Classification Head:** 
  - `Flatten`
  - `Dense` layer with `LeakyReLU` and batch normalization.
  - Output `Dense` layer with `softmax` activation ($5$ classes).

**Optimizers & Loss:** Compiled using categorical crossentropy loss and the Adam optimizer.

---

## Results & Performance

* **Training Accuracy:** $\sim 92\%$ with minimal overfitting on training/validation splits.
* **Test Accuracy:** $\sim 97\%$ on a held-out test set ($80/10/10$ data split).
* **Robustness & Challenges:** The model demonstrates robust performance across most categories, with slight classification overlap between alarms and human voices due to overlapping frequency ranges.

---

## Repository File Structure

* `hearing_aids_sound_classifier.ipynb`: The complete Google Colab Jupyter Notebook containing data loading, preprocessing, Keras-Tuner hyperparameter search, model training, evaluation, and confusion matrix visualizations.
* `317k_LReLU_97-percent.h5`: The serialized pre-trained TensorFlow/Keras model achieving $97\%$ test accuracy ($317,365$ parameters).
* `predictions.txt`: Unlabeled test set classification output generated for Oticon's evaluation pipeline.