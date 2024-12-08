# 🩺 Metastatic Cancer Detection

This repository implements an end-to-end deep learning pipeline to detect metastatic cancer in pathology image patches. The project uses a Convolutional Neural Network (CNN) to classify image patches as either containing tumor tissue (label 1) or not (label 0). The data comes from high-resolution pathology images with binary labels for classification.
📁 Project Structure

    data/: Dataset with labeled image patches for training and testing.
    notebooks/: Jupyter notebooks for exploratory data analysis (EDA), preprocessing, and training.
    src/: Python scripts for data augmentation, model definition, and training pipeline.
    results/: Training logs, visualizations, and evaluation metrics.

# 🧠 Problem Statement

Detect metastatic cancer in small image patches of size 96x96 pixels, with the central 32x32 pixel region containing the tissue of interest. The main challenges include class imbalance, effective feature extraction, and generalization to unseen data.

# ⚙️ Key Features
Exploratory Data Analysis (EDA):

    Class distribution analysis shows a slight imbalance (~40.5% tumor-positive samples).
    Verified the absence of null and duplicate values.

Data Preprocessing:

    Applied data augmentation using horizontal/vertical flips and rescaling pixel values to [0, 1].
    Implemented train-validation split (80-20).

Model Architecture (CNN):

    Convolutional Layers: Three convolutional layers with increasing filters (32 → 64 → 128) for hierarchical feature extraction.
    Pooling Layers: MaxPooling for spatial dimension reduction.
    Dense Layers: Fully connected layers with ReLU activation.
    Output Layer: Single neuron with sigmoid activation for binary classification.


# 🔍 Results
Training Performance:

    Training Accuracy: Up to 96.88%.
    Validation Accuracy: Plateaued at 86.74%, indicating some overfitting.

Loss Trends:

    Training loss consistently decreased.
    Validation loss fluctuated, suggesting potential regularization or hyperparameter tuning opportunities.

# 🛠️ Tools & Libraries

    Python: Core programming language for implementation.
    Keras/TensorFlow: Deep learning framework for CNN.
    Pandas, NumPy: Data manipulation and preprocessing.
    Matplotlib, Seaborn: Visualization tools.

# 🚀 Future Improvements

    Implement advanced augmentation techniques (e.g., rotation, brightness/contrast adjustment).
    Explore transfer learning with pre-trained models like ResNet or DenseNet.
    Optimize hyperparameters using grid or random search.
    Address overfitting with dropout layers or L2 regularization.


