Cross-Domain Pneumonia Classification Using Deep Learning and DenseNet121
Research Project | Medical Image Classification | Transfer Learning

Researcher: Niharika Rawat
Abstract
This project focuses on developing a deep learning model to detect pneumonia from chest X-ray images using DenseNet121 and transfer learning. The dataset contains 5,856 chest X-ray images classified into two categories: Normal and Pneumonia. The project includes data cleaning, exploratory data analysis (EDA), image preprocessing, data augmentation, model development, training, and performance evaluation.
Objectives
- Analyse and clean the chest X-ray dataset.
- Perform exploratory data analysis to understand the distribution of images.
- Preprocess images into a standard size and normalise pixel values.
- Apply data augmentation to improve model generalisation.
- Build a pneumonia classification model using DenseNet121.
- Evaluate model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
Dataset
The dataset was obtained from Kaggle's Chest X-Ray Images (Pneumonia) collection.
- Total images: 5,856
- Normal images: 1,583 (27.03%)
- Pneumonia images: 4,273 (72.97%)
Data Cleaning and EDA
The dataset was checked for missing values, duplicate image names, duplicate image paths, and corrupted images. No missing values, duplicates, or corrupted images were found.

Exploratory data analysis was performed to examine class distribution and understand the dataset before model development.
Data Preprocessing
The following preprocessing steps were applied:
1. Converted images to grayscale.
2. Resized and padded images to 224 × 224 pixels.
3. Normalised pixel values to the range 0–1.
4. Converted grayscale images into three-channel images for DenseNet121 compatibility.
5. Split the dataset into training, validation, and testing sets using stratified sampling.
Dataset	Images
Training	4,684
Validation	586
Testing	586
Total	5,856

Data Augmentation
Data augmentation was applied to training images using rotation, zoom, horizontal flipping, and width and height shifts to improve the model's ability to generalise to unseen images.
Model Architecture
DenseNet121 was used as the base model through transfer learning. The classification layers included:
- Dense layer with 256 units
- Dropout layer
- Output layer with one neuron and sigmoid activation
The model was compiled using the Adam optimiser, binary cross-entropy loss, and accuracy, precision, recall, and AUC metrics.
Model Performance
The reported evaluation results were:
Metric	Score
Accuracy	97.78%
Precision	99.05%
Recall	97.90%
F1-score	98.47%
ROC-AUC	99.31%

These results reflect evaluation on the project's test split. They do not establish independent cross-domain generalisation or clinical validation.
Visualisations
The project includes visualisations of training accuracy, training loss, the confusion matrix, and the ROC curve to help assess model performance.
Technologies Used
- Python
- TensorFlow and Keras
- DenseNet121
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- PIL (Pillow)
- Jupyter Notebook
Future Scope
- Evaluate the model on independent datasets from different hospitals.
- Investigate class imbalance and domain adaptation techniques.
- Compare DenseNet121 with other deep learning architectures.
- Improve model interpretability using Grad-CAM.
- Conduct further validation before considering any clinical application.
Disclaimer
This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.
















i want full












Worked for 23s
Searched files and read full content













Here is the full README text in simple, plain text for your Cross-Domain Pneumonia Classification project. You can copy it directly into your GitHub README.md editor. It includes the project overview, dataset, preprocessing, model architecture, evaluation, repository structure, limitations, and references.

CROSS-DOMAIN PNEUMONIA CLASSIFICATION USING DEEP LEARNING AND DENSENET121

Research Project | Medical Image Classification | Transfer Learning

Researcher: Niharika Rawat

GitHub Repository: https://github.com/niharikarawat10/cross-domain-pneumonia-classification

ABSTRACT

Pneumonia is a respiratory infection that can be investigated using chest X-ray imaging. Deep learning techniques, particularly convolutional neural networks (CNNs), provide a method for learning image features and classifying medical images.

This project investigates a deep learning-based approach for binary classification of chest X-ray images into NORMAL and PNEUMONIA categories using transfer learning with DenseNet121. The workflow covers dataset collection, exploratory data analysis (EDA), image integrity checks, stratified data splitting, image preprocessing, data augmentation, model construction, fine-tuning, training, and performance evaluation.

The dataset contains 5,856 chest X-ray images. Images are converted to grayscale, padded to 224 × 224 pixels, normalised, and replicated into three channels to match the input shape of the DenseNet121 architecture. Training-time augmentation and class weighting are used to address variation in the training images and class imbalance.

On the reported test split of 586 images, the trained model achieved 97.78% accuracy, 99.05% precision, 97.90% recall, and an F1-score of 98.47%. The reported ROC-AUC was approximately 99.31%. These results describe performance on the current experimental split and should not be interpreted as evidence of clinical readiness or independent cross-domain generalisation.

1. INTRODUCTION

Chest X-ray image classification is an important research application of computer vision in medical imaging. Manually interpreting radiographs requires appropriate clinical expertise, and automated image-classification methods can be investigated as decision-support research tools.

This project uses DenseNet121, a convolutional neural network architecture based on dense connections between layers. Transfer learning allows a model initialised with pretrained ImageNet weights to be adapted to a new image-classification task.

The project focuses on developing and evaluating a reproducible pipeline for distinguishing normal chest X-rays from images labelled as showing pneumonia.

Research Objectives
- Collect and organise chest X-ray image paths and class labels.
- Perform exploratory data analysis to understand class distribution and sample images.
- Check the dataset for missing metadata, duplicate records, and corrupted images.
- Prepare consistent image dimensions and pixel values for model input.
- Apply data augmentation and class weighting during training.
- Adapt a pretrained DenseNet121 model for binary classification.
- Evaluate model performance using classification metrics and visualisations.
- Document the current experiment and identify the work needed for independent cross-domain validation.
2. DATASET DESCRIPTION

The project uses the Chest X-Ray Images (Pneumonia) dataset distributed through Kaggle.

Dataset: Chest X-Ray Images (Pneumonia) — Kaggle

Dataset link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Image format: JPEG

Task: Binary image classification

Classes: NORMAL and PNEUMONIA

Total images: 5,856

Class Distribution

NORMAL
Label: 0
Number of images: 1,583
Percentage: 27.03%

PNEUMONIA
Label: 1
Number of images: 4,273
Percentage: 72.97%

Total images: 5,856
Total percentage: 100%

The dataset is imbalanced, with pneumonia-labelled images representing approximately 72.97% of the collection. This imbalance is considered during training through class weighting.

3. EXPLORATORY DATA ANALYSIS (EDA)

EDA was performed to inspect the dataset structure, image records, label distribution, and representative images.

The initial metadata table contains the following columns:
- image_name: the image filename.
- image_path: the full path to the image.
- label: the numerical class label.
The label was assigned from the class directory in the image path, with NORMAL = 0 and PNEUMONIA = 1.

EDA Activities
1. Inspected the original train, test, and validation directory structure.
2. Collected image filenames and paths into a Pandas DataFrame.
3. Examined the number of records and unique image paths.
4. Calculated the class distribution.
5. Visualised class frequencies.
6. Displayed representative normal and pneumonia images for visual inspection.
The class distribution showed that pneumonia-labelled images substantially outnumbered normal images. This observation informed the use of class weighting during model training.

4. DATA CLEANING AND INTEGRITY CHECKS

The image metadata and image files were checked before preprocessing.

Reported Results:
- Total image records: 5,856
- Missing values in metadata columns: 0
- Duplicate image names: 0
- Duplicate image paths: 0
- Duplicate unique paths: 0
- Corrupted images detected using PIL verification: 0
The checks did not identify missing metadata values, duplicate filenames, duplicate image paths, or unreadable images in the collected dataset.

These checks establish basic data integrity; they do not independently establish that the dataset is free from duplicate patients, related-image leakage, label errors, or acquisition bias.

5. TRAIN, VALIDATION, AND TEST SPLIT

The collected records were split using train_test_split with stratification on the class label and a random seed of 42.

The original dataset's train, test, and validation directories were combined before the new stratified split was created.

Training set: 4,684 images (80%)

Validation set: 586 images (10%)

Testing set: 586 images (10%)

Total: 5,856 images (100%)

The training set was used to optimise model parameters.

The validation set was used to monitor performance and guide training callbacks.

The test set was used for final evaluation after training.

Experimental Consideration

Because images from the original directory splits were combined and re-split, this experiment does not preserve the original test partition. The resulting test metrics should therefore be interpreted as results from the new stratified split. Future validation should use an appropriately independent dataset or a clearly defined external test domain.

6. IMAGE PREPROCESSING

All images were transformed into a consistent representation before being supplied to the model.

Preprocessing Pipeline
1. Image loading: Images were opened using the Python Imaging Library (PIL).
2. Grayscale conversion: Each image was converted to grayscale using convert('L').
3. Size standardisation: ImageOps.pad was used to produce a 224 × 224 image while preserving the image's aspect ratio and adding padding where necessary.
4. Array conversion: Each processed image was converted into a NumPy array.
5. Normalisation: Pixel values were divided by 255.0 to scale them to the range from 0 to 1.
6. Channel expansion: A grayscale channel dimension was added, producing an image shape of (224, 224, 1).
7. Three-channel conversion: The grayscale channel was repeated three times to produce the three-channel input required by DenseNet121.
Input Dimensions

After grayscale preprocessing: (224, 224, 1)

After three-channel replication: (224, 224, 3)

Resulting Array Shapes

x_train_rgb: (4684, 224, 224, 3)

x_val_rgb: (586, 224, 224, 3)

x_test_rgb: (586, 224, 224, 3)

The same image-size and normalisation procedure was used for the training, validation, and test images.

Implementation Note

The current pipeline uses pixel scaling to [0, 1] and repeats grayscale values across three channels. This is the preprocessing implemented in the experiment; it should not be confused with the separate preprocess_input transformation provided for pretrained DenseNet models.

7. DATA AUGMENTATION AND CLASS IMBALANCE

Data augmentation was applied to training images to introduce controlled image variation during model fitting.

Configured Augmentation Operations
- Rotation: Up to 10 degrees
- Zoom: 0.1
- Horizontal flip: Enabled
- Width shift: 0.05
- Height shift: 0.05
Augmentation is applied through Keras ImageDataGenerator to the training data. Validation and test generators use an unaugmented data generator.

The batch size was set to 32.

Class Weighting

The class weights were calculated from the training labels using scikit-learn's compute_class_weight utility and passed to model.fit().

Class weighting helps account for the unequal number of normal and pneumonia-labelled examples by adjusting the contribution of each class to the training loss.

8. MODEL ARCHITECTURE: DENSENET121

DenseNet121 was selected as the feature-extraction backbone for the binary image-classification task.

The model was initialised with ImageNet pretrained weights and configured with include_top=False, removing the original ImageNet classification head.

Architecture Overview
1. DenseNet121 backbone: Extracts hierarchical image features from the chest X-ray input.
2. Global Average Pooling 2D: Reduces the spatial feature maps into a feature vector.
3. Dense layer: 256 units with ReLU activation.
4. Dropout: Rate of 0.5 to help reduce overfitting.
5. Output layer: One unit with sigmoid activation to produce a binary-classification probability.
The backbone was made trainable, and the implementation froze the earlier layers while leaving the final 30 backbone layers trainable. This configuration combines pretrained feature representations with task-specific fine-tuning.

Compilation Settings

Optimiser: Adam

Initial learning rate: 0.0001

Loss function: Binary cross-entropy

Metrics: Accuracy, precision, recall, AUC

Input shape: (224, 224, 3)

Output activation: Sigmoid

Binary cross-entropy is used for the two-class prediction task. Precision, recall, accuracy, and AUC are monitored during model training and validation.

9. TRAINING STRATEGY

The configured maximum number of training epochs was 20. Training used the training generator and monitored validation performance.

Three callbacks were configured:
1. EarlyStopping: Monitors validation loss, uses a patience of 3, and restores the best weights.
2. ReduceLROnPlateau: Monitors validation loss, reduces the learning rate by a factor of 0.2 after 2 epochs without improvement, and uses a minimum learning rate of 1e-6.
3. ModelCheckpoint: Saves the model with the best validation loss to best_densenet.keras.
These callbacks are intended to control optimisation, respond to a plateau in validation performance, and preserve the best checkpoint observed during training.

The saved model file is included in this repository.

10. MODEL EVALUATION

The saved model was evaluated on the 586-image test split. The reported evaluation includes accuracy, precision, recall, F1-score, ROC-AUC, a classification report, and a confusion matrix.

Reported Performance

Accuracy: 97.78%

Precision — positive class (PNEUMONIA): 99.05%

Recall — positive class (PNEUMONIA): 97.90%

F1-score — positive class (PNEUMONIA): 98.47%

ROC-AUC: 99.31%

Test images: 586

These values are the results reported by the current notebook. They are experimental results for the specified dataset split, not a guarantee of performance on other hospitals, patient populations, imaging equipment, or clinical settings.

Classification Report

NORMAL:
- Precision: 0.94
- Recall: 0.97
- F1-score: 0.96
- Support: 158
PNEUMONIA:
- Precision: 0.99
- Recall: 0.98
- F1-score: 0.98
- Support: 428
Overall accuracy: 0.98 across 586 images.

Macro average:
- Precision: 0.97
- Recall: 0.98
- F1-score: 0.97
Weighted average:
- Precision: 0.98
- Recall: 0.98
- F1-score: 0.98
Confusion Matrix

The reported confusion matrix, with rows representing actual labels and columns representing predicted labels in the order NORMAL, PNEUMONIA, is:

Actual NORMAL, predicted NORMAL: 154

Actual NORMAL, predicted PNEUMONIA: 4

Actual PNEUMONIA, predicted NORMAL: 9

Actual PNEUMONIA, predicted PNEUMONIA: 419

According to this matrix:
- 154 normal images were correctly classified as normal.
- 4 normal images were incorrectly classified as pneumonia.
- 419 pneumonia images were correctly classified as pneumonia.
- 9 pneumonia images were incorrectly classified as normal.
The false-negative cases are particularly important to investigate in a medical-image classification context.

11. VISUAL RESULTS

The repository contains plots generated during model evaluation.

Accuracy Curve

Image file: accuracy_curve.png

GitHub image path:
Kaggle Pediatric Pneumonia Chest X-Ray Images (Pneumonia)/accuracy_curve.png

The accuracy curve compares training and validation accuracy over the recorded epochs.

Loss Curve

Image file: loss_curve.png

GitHub image path:
Kaggle Pediatric Pneumonia Chest X-Ray Images (Pneumonia)/loss_curve.png

The loss curve shows the recorded training and validation loss and can help identify convergence patterns or possible overfitting.

Confusion Matrix

Image file: confusion_matrix.png

GitHub image path:
Kaggle Pediatric Pneumonia Chest X-Ray Images (Pneumonia)/confusion_matrix.png

The confusion matrix summarises correct and incorrect predictions for both classes.

ROC Curve

Image file: roc_curve_densenet.png

GitHub image path:
Kaggle Pediatric Pneumonia Chest X-Ray Images (Pneumonia)/roc_curve_densenet.png

The ROC curve visualises the relationship between the true-positive rate and false-positive rate across classification thresholds.

12. REPOSITORY STRUCTURE

cross-domain-pneumonia-classification/

README.md

.gitignore

Kaggle Pediatric Pneumonia Chest X-Ray Images (Pneumonia)/
- pneumonia-detection.ipynb
- best_densenet.keras
- accuracy_curve.png
- loss_curve.png
- confusion_matrix.png
- roc_curve_densenet.png
The notebook contains the implementation and experimental workflow. The saved Keras model and evaluation plots are provided as supporting project artefacts.

13. TECHNOLOGIES USED

Programming language: Python

Data manipulation: Pandas, NumPy

Visualisation: Matplotlib, Seaborn

Image processing: PIL / Pillow

Machine learning utilities: scikit-learn

Deep learning: TensorFlow and Keras

Model architecture: DenseNet121

Development environment: Jupyter Notebook / Kaggle notebook environment

14. HOW TO EXPLORE THE PROJECT
1. Clone or download this repository.
2. Open pneumonia-detection.ipynb in Jupyter Notebook or JupyterLab.
3. Review the dataset collection, EDA, cleaning, preprocessing, training, and evaluation sections.
4. Ensure the dataset is downloaded and the notebook's dataset paths are updated for your environment.
5. Install compatible versions of the required Python libraries.
6. Run the notebook cells in sequence to reproduce the workflow.
The original notebook uses a Kaggle dataset path. Running it locally requires changing that path to the location of the downloaded dataset. Reproducing the exact results may also require matching the library versions, random seeds, hardware, and training configuration.

The trained model is available as best_densenet.keras. Loading and using it requires a compatible TensorFlow/Keras environment and matching input preprocessing.

15. LIMITATIONS AND FUTURE RESEARCH

The current results should be interpreted within the scope of this experiment.

Potential directions for further research include:
- Independent-domain evaluation: Test on an external dataset from a different source or institution.
- Patient-level splitting: Establish patient identities where possible and ensure related images cannot leak across training and test sets.
- Preprocessing comparison: Evaluate the current [0, 1] normalisation against the preprocessing expected by pretrained DenseNet121.
- Robustness analysis: Examine sensitivity to image quality, acquisition conditions, and changes in class distribution.
- Error analysis: Inspect false-negative and false-positive cases.
- Threshold analysis: Evaluate different classification thresholds and their effects on sensitivity and specificity.
- Explainability: Use and validate methods such as Grad-CAM to investigate image regions influencing predictions.
- Reproducibility: Record dependency versions, training history, and the exact data split used for each experiment.
These are proposed research directions and should not be interpreted as completed experiments unless supported by additional results.

16. RESEARCH AND RESPONSIBLE USE STATEMENT

This repository documents an experimental deep learning project for educational and research purposes. The reported metrics are based on the current experimental test split and do not establish clinical validity.

The model is not a medical diagnostic device and must not be used as a substitute for assessment by a qualified healthcare professional. Clinical use would require appropriate independent validation, safety evaluation, bias assessment, and regulatory review.

17. REFERENCES

1. Paul Mooney, Chest X-Ray Images (Pneumonia) dataset, Kaggle.
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. Huang, G., Liu, Z., van der Maaten, L., and Weinberger, K. Q. “Densely Connected Convolutional Networks.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
https://arxiv.org/abs/1608.06993

3. TensorFlow, Keras Applications: DenseNet.
https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121

4. TensorFlow, ImageDataGenerator documentation.
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

5. scikit-learn, Model evaluation documentation.
https://scikit-learn.org/stable/modules/model_evaluation.html

Author: Niharika Rawat

Project: Cross-Domain Pneumonia Classification Using Deep Learning and DenseNet121

Repository: https://github.com/niharikarawat10/cross-domain-pneumonia-classification
