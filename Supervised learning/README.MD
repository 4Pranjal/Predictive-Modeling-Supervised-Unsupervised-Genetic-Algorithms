# Machine Learning Classification README

## Overview

This repository contains Python code for performing data classification using various machine learning algorithms: Support Vector Machines (SVM), Back-Propagation (BP), and Multiple Linear Regression (MLR). The objective is to analyze datasets, select appropriate parameters for each algorithm, and evaluate the classification results.

## Objective

The objective of this project is to:

1. Analyze datasets: Ring data set, Bank dataset, and Boston House Price.
2. Perform supervised training of classification models using SVM, BP, and MLR.
3. Select optimal parameters for each algorithm through cross-validation.
4. Evaluate the quality of classification results on test and validation sets.
5. Compare confusion matrices and ROC curves for the three algorithms.

## Dataset

The following datasets are used for training and evaluation:

1. Ring data set
2. Bank dataset
3. Boston House Price

## Libraries Used

- Python libraries for SVM: scikit-learn, LibSVM
- Python libraries for BP: TensorFlow, Keras
- Python libraries for MLR: scikit-learn

## Folder Structure for Supervised Learning
Predictive-Modeling-Supervised-Unsupervised-Genetic-Algorithms/

```bash 
│
├── Supervised_learning/
│   ├── dataset/                             # Contains datasets for training models
│   │   └── boston_housing.csv               # Boston Housing dataset
│   │   └── Turbine_Data.csv                 # Turbine dataset
│   ├── figures/                             # Visualization of results
│   │   └── loss_acc.png                     # Loss and accuracy plot
│   │   └── performance_comparison.png       # Performance comparison between models
│   ├── model/                               # Trained model files and related scripts
│   │   └── bp_model.py                      # Backpropagation model script
│   │   └── linear_regression.py             # Linear regression model script
│   ├── results/                             # Folder to store results, reports, and metrics
│   │   └── BP_results.txt                   # Backpropagation model performance results
│   │   └── MLR_results.txt                  # Multiple linear regression performance results
│   ├── supervised_learning_notebook.ipynb   # Jupyter notebook for analysis and modeling
│   ├── requirements.txt                     # File listing project dependencies
│   ├── README.md                            # Overview and instructions for the project
│   ├── LICENSE                              # License information for the project

```

## Instructions

To implement the code, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/4Pranjal/Machine_Learning_Project.git
   ```

2. Install the required libraries:

   ```bash
   pip install scikit-learn tensorflow keras
   ```

3. Navigate to the directory:

   ```bash
   cd Machine_Learning_Project
   ```

4. Run the Python scripts provided in the repository for parameter selection and evaluation.

 ## MLR (Multiple Linear Regression)

| **Dataset**               | **Accuracy**    | **AUC**         | **Confusion Matrix**               | **Classification Error** |
|---------------------------|-----------------|-----------------|-----------------------------------|--------------------------|
| **Separable Ring Dataset** | 0.4665          | 0.443862987630827| [[933, 118], [949, 0]]             | 0.4665                   |
| **Merged Ring Dataset**    | 0.5585          | 0.5             | [[1117, 0], [883, 0]]              | 0.4415                   |
| **Bank Marketing Dataset** | 0.912621359     | 0.650403226     | [[726, 18], [54, 26]]              | -                        |
| **Boston**                | 0.862745098     | 0.949220311     | [[50, 11], [3, 38]]                | 0.1372549019607842        |

---

## SVM (Support Vector Machine)

| **Dataset**               | **Accuracy**    | **AUC**         | **Confusion Matrix**               | **Classification Error** |
|---------------------------|-----------------|-----------------|-----------------------------------|--------------------------|
| **Separable Ring Dataset** | 0.9764          | 0.977726444     | [[5108, 225], [11, 4656]]          | 0.0236                   |
| **Merged Ring Dataset**    | 0.9608          | -               | [[4999, 334], [74, 4593]]          | 0.0395                   |
| **Bank Marketing Dataset** | 0.901699029     | 0.89875672      | [[726, 18], [63, 17]]              | 0.098300971               |
| **Boston**                | 0.81372549      | 0.891566265     | [[83, 0], [19, 0]]                 | 0.18627451                |

---

## BP (Backpropagation)

| **Dataset**               | **Accuracy**    | **AUC**         | **Confusion Matrix**               | **Classification Error** |
|---------------------------|-----------------|-----------------|-----------------------------------|--------------------------|
| **Separable Ring Dataset** | 0.9315          | -               | [[975, 76], [61, 888]]             | 0.0375                   |
| **Merged Ring Dataset**    | 0.7600          | 0.759634639     | [[852, 265], [215, 668]]           | 0.2415                   |
| **Bank Marketing Dataset** | 0.905339777     | 0.67983871      | [[714, 30], [48, 32]]              | 0.201456308               |
| **Boston**                | 0.901960784     | 0.818783069     | [[83, 1], [9, 9]]                  | 0.098039216               |

 ## Discussion

- **MLR Model Performance**:  
  The performance of the MLR model varies significantly across different datasets, indicating that the characteristics of the data play a crucial role in model performance.
  - **Separable Ring Dataset**: MLR struggles with accuracy and discriminatory power, resulting in poor model performance.
  - **Merged Ring Dataset**: While the accuracy improves, the AUC of 0.5 reflects that the model lacks discriminatory power.
  - **Bank Marketing Dataset**: The MLR model performs well here, achieving high accuracy and a reasonable AUC, indicating good predictive capabilities.
  
- **SVM Model Performance**:  
  - Achieves high accuracy and AUC on the **Separable Ring Dataset** but shows poor performance on the **Merged Ring Dataset**.
  - On the **Bank Marketing Dataset**, the model demonstrates good accuracy and AUC values.

- **BP Model Performance**:  
  - Performs well on the **Separable Ring Dataset**, but struggles with the **Merged Ring Dataset**.
  - On the **Bank Marketing Dataset**, the model achieves good accuracy and AUC.
  - For the **Boston Dataset**, BP has high accuracy but a lower AUC, suggesting a gap between model accuracy and its ability to discriminate between classes.

---

### Key Metric Comparisons:

- **Accuracy**:
  - MLR: Highest accuracy at **0.9126**, indicating the best overall correctness in predictions.
  - BP: **0.9053**.
  - SVM: **0.9017**.

- **AUC (Area Under the ROC Curve)**:
  - **SVM**: Highest AUC at **0.8988**, indicating better discrimination ability.
  - **MLR**: AUC of **0.6504**.
  - **BP**: AUC of **0.6798**.

- **Confusion Matrix**:
  - MLR and SVM show similar confusion matrices but differ in the number of false positives (FP) and false negatives (FN).
  - BP shows a different distribution of TP (True Positives), TN (True Negatives), FP, and FN compared to MLR and SVM.

- **Classification Error**:
  - **MLR**: Lowest classification error at **0.0874**, suggesting fewer misclassifications.
  - **BP**: Classification error of **0.0947**.
  - **SVM**: Classification error of **0.0983**.

- **Precision, Recall, Specificity, F1 Score**:
  - These metrics can be calculated using the confusion matrix values to provide deeper insights into the model’s performance, especially regarding false positives and false negatives.

---

In conclusion, each model performs differently based on the dataset, with MLR achieving the highest accuracy overall, while SVM demonstrates superior discriminatory power based on AUC.

## Contributors

- [Pranjal Jain](https://github.com/4Pranjal)

