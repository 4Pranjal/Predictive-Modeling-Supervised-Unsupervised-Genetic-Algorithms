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

## Example Usage

```bash
python parameter_selection.py
python evaluation.py
```

## Contributors

- [Pranjal Jain](https://github.com/4Pranjal)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
