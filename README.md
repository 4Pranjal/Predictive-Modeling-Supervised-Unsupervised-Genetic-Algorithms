## Predictive Modelling, Supervised, Unsupervised & Genetic Algorithms: AI/ML Project

## Overview

This repository contains Python code for various machine learning projects include Supervised, Unsupervised, Predicitve & Genetic Algorithm. Each part of project addresses a specific problem or task and utilizes different machine learning algorithms and techniques.

## Getting Started
To get started with the complete projects including all the algorithm, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/4pranjal/machine-learning-projects.git
   ```

2. Navigate to the directory:

   ```bash
   cd machine-learning-projects
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Projects
To implement the specific part of algorithm, please follow the instructions.

1. **Predictive Modeling**
   - **Objective:** Perform predictive modeling using neural networks and multiple linear regression.
   - **Libraries Used:** numpy, pandas, scikit-learn, matplotlib
   - **Instructions:** Follow the provided README for implementation details.
   - And overview of EDA of data is shown below:-
![44](https://github.com/user-attachments/assets/16e7196c-08bc-423c-96e9-0c7424710a94)
---

### Project Folder Structure
The following structure represents the organization of the project files:
```bash
├── Prediction_with_BackPropagation_and_Linear_Regression/
│   ├── data/
│   │   └── raw/                       # Raw datasets used for modeling
│   │   └── processed/                 # Cleaned and preprocessed datasets
│   ├── models/                        # Trained model files (e.g., weights or serialized models)
│   │   └── backpropagation_model.pkl
│   │   └── linear_regression_model.pkl
│   ├── notebooks/                     # Jupyter notebooks for exploratory analysis and training
│   │   └── backpropagation_analysis.ipynb
│   │   └── linear_regression_analysis.ipynb
│   ├── scripts/                       # Python scripts for preprocessing, training, and testing
│   │   └── preprocess_data.py
│   │   └── train_backpropagation.py
│   │   └── train_linear_regression.py
│   ├── results/                       # Results from the experiments (e.g., plots, reports, performance metrics)
│   │   └── model_performance.txt
│   │   └── loss_accuracy_plot.png
│   ├── requirements.txt               # Python dependencies
│   ├── README.md                      # Project overview and instructions
│   ├── LICENSE                        # License for the project
```
---

2. **Supervised Learning with Classification**

This Supervised Learning contains Python code for performing data classification using various machine learning algorithms: Support Vector Machines (SVM), Back-Propagation (BP), and Multiple Linear Regression (MLR). The objective is to analyze datasets, select appropriate parameters for each algorithm, and evaluate the classification results.

## Objective

The objective of this project is to:

1. Analyze datasets: Ring data set, Bank dataset, and Boston House Price.
2. Perform supervised training of classification models using SVM, BP, and MLR.
3. Select optimal parameters for each algorithm through cross-validation.
4. Evaluate the quality of classification results on test and validation sets.
5. Compare confusion matrices and ROC curves for the three algorithms.
   
   ![R2](https://github.com/user-attachments/assets/b0006652-a641-4fe7-ae78-af34a604be1c)

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

```bash
Predictive-Modeling-Supervised-Unsupervised-Genetic-Algorithms/
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
---

4. **Unsupervised Learning Comparison**
   - **Objective:** Compare unsupervised learning techniques including PCA, t-SNE, k-means, AHC, and SOM.
   - **Libraries Used:** numpy, pandas, scikit-learn, matplotlib
   - **Instructions:** Follow the provided README for implementation details.

For each algorithm:

- PCA: Plot PCA projection in two dimensions and scree plot with accumulated variance.
- t-SNE: Plot t-SNE projection in two dimensions with different parameter settings.
- k-means: Use k-means for different values of k and compare obtained classes with real ones.
- AHC: Use UPGMA and complete linkage methods of AHC and plot resulting dendrograms.
- SOM: Visualize the data using SOM with different settings and plot component planes.

## Folder Structure for Unsupervised Learning

```bash
Predictive-Modeling-Supervised-Unsupervised-Genetic-Algorithms/
│
├── Unsupervised_learning/
│   ├── dataset/                           # Contains datasets used for unsupervised learning
│   │   └── bank_data.csv                  # Bank dataset for clustering
│   │   └── boston_housing.csv             # Boston Housing dataset for clustering
│   ├── model/                             # Trained model files and related scripts
│   │   └── kmeans.py                      # K-Means clustering implementation
│   │   └── hierarchical_clustering.py     # Hierarchical clustering implementation
│   │   └── pca.py                         # Principal Component Analysis (PCA) script
│   ├── results/                           # Folder to store results, reports, and metrics
│   │   └── kmeans_results.txt             # Results of K-Means clustering
│   │   └── hierarchical_results.txt       # Results of Hierarchical clustering
│   ├── figures/                           # Visualizations of clustering results
│   │   └── kmeans_clusters.png            # Visualization of K-Means clusters
│   │   └── hierarchical_dendrogram.png    # Dendrogram for Hierarchical clustering
│   ├── unsupervised_learning_notebook.ipynb # Jupyter notebook for analysis and clustering
│   ├── requirements.txt                   # Project dependencies for unsupervised learning
│   ├── README.md                          # Overview and instructions for the unsupervised learning part of the project
│   ├── LICENSE                            # License information for the project

```
### Instructions

To implement the code, follow these steps:

1. Clone the UnSupervised learning follow the steps to implement:

   ```bash
   git clone https://github.com/4Pranjal/Predictive-Modeling-Supervised-Unsupervised-Genetic-Algorithms.git
   ```

2. Install the required libraries:

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. Navigate to the directory:

   ```bash
   cd Predictive-Modeling-Supervised-Unsupervised-Genetic-Algorithms
   ```

4. Run the Python scripts provided in the repository for applying unsupervised learning techniques and comparing results.

### Usage

The main Python scripts are as follows:

- `A3_Boston_dataset.ipynb`: This script applies unsupervised learning techniques to the datasets and compares the results.
  
- `A3_Implementation.ipynb`: This script visualizes the results obtained from different unsupervised learning techniques.
  
---

4. **Optimization with Genetic Algorithms**
   - **Objective:** Solve the Traveling Salesman Problem using genetic algorithms.
   - **Libraries Used:** tsplib95
   - **Instructions:** Follow the provided README for implementation details.
![Picture4](https://github.com/user-attachments/assets/c4a837b5-fa9b-4b71-9fbe-cbf22a31eb07)


## Contributors

- [Pranjal Jain](https://github.com/4Pranjal)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
