# Predictive Modeling README

## Overview

This repository contains Python code for predictive modeling tasks using various machine learning techniques. The code implements data preprocessing, visualization, and prediction using libraries such as Pandas, Seaborn, Matplotlib, and Scikit-learn.

## Objective

The objective of this project is to predict outcomes using three different approaches:

1. Neural network with back-propagation implemented (BP).
2. Neural network with back-propagation using free software (BP-F).
3. Multiple linear regression using free software (MLR-F).

## Dataset

The following datasets are used for training and evaluation:

1. Turbine dataset
2. Synthetic dataset
3. Boston House Price: This dataset is available in the sklearn datasets module. More information can be found [here](https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html).

## Libraries Used

- numpy
- pandas
- scikit-learn
- matplotlib

## Instructions

To implement the code, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/4Pranjal/Machine_Learning_Project.git
   ```

2. Install the required libraries:

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. Navigate to the directory:

   ```bash
   cd Prediction with BackPropagation and Linear Regression
   ```

4. Run the Python scripts provided in the repository for data preprocessing, visualization, and predictive modeling.

## Usage

The main Python scripts are as follows:

- `data_preprocessing.py`: This script performs data preprocessing tasks such as cleaning, scaling, and splitting the dataset into training and testing sets.
  
- `visualization.py`: This script generates visualizations of the dataset to gain insights and understand the data distribution.
  
- `neural_network_backpropagation.py`: Implements a neural network with back-propagation for prediction.
  
- `neural_network_backpropagation_free.py`: Implements a neural network with back-propagation using free software for prediction.
  
- `multiple_linear_regression_free.py`: Implements multiple linear regression using free software for prediction.

## Example Usage

```bash
python data_preprocessing.py
python visualization.py
python neural_network_backpropagation.py
python neural_network_backpropagation_free.py
python multiple_linear_regression_free.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [Pranjal Jain](https://github.com/4Pranjal)
