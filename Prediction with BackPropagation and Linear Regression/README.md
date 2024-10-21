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

## Project Folder Structure

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


## Result 
1. EDA of dataset
   
![44](https://github.com/user-attachments/assets/16e7196c-08bc-423c-96e9-0c7424710a94)


## Comparison of Backpropagation (BP)
| Dataset     | Number of Layers | Layer Structure | Num Epochs | Learning Rate | Momentum | Activation Function | MAPE   |
|-------------|------------------|-----------------|------------|---------------|----------|---------------------|--------|
| Turbine     | 3                | 4,10,1          | 100        | 0.001         | 0.9      | Relu                | 0.2147 |
| Synthetic   | 3                | 9,10,1          | 1000       | 0.001         | 0.7      | Relu                | 0.651  |
| Boston      | 3                | 12,10,1         | 100        | 0.001         | 0.8      | Relu                | 56.27% |


### Discussion on BP:
- **Turbine Dataset**: The model has a relatively simple structure and performs reasonably well with a **MAPE of 0.2147** (21.47% average relative error).
- **Synthetic Dataset**: The model shows a higher error (**MAPE 0.651**) compared to the Turbine dataset, suggesting a need for further tuning or a more complex model.
- **Boston Dataset**: The model's predictions are the least accurate here, with a **MAPE of 56.27%**, indicating a significant relative error. The model may require further optimization for this dataset.

---

## Comparison of Backpropagation Feedforward (BP-F)

| Dataset     | MAPE    | MSE       | R² Score  |
|-------------|---------|-----------|-----------|
| Turbine     | 0.1678  | 0.0074    | 0.9935    |
| Synthetic   | 0.3698  | 0.0479    | 0.9549    |
| Boston      | N/A     | 0.0099    | 0.7263    |


![9888](https://github.com/user-attachments/assets/92b8bbb5-074f-4d8b-b92c-33edc34eee53)

### Discussion on BP-F:
- **Turbine Dataset**: The BP-F model performs exceptionally well with **MAPE 0.1678**, **MSE 0.0074**, and an **R² score of 0.9935**, explaining 99.35% of the variance in the data.
- **Synthetic Dataset**: Strong performance is reflected with **MAPE 0.3698**, **MSE 0.0479**, and **R² 0.9549**, capturing 95.49% of the variance.
- **Boston Dataset**: Although **MSE is low (0.0099)**, the **R² score (0.7263)** suggests only moderate model performance for this dataset.

![999](https://github.com/user-attachments/assets/8c928096-7693-43d0-9873-65c21b3dcb03)
![75555](https://github.com/user-attachments/assets/5e9d26d9-4e2c-4674-8874-a7ae7bb667f5)

---

## Comparison of Multiple Linear Regression (MLR)

| Dataset     | MAPE          | MSE      | R² Score  |
|-------------|---------------|----------|-----------|
| Turbine     | 20.39%        | 0.0268   | 0.9767    |
| Synthetic   | 22.81%        | 0.0280   | 0.9736    |
| Boston      | 261284035.52% | 0.0120   | 0.6688    |

### Discussion on MLR:
- **Turbine Dataset**: The MLR model shows relatively good performance with **MAPE 20.39%**, **MSE 0.0268**, and **R² score of 0.9767**.
- **Synthetic Dataset**: The model's performance is similar to Turbine with **MAPE 22.81%**, **MSE 0.0280**, and **R² score of 0.9736**.
- **Boston Dataset**: The **extremely high MAPE (261284035.52%)** suggests significant issues in predicting the Boston dataset with the MLR model, despite the **low MSE (0.0120)** and **moderate R² score (0.6688)**.

---

## Summary:
- **Backpropagation (BP)**: Performs reasonably well on the Turbine dataset but struggles with the Boston dataset, indicating a need for further optimization.
- **Backpropagation Feedforward (BP-F)**: Shows strong performance across all datasets, particularly with a high R² score for the Turbine dataset.
- **Multiple Linear Regression (MLR)**: Performs well for the Turbine and Synthetic datasets but fails to predict the Boston dataset effectively due to an extremely high MAPE.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [Pranjal Jain](https://github.com/4Pranjal)
