# MLFlow Simple Project

This a simple project on how to use **MLFlow** to track machine learning experiments. The project trains a linear regression model using scikit-learn on the Diabetes dataset, logs parameters and metrics, and saves the model using MLflow.

## Requirements

- Python 3.6 or higher
- scikit-learn
- MLflow

## Setup

1. **Clone the repository:**

```bash
git clone git@github.com:iamhmh/mlflow_project.git
cd mlflow_project
```

2. **Create and activate a virtual environment:**
- On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
- On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt # all dependencies are in requirements.txt
```

## Usage

1. **Run the training script:**

```bash
python train.py
```

2. **Start the MLflow UI to visualize the results:**
```bash
mlflow ui
```
Open your web browser and navigate to `http://localhost:5000` to view the logged experiments.

## Project Structure

- `train.py`: The main script that trains the model and logs data to MLFlow.

## Description

- Data Loading: Loads the Diabetes dataset.
- Data Splitting: Splits the data into training and testing sets.
- Model Training: Trains a Linear Regression model.
- Logging with MLflow: Logs parameters, metrics, and the trained model.