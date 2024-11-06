import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    fit_intercept = True

    with mlflow.start_run():
        mlflow.log_param("fit_intercept", fit_intercept)

        model = LinearRegression(fit_intercept=fit_intercept)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error: {mse}")

        mlflow.log_metric("mse", mse)

        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
