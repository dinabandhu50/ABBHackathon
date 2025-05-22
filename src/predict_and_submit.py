import os
import joblib
import numpy as np
import pandas as pd

from config import settings
from model_dispatcher import models


def predict_from_folds(
    model_name: str, test_df: pd.DataFrame, n_folds: int = 5
) -> np.ndarray:
    feature_cols = test_df.drop(
        columns=["Item_Identifier", "Outlet_Identifier"]
    ).columns.tolist()

    predictions = []

    for fold in range(n_folds):
        model_path = os.path.join(
            settings.PROJECT_DIR, "models", model_name, f"model_{fold}.pkl"
        )
        model = joblib.load(model_path)

        preds = model.predict(test_df[feature_cols])
        predictions.append(preds)

    # Average predictions across folds
    final_preds = np.mean(predictions, axis=0)
    return final_preds


def main():
    model_name = "xgb"  # change as needed
    test_path = os.path.join(settings.DATA_DIR, "fe", "00", "test.csv")
    submission_path = os.path.join(
        settings.PROJECT_DIR, "submissions", f"{model_name}_submission_fe_00.csv"
    )

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    test_df = pd.read_csv(test_path)

    # predict
    preds = predict_from_folds(model_name, test_df, n_folds=5)

    # construct submission
    submission = pd.DataFrame(
        {
            "Item_Identifier": test_df["Item_Identifier"],
            "Outlet_Identifier": test_df["Outlet_Identifier"],
            "Item_Outlet_Sales": preds,
        }
    )

    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
