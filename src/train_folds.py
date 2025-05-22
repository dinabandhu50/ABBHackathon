import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from model_dispatcher import models
from config import settings


def train_folds(fold, df, model_name):
    # separate cols
    target_cols = ["Item_Outlet_Sales"]
    feature_cols = df.drop(
        columns=["Item_Identifier", "Outlet_Identifier", "kfold", "Item_Outlet_Sales"]
    ).columns.tolist()

    # separate train valid
    df_train = df[df.kfold != fold]
    df_valid = df[df.kfold == fold]

    # feature and target
    xtrain = df_train[feature_cols]
    xvalid = df_valid[feature_cols]
    ytrain = df_train[target_cols].values.ravel()
    yvalid = df_valid[target_cols].values.ravel()

    # model training
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not found in model_dispatcher")

    # model train
    if model_name == "xgb":
        model.fit(
            xtrain,
            ytrain,
            eval_set=[(xvalid, yvalid)],
            verbose=False,
        )

    elif model_name == "cat":
        model.fit(xtrain, ytrain, eval_set=(xvalid, yvalid), early_stopping_rounds=1000)
    else:
        model.fit(xtrain, ytrain)

    # y predictions
    y_pred_train = model.predict(xtrain)
    y_pred_valid = model.predict(xvalid)

    # metrics
    rmse_train = root_mean_squared_error(ytrain, y_pred_train)
    rmse_valid = root_mean_squared_error(yvalid, y_pred_valid)

    print(
        f"model={model_name}, fold={fold}, RMSE: train {rmse_train:.4f}, valid {rmse_valid:.4f}"
    )

    # save the model
    os.makedirs(os.path.join(settings.PROJECT_DIR, "models", model_name), exist_ok=True)
    joblib.dump(
        model,
        os.path.join(settings.PROJECT_DIR, "models", model_name, f"model_{fold}.pkl"),
    )

    result_df = pd.concat(
        [
            df_valid[["kfold"]].reset_index(drop=True),
            pd.DataFrame(y_pred_valid, columns=target_cols),
            pd.DataFrame(yvalid, columns=target_cols),
        ],
        axis=1,
    )
    fold_dict = {"train_rmse": rmse_train, "valid_rmse": rmse_valid}
    return result_df, fold_dict


if __name__ == "__main__":
    # model_names = ["rf", "xgb", "cat"]
    model_names = [
        # "rf",
        "xgb",
    ]

    folds = 5
    df = pd.read_csv(os.path.join(settings.DATA_DIR, "fe", "00", "train_folds.csv"))

    for model_name in model_names:
        start_time = time.time()
        rmse_train_avg = []
        rmse_valid_avg = []

        dfs = []
        for fold in range(folds):
            fold_df, fold_dict = train_folds(fold=fold, df=df, model_name=model_name)
            dfs.append(fold_df)
            rmse_train_avg.append(fold_dict["train_rmse"])
            rmse_valid_avg.append(fold_dict["valid_rmse"])

        dfs = pd.concat(dfs)
        print(
            f"model={model_name}, Average, rmse_train={np.mean(rmse_train_avg):2.7f} \u00b1 {np.std(rmse_train_avg):2.7f}, rmse_valid={np.mean(rmse_valid_avg):2.7f} \u00b1 {np.std(rmse_valid_avg):2.7f} "
        )
        dfs.to_csv(
            os.path.join(settings.PROJECT_DIR, "oofs", f"{model_name}_preds.csv"),
            index=False,
        )
        print(f"---------- {time.time() - start_time:.4f} seconds ----------")
