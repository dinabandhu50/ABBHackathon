import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from catboost import CatBoostRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

# from sklearn.ensemble import StackingRegressor
from config import settings

SEED = int(settings.SEED)

rf_param = {
    "random_state": SEED,
    # "n_estimators": 500,
    # "max_depth": 12,
    # 'min_samples_split': 0.001,
    # "min_samples_leaf": 0.001,
    # 'max_features':'sqrt',
    # 'ccp_alpha':0.000001,
}

xgb_param = {
    "random_state": SEED,
    "verbosity": 0,
    "tree_method": "hist",
    "use_label_encoder": False,
    "booster": "gbtree",
    "n_estimators": 1000,  # Required for early stopping
    "learning_rate": 0.05,  # Optional but good to include for tuning
    "eval_metric": "rmse",  # Optional, can add if needed
    "early_stopping_rounds": 2,
    "eval_matric": "rmse",
}


cat_param = {
    "random_seed": SEED,
    "verbose": False,
    # 'task_type':"GPU",
    # "iterations": 10000,
    # "early_stopping_rounds": 1000,
    # "depth": 5,
    # "l2_leaf_reg": 12.06,
    # "bootstrap_type": "Bayesian",
    # "boosting_type": "Plain",
    # "loss_function": "MAE",
    # "eval_metric": "SMAPE",
    # "od_type": "Iter",  # type of overfitting detector
    # "od_wait": 40,
    # "has_time": True,
}

meta_rf_param = {
    "random_state": SEED + 273,
    "n_estimators": 500,
    "max_depth": 6,
    # 'min_samples_split': 0.01,
    # 'min_samples_leaf':0.001,
    "min_samples_leaf": 0.008,
    # 'max_features':'sqrt',
    # 'ccp_alpha':0.000001,
}

# model loading
model1 = CatBoostRegressor(**cat_param)
model2 = XGBRegressor(**xgb_param)
model3 = RandomForestRegressor(**rf_param, n_jobs=-1)
meta_model = RandomForestRegressor(**meta_rf_param, n_jobs=-1)


# stack model define and fitting
stack_param = {
    "regressors": [model1, model2, model3],
    "meta_regressor": meta_model,
    "use_features_in_secondary": True,
    # "verbose": 1,
    "verbose": 0,
}
skvote_param = {
    "estimators": [
        ("cat", model1),
        ("xgb", model2),
        ("rf", model3),
    ],
    "weights": [0.5, 0.3, 0.2],
}


# model dictionary
models = {
    "rf": RandomForestRegressor(**rf_param, n_jobs=-1),
    "xgb": XGBRegressor(**xgb_param),
    "cat": CatBoostRegressor(**cat_param),
    "stack": StackingRegressor(**stack_param),
    "skvote": VotingRegressor(**skvote_param, n_jobs=-1),
}
