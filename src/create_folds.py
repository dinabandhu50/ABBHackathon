import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import settings

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(settings.DATA_DIR, "raw", "train_v9rqX0R.csv"))
    df.loc[:, "kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Bin the target into discrete intervals for stratification
    num_bins = int(np.floor(1 + np.log2(len(df))))  # Sturge's rule
    print(f"Number of bins: {num_bins}")
    df["bins"] = pd.cut(df["Item_Outlet_Sales"], bins=num_bins, labels=False)

    # Apply StratifiedKFold on the binned target
    skf = StratifiedKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df["bins"])):
        df.loc[val_idx, "kfold"] = fold

    # Drop the temporary bins column
    df.drop("bins", axis=1, inplace=True)

    # Save processed data
    df.to_csv(
        os.path.join(settings.DATA_DIR, "processed", "train_folds.csv"), index=False
    )

    # Save test data
    test_data = pd.read_csv(os.path.join(settings.DATA_DIR, "raw", "test_AbJTz2l.csv"))
    test_data.to_csv(
        os.path.join(settings.DATA_DIR, "processed", "test.csv"), index=False
    )
    # Save sample submission
    sample_submission = pd.read_csv(
        os.path.join(settings.DATA_DIR, "raw", "sample_submission_8RXa3c6.csv")
    )
    sample_submission.to_csv(
        os.path.join(settings.DATA_DIR, "processed", "sample_submission.csv"),
        index=False,
    )
