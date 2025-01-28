import ResultsFuncs as RF
from joblib import Parallel, delayed
import pandas as pd
import os

N_ESTS = 250
N_JOBS = -1

RS = 52637
converted = os.listdir("data/PUBCHEM/converted_data")
converted = [i for i in converted if i[0] != "." and i[0] != "A"]
DATASETS_TO_CHECK = [int(converted[i].split(".")[0]) for i in range(len(converted))]


def dict_to_dataframe(data):
    df = pd.DataFrame()
    # Loop through each model and its corresponding values
    for name, metrics in data.items():
        df[f"{name}-mse"] = [metrics[0]]
        df[f"{name}-r2"] = [metrics[1]]
        df[f"{name}-sig"] = [metrics[2]]
    return df


def main():
    res = Parallel(n_jobs=N_JOBS)(
        delayed(RF.test_dataset)(i, RS, N_ESTS, source="PUBCHEM")
        for i in DATASETS_TO_CHECK
    )
    final_df = pd.DataFrame()
    for res_entry in res:
        id, half1, half2 = res_entry
        df1 = dict_to_dataframe(half1)
        df2 = dict_to_dataframe(half2)
        df1.index = [f"{id} + standard"]
        df2.index = [f"{id} + bayes"]
        final_df = pd.concat([final_df, df1, df2])
    final_df.to_csv("results/PUBCHEM/ResultSummary.csv")


if __name__ == "__main__":
    main()
