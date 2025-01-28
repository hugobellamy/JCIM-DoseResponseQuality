import OptimisationFuncs as OF
from joblib import Parallel, delayed
import os

RS = 231

N_RUNS = 9
CORES_PER_RUN = 4
N_ESTS = 250

converted = os.listdir("data/PUBCHEM/converted_data")
converted = [i for i in converted if i[0] != "." and i[0] != "A"]
DATASETS_TO_TEST = [int(converted[i].split(".")[0]) for i in range(len(converted))]

max_features = [0.1, 0.33, 0.5, 1.0]
min_samples_split_fractions = [0.01, 0.025, 0.05, 0.1]
power_values = [0, 0.33, 1, 2]
os_power_values = [-i for i in power_values]
smear_values = [0.25, 1, 1.5, 2.5]
epsilons = [0.05, 0.1, 0.2, 0.5]


params = {
    "min_samples_split_fractions": min_samples_split_fractions,
    "max_features": max_features,
    "power_values": power_values,
    "smear_values": smear_values,
    "os_power_values": os_power_values,
    "epsilons": epsilons,
}


def main():
    Parallel(n_jobs=N_RUNS)(
        delayed(OF.test_both_halves)(
            i, params, N_ESTS, RS, CORES_PER_RUN, source="PUBCHEM"
        )
        for i in DATASETS_TO_TEST
    )


if __name__ == "__main__":
    main()
