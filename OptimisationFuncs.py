import numpy as np
from variable_bootstrap_forest import vbRandomForest
from outputsmearing_forest import osRandomForest
from WeightedSVM import WeightedSVM
from sklearn.model_selection import train_test_split
import pickle


def hill_equation(dose, neg_log_EC50, slope):
    # Convert negative log EC50 back to EC50 for the calculation
    EC50 = 10 ** (-neg_log_EC50)
    return (100 * (dose**slope)) / (EC50**slope + dose**slope)


def shuffle(X1, X2, X3, X4, random_state=None):
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    if random_state is not None:
        np.random.seed(random_state)
    p = np.random.permutation(len(X1))
    new_X4 = [X4[i] for i in p]
    return X1[p], X2[p], X3[p], new_X4


def dataset_score(y_preds, dr_test, return_N=False):
    total_tests = 0
    total_squared_error = 0
    slopes = np.ones(len(y_preds))
    for i in range(len(y_preds)):
        doses, responses = zip(*dr_test[i])
        pred_responses = [hill_equation(dose, y_preds[i], slopes[i]) for dose in doses]
        total_squared_error += sum(
            [(responses[j] - pred_responses[j]) ** 2 for j in range(len(responses))]
        )
        total_tests += len(responses)
    if return_N:
        return np.array([total_squared_error, total_tests])
    return total_squared_error / total_tests


def cross_val_score_custom(model, X, y, y_uncert, dr, cv=5, random_state=None):
    X, y, y_uncert, dr = shuffle(X, y, y_uncert, dr, random_state=random_state)
    results = np.array([float(0), float(0)])
    indexes = [int((len(X) * i) / cv) for i in range(cv + 1)]
    for i in range(cv):
        X_train = np.concatenate([X[: indexes[i]], X[indexes[i + 1] :]], axis=0)
        y_train = np.concatenate([y[: indexes[i]], y[indexes[i + 1] :]], axis=0)
        y_uncert_train = np.concatenate(
            [y_uncert[: indexes[i]], y_uncert[indexes[i + 1] :]], axis=0
        )
        X_test = X[indexes[i] : indexes[i + 1]]
        dr_test = dr[indexes[i] : indexes[i + 1]]
        model.fit(X_train, y_train, y_uncert_train)
        y_preds = model.predict(X_test)
        results += dataset_score(y_preds, dr_test, return_N=True)
    return results[0] / results[1]


def random_search_vb(
    X,
    y,
    y_uncert,
    dr,
    features,
    max_samples,
    powers_boot,
    powers_weights,
    m_iterations,
    n_estimators=100,
    random_state=0,
    n_jobs=1,
):
    best_score = -10000
    for _ in range(m_iterations):
        max_feature = np.random.choice(features)
        min_sample = np.random.choice(max_samples)
        power_boot = np.random.choice(powers_boot)
        power_weight = np.random.choice(powers_weights)
        rf = vbRandomForest(
            n_estimators=n_estimators,
            max_features=max_feature,
            min_samples_split=min_sample,
            probs_power=power_boot,
            weight_power=power_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        score = cross_val_score_custom(
            rf, X, y, y_uncert, dr, cv=5, random_state=random_state
        )  # differes from sklearn convention so we can normalise over experiments and not datapoints
        if score > best_score:
            best_score = score
            best_params = {
                "max_features": max_feature,
                "min_samples_split": min_sample,
                "power_boot": power_boot,
                "power_weight": power_weight,
            }
    return best_params


def random_search_os(
    X,
    y,
    y_uncert,
    dr,
    max_samples,
    smears,
    powers_smear,
    m_iterations,
    n_estimators=100,
    random_state=0,
    n_jobs=1,
):
    best_score = -10000
    for _ in range(m_iterations):
        min_sample = np.random.choice(max_samples)
        smear = np.random.choice(smears)
        power_smear = np.random.choice(powers_smear)
        rf = osRandomForest(
            n_estimators=n_estimators,
            min_samples_split=min_sample,
            input_smear=smear,
            power_smear=power_smear,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        score = cross_val_score_custom(
            rf, X, y, y_uncert, dr, cv=5, random_state=random_state
        )  # differes from sklearn convention so we can normalise over experiments and not datapoints
        if score > best_score:
            best_score = score
            best_params = {
                "min_samples_split": min_sample,
                "smear": smear,
                "power_smear": power_smear,
            }
    return best_params


def random_search_svm(X, y, y_uncert, dr, epsilons, powers, m_iterations, random_state):
    best_score = -10000
    for _ in range(m_iterations):
        epsilon = np.random.choice(epsilons)
        power = np.random.choice(powers)
        rf = WeightedSVM(
            kernel="rbf",
            epsilon=epsilon,
            alpha=power,
        )
        score = cross_val_score_custom(
            rf, X, y, y_uncert, dr, cv=5, random_state=random_state
        )  # differes from sklearn convention so we can normalise over experiments and not datapoints
        if score > best_score:
            best_score = score
            best_params = {"power": power, "epsilon": epsilon}
    return best_params


def test_dataset(
    X_train,
    y_train,
    y_uncert_train,
    dr,
    test_grid,
    iterations,
    n_estimators=100,
    random_state=0,
    n_jobs=1,
    bayes=False,
):
    min_samples_split_fractions = test_grid["min_samples_split_fractions"]
    max_features = test_grid["max_features"]
    power_values = test_grid["power_values"]
    smear_values = test_grid["smear_values"]
    os_power_values = test_grid["os_power_values"]
    epsilons = test_grid["epsilons"]
    min_samples_split = [
        int(len(X_train) * i)
        for i in min_samples_split_fractions
        if int(len(X_train) * i) > 1
    ]

    y_uncert_train = np.array(y_uncert_train) / np.max(
        y_uncert_train
    )  # normalise to range 0-1, to prevent 0 values and divide by 0 later
    # use kernel on y_uncert_train
    y_uncert_train = np.array([np.exp(-x) for x in y_uncert_train])
    # basecase
    best_params_base = random_search_vb(
        X_train,
        y_train,
        y_uncert_train,
        dr,
        max_features,
        min_samples_split,
        [0],
        [0],
        iterations,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    # variable bootstrap
    best_params_vb = random_search_vb(
        X_train,
        y_train,
        y_uncert_train,
        dr,
        max_features,
        min_samples_split,
        power_values,
        [0],
        iterations,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # weighted
    best_params_w = random_search_vb(
        X_train,
        y_train,
        y_uncert_train,
        dr,
        max_features,
        min_samples_split,
        [0],
        power_values,
        iterations,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    # output smearing
    best_params_os = random_search_os(
        X_train,
        y_train,
        y_uncert_train,
        dr,
        min_samples_split,
        smear_values,
        [0],
        iterations,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    # varaible output smearing
    best_params_vos = random_search_os(
        X_train,
        y_train,
        y_uncert_train,
        dr,
        min_samples_split,
        smear_values,
        os_power_values,
        iterations,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    # SVM
    best_params_svm = random_search_svm(
        X_train, y_train, y_uncert_train, dr, epsilons, [0], iterations, random_state
    )

    best_params_wsvm = random_search_svm(
        X_train,
        y_train,
        y_uncert_train,
        dr,
        epsilons,
        power_values,
        iterations,
        random_state,
    )

    if bayes:
        y_uncert_train_alt = [
            1 / i for i in y_uncert_train
        ]  # it is this because base case is variance
        best_params_vb_alt = best_params_vb = random_search_vb(
            X_train,
            y_train,
            y_uncert_train_alt,
            dr,
            max_features,
            min_samples_split,
            [1],
            [0],
            iterations,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # weighted
        best_params_w_alt = random_search_vb(
            X_train,
            y_train,
            y_uncert_train,
            dr,
            max_features,
            min_samples_split,
            [0],
            [1],
            iterations,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # svm
        best_params_svm_alt = random_search_svm(
            X_train,
            y_train,
            y_uncert_train_alt,
            dr,
            epsilons,
            [1],
            iterations,
            random_state,
        )
    else:
        best_params_vb_alt = None
        best_params_w_alt = None
        best_params_svm_alt = None

    return {
        "base": (best_params_base),
        "vb": (best_params_vb),
        "w": (best_params_w),
        "os": (best_params_os),
        "vos": (best_params_vos),
        "vb_alt": (best_params_vb_alt),
        "w_alt": (best_params_w_alt),
        "svm": (best_params_svm),
        "wsvm": (best_params_wsvm),
        "svm_alt": (best_params_svm_alt),
    }


def load_data(dataset_id, half=0, source="PUBCHEM", test=False):
    data = pickle.load(
        open(f"data/{source}/converted_data/" + str(dataset_id) + ".pkl", "rb")
    )
    sub_data = data[half]
    if test:
        dose_response = data[2]  # 3 is for training, 2 is for testing
    else:
        dose_response = data[3]
    X, y, y_uncert = sub_data
    return X, y, y_uncert, dose_response


def test_both_halves(
    dataset_id, params, n_estimators, random_state, cores_per_run, source="PUBCHEM"
):
    both_results = []

    if source == "PUBCHEM":
        test_size = 0.25
    else:
        test_size = 0.5
    print("Testing dataset:", dataset_id)
    for half in [0, 1]:
        X, y, y_uncert, dose_response = load_data(dataset_id, half, source)
        X_train, _, y_train, _, y_uncert_train, _, dose_response_train, _ = (
            train_test_split(
                X,
                y,
                y_uncert,
                dose_response,
                test_size=test_size,
                random_state=random_state,
            )
        )
        opt_res = test_dataset(
            X_train,
            y_train,
            y_uncert_train,
            dose_response_train,
            params,
            10,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=cores_per_run,
            bayes=half,
        )
        both_results.append(opt_res)
    with open(f"results/{source}/Optimisation/{dataset_id}.pkl", "wb") as f:
        pickle.dump(both_results, f)
