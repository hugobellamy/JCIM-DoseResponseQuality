from variable_bootstrap_forest import vbRandomForest
from outputsmearing_forest import osRandomForest
from WeightedSVM import WeightedSVM
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import ttest_rel
import numpy as np
import pickle
import OptimisationFuncs as OF
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings(
    "ignore",
)


def dataset_score(y_preds, dr_test, base_line_errors=None):
    predicted = []
    actual = []
    # get best slope via regression
    slopes = np.ones(len(y_preds))
    for i in range(len(y_preds)):
        doses, responses = zip(*dr_test[i])
        pred_responses = [
            OF.hill_equation(dose, y_preds[i], slopes[i]) for dose in doses
        ]
        predicted.extend(pred_responses)
        actual.extend(responses)

    if base_line_errors is None:
        return (
            root_mean_squared_error(actual, predicted),
            r2_score(actual, predicted),
            [(i - j) ** 2 for i, j in zip(predicted, actual)],
        )
    model_errors = [(i - j) ** 2 for i, j in zip(predicted, actual)]
    sig_score = ttest_rel(base_line_errors, model_errors, alternative="greater").pvalue
    return (
        root_mean_squared_error(actual, predicted),
        r2_score(actual, predicted),
        sig_score,
    )


def VB_result(
    X,
    y,
    y_uncert,
    X_test,
    dr_test,
    params,
    n_estimators=100,
    random_state=None,
    base_line_errors=None,
):
    features = params["max_features"]
    min_samples = params["min_samples_split"]
    powers_boot = params["power_boot"]
    powers_weights = params["power_weight"]
    model = vbRandomForest(
        n_estimators=n_estimators,
        max_features=features,
        min_samples_split=min_samples,
        probs_power=powers_boot,
        weight_power=powers_weights,
        random_state=random_state,
    )
    model.fit(X, y, y_uncert)
    return dataset_score(
        model.predict(X_test), dr_test, base_line_errors=base_line_errors
    )


def OS_result(
    X,
    y,
    y_uncert,
    X_test,
    dr_test,
    params,
    n_estimators=100,
    random_state=None,
    base_line_errors=None,
):
    min_samples = params["min_samples_split"]
    smears = params["smear"]
    powers_smear = params["power_smear"]
    model = osRandomForest(
        n_estimators=n_estimators,
        min_samples_split=min_samples,
        input_smear=smears,
        power_smear=powers_smear,
        random_state=random_state,
    )
    model.fit(X, y, y_uncert)
    return dataset_score(
        model.predict(X_test), dr_test, base_line_errors=base_line_errors
    )  # i want to return mse, r2, significance score


def SVR_result(
    X,
    y,
    y_uncert,
    X_test,
    dr_test,
    params,
    base_line_errors=None,
):
    epsilon = params["epsilon"]
    w_power = params["power"]
    model = WeightedSVM(kernel="rbf", epsilon=epsilon, alpha=w_power)
    model.fit(X, y, y_uncert)
    return dataset_score(
        model.predict(X_test), dr_test, base_line_errors=base_line_errors
    )


def get_full_results(
    optimised_params, X, y, y_uncert, X_test, dr_test, n_ests, random_state, bayes=False
):
    y_uncert = np.array(y_uncert) / np.amax(y_uncert)
    y_uncert = np.array([np.exp(-x) for x in y_uncert])
    results = {}

    base_mse, base_r2, base_line_errors = VB_result(
        X,
        y,
        y_uncert,
        X_test,
        dr_test,
        optimised_params["base"],
        n_ests,
        random_state,
    )
    results["base"] = (base_mse, base_r2, "N/A")
    results["vb"] = VB_result(
        X,
        y,
        y_uncert,
        X_test,
        dr_test,
        optimised_params["vb"],
        n_ests,
        random_state,
        base_line_errors,
    )
    results["w"] = VB_result(
        X,
        y,
        y_uncert,
        X_test,
        dr_test,
        optimised_params["w"],
        n_ests,
        random_state,
        base_line_errors,
    )
    os_mse, os_r2, os_line_errors = OS_result(
        X,
        y,
        y_uncert,
        X_test,
        dr_test,
        optimised_params["os"],
        n_ests,
        random_state,
    )
    results["os"] = (os_mse, os_r2, "N/A")
    results["vos"] = OS_result(
        X,
        y,
        y_uncert,
        X_test,
        dr_test,
        optimised_params["vos"],
        n_ests,
        random_state,
        os_line_errors,
    )
    svr_mse, svr_r2, svr_line_errors = SVR_result(
        X, y, y_uncert, X_test, dr_test, optimised_params["svm"]
    )
    results["svm"] = (svr_mse, svr_r2, "N/A")
    results["wsvm"] = SVR_result(
        X,
        y,
        y_uncert,
        X_test,
        dr_test,
        optimised_params["wsvm"],
        svr_line_errors,
    )

    if bayes:
        y_uncert_alt = [1 / i for i in y_uncert]

        results["vb_alt"] = results["vb"] = VB_result(
            X,
            y,
            y_uncert_alt,
            X_test,
            dr_test,
            optimised_params["vb_alt"],
            n_ests,
            random_state,
            base_line_errors,
        )

        results["w_alt"] = VB_result(
            X,
            y,
            y_uncert_alt,
            X_test,
            dr_test,
            optimised_params["w_alt"],
            n_ests,
            random_state,
            base_line_errors,
        )

        results["svm_alt"] = SVR_result(
            X,
            y,
            y_uncert_alt,
            X_test,
            dr_test,
            optimised_params["svm_alt"],
            svr_line_errors,
        )
    else:
        results["vb_alt"] = ("NA", "NA", "NA")
        results["w_alt"] = ("NA", "NA", "NA")
        results["svm_alt"] = ("NA", "NA", "NA")
    return results


def test_dataset(id, random_state, n_estimators, source="PUBCHEM"):
    all_performance = []

    results = pickle.load(open(f"results/{source}/Optimisation/{id}.pkl", "rb"))
    if source == "PUBCHEM":
        test_size = 0.25
    else:
        test_size = 0.5
    for i in range(2):
        data = OF.load_data(id, i, source=source, test=True)
        X, y, y_uncert, dose_response = data
        X_train, X_test, y_train, _, y_uncert_train, _, _, DR_test = train_test_split(
            X,
            y,
            y_uncert,
            dose_response,
            test_size=test_size,
            random_state=random_state,
        )
        real_results = get_full_results(
            results[i],
            X_train,
            y_train,
            y_uncert_train,
            X_test,
            DR_test,
            n_estimators,
            random_state,
            bayes=i,
        )
        print(f"ID: {id}, half: {i}")
        for model_type in real_results:
            if type(real_results[model_type][1]) is float:
                print(f"{model_type} - {real_results[model_type][1]:.3f}")
            else:
                print(f"{model_type} - {real_results[model_type][1]}")
        all_performance.append(real_results)
    with open(f"results/{source}/Performance/{id}.pkl", "wb") as f:
        pickle.dump(all_performance, f)  # save results
    return [id, all_performance[0], all_performance[1]]
