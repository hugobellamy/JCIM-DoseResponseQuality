from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import curve_fit
import random


def get_extreme_concs(all_info):
    doses = [list(zip(*dose_response))[0] for dose_response in all_info]
    doses = [item for sublist in doses for item in sublist]
    return min(doses), max(doses)


def hill_equation(dose, neg_log_EC50, slope=1):
    # Convert negative log EC50 back to EC50 for the calculation
    EC50 = 10 ** (-neg_log_EC50)
    return (100 * (dose**slope)) / (EC50**slope + dose**slope)


def fit_hill_equation(data, bounds=[0.01, 25000], slope_bounds=None):
    doses, responses = zip(*data)
    doses = np.array(doses)
    responses = np.array(responses)
    if slope_bounds is None:
        real_bounds = [(-np.log10(bounds[1])), (-np.log10(bounds[0]))]
        initial_guesses = [-np.log10(np.median(doses))]
        params, _ = curve_fit(
            hill_equation,
            doses,
            responses,
            p0=initial_guesses,
            bounds=real_bounds,
            maxfev=10000,
        )
        neg_log_EC50 = params[0]
        slope = 1
    else:
        real_bounds = [
            (-np.log10(bounds[1]), slope_bounds[0]),
            (-np.log10(bounds[0]), slope_bounds[1]),
        ]
        initial_guesses = [-np.log10(np.median(doses)), 1]
        params, _ = curve_fit(
            hill_equation,
            doses,
            responses,
            p0=initial_guesses,
            bounds=real_bounds,
            maxfev=10000,
        )
        neg_log_EC50, slope = params
    return neg_log_EC50, slope


def get_ec50s(DR, X, bounds=[0.01, 25000]):
    res = []
    for j, i in enumerate(DR):
        doses, _ = zip(*i)
        try:
            assert len(set(doses)) > 1
            log_ec50, slope = fit_hill_equation(i, bounds, slope_bounds=[0.1, 10])
        except (AssertionError, RuntimeError):
            log_ec50, slope = fit_hill_equation(i, bounds)
        res.append((X[j], log_ec50, slope, i))
    new_X, y, slopes, DR_out = zip(*res)
    return new_X, y, slopes, DR_out


def mean_curve_error(dose_response, nlec50, slope):
    squared_errors = [
        (hill_equation(dose, nlec50, slope) - response) ** 2
        for dose, response in dose_response
    ]
    return np.mean(squared_errors)


def get_errors_basic(DR, ec50s, slopes):
    errors = []
    for i, j in enumerate(DR):
        errors.append(mean_curve_error(j, ec50s[i], slopes[i]))
    return errors


def estimate_noise(all_dr):
    # need to take each dr and find the repeat concs, than get a pooled variance estiamate from these
    total_variance = 0
    total_length = 0
    for dr in all_dr:
        doses, responses = zip(*dr)
        unique_doses = set(doses)
        for dose in unique_doses:
            dose_responses = [
                responses[i]
                for i in range(len(doses))
                if (doses[i] == dose and responses[i] > 0 and responses[i] < 100)
            ]
            if len(dose_responses) >= 2:
                total_variance += (len(dose_responses)) * np.var(dose_responses)
                total_length += len(dose_responses) - 1
    if total_length == 0:
        return np.nan
    return total_variance / total_length


def post_val(dose_response, hill_value, ec50, sigma):
    Q = mean_curve_error(dose_response, -np.log10(ec50), hill_value) * len(
        dose_response
    )
    return -Q / (2 * sigma)


def log_average(log_vals):
    log_vals_max = np.max(log_vals)
    log_vals = log_vals - log_vals_max

    return log_vals_max + np.log(np.mean(np.exp(log_vals)))


def get_post_dist(
    dose_response, hill_value_range, ec50_range, sigma, n_points=100, withDraws=False
):
    ec50_vals = np.linspace(ec50_range[0], ec50_range[1], n_points)  # uniform prior
    hill_vals = np.linspace(
        hill_value_range[0], hill_value_range[1], int(n_points / 10)
    )
    log_post_weights = [
        log_average(
            [
                post_val(dose_response, hill_value, ec50, sigma)
                for hill_value in hill_vals
            ]
        )
        for ec50 in ec50_vals
    ]

    log_post_weights = log_post_weights - np.max(log_post_weights)
    post_weights = np.exp(log_post_weights)
    # replace nan with 0
    post_weights = np.nan_to_num(post_weights)
    post_draws = np.random.choice(
        ec50_vals, p=post_weights / np.sum(post_weights), size=10 * n_points
    )
    post_draws = -np.log10(post_draws)
    if withDraws:
        return np.mean(post_draws), np.var(post_draws), post_draws
    return np.mean(post_draws), np.var(post_draws)


def reduce_data(data):  # on half the datapoint delete half the measurmenets
    if len(data) < 2:
        return data
    reduce = np.random.choice([True, False], p=[0.9, 0.1])
    if reduce:
        ids_to_keep = np.random.choice(
            [i for i in range(len(data))],
            min(3, len(data)),
            replace=False,
        )
        return [data[i] for i in ids_to_keep]
    return data


def make_X_y(
    X, dose_response, dose_factor=10, random_seed=None, reduce=False, base_noise=100
):
    # shuffle X and dose_response to remove any ordering
    input_numbers = [i for i in range(len(X))]
    new_order = np.random.RandomState(random_seed).permutation(input_numbers)
    X = [X[i] for i in new_order]
    dose_response = [dose_response[i] for i in new_order]

    if reduce:
        use_dose_response = [reduce_data(i) for i in dose_response]
    else:
        use_dose_response = dose_response

    min_dose, max_dose = get_extreme_concs(dose_response)

    # standard data
    new_X, new_y, hill_slopes, dr_out = get_ec50s(
        dose_response, X, bounds=[min_dose / dose_factor, max_dose * dose_factor]
    )
    hill_slopes = np.nan_to_num(hill_slopes)
    y_uncert = get_errors_basic(dr_out, new_y, hill_slopes)
    basic_data = [new_X, new_y, y_uncert]
    # bayesian data
    sigma_estimate = estimate_noise(dr_out)
    if sigma_estimate == 0 or np.isnan(sigma_estimate):
        sigma_estimate = (
            base_noise  # this is mean of datasets we do get a noise est for
        )
        print(f"failed to estimate noise assuming {base_noise}")
    bayes_dists = [
        get_post_dist(
            dose_response[i],
            [0.1, 10],
            [min_dose / 10, max_dose * 10],
            sigma_estimate,
            500,
        )
        for i in range(len(dose_response))
    ]
    new_y2, y_uncert2 = zip(*bayes_dists)
    bayes_data = [X, new_y2, y_uncert2]

    return [basic_data, bayes_data, dose_response, use_dose_response, sigma_estimate]
