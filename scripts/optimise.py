import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import simulate

search_space = list()
search_space.append(Real(0.001, 1, name="learning_rate"))
search_space.append(Real(0.001, 1, name="epsilon"))
search_space.append(Real(0.001, 1 , name="phi"))

initial_params = {
    "learning_rate": 0.001,
    "epsilon": 0.4028566874373059,
    "phi": 0.003955266345682853
}


def run_simulation(params):
    return simulate.simulate_erevroth(
        False,
        simulate.one_threshold_change,
        10,
        5,
        5,
        1,
        params["learning_rate"],
        params["epsilon"],
        params["phi"]
    )


@use_named_args(search_space)
def evaluate_model(**params):
    scores = []

    for x in range(5):
        score = run_simulation(params)
        if score is not None:
            scores.append(score)

    return np.median(scores)


initial_params_list = [param for param in initial_params.values()]
result = gp_minimize(
    evaluate_model, search_space, x0=initial_params_list, n_jobs=-1, verbose=True
)

print("Best Accuracy: %.3f" % (1.0 - result.fun))
print("Best Parameters: %s" % (result.x))
