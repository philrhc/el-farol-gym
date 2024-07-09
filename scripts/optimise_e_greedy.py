import numpy as np
import orjson
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import simulate

search_space = list()
search_space.append(Real(0, 1, name="learning_rate"))
search_space.append(Real(0, 1, name="recency_rate"))
search_space.append(Real(0, 0.5, name="initial_epsilon"))
search_space.append(Real(0, 0.1, name="epsilon_decay"))
search_space.append(Real(0, 0.2, name="final_epsilon"))

initial_params = {
    "learning_rate": 0.9,
    "recency_rate": 0.8,
    "initial_epsilon": 0.5,
    "epsilon_decay": 0.01,
    "final_epsilon": 0.01
}


def run_simulation(params):
    return simulate.simulate_egreedy(
        False,
        simulate.random_threshold_changes,
        10,
        5,
        5,
        1,
        params["learning_rate"],
        params["recency_rate"],
        params["initial_epsilon"],
        params["epsilon_decay"],
        params["final_epsilon"]
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
