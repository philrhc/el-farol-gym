import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import simulation
from agent.e_greedy import EGreedyAgent
from scripts import capacity_changes, hyperparams

erev_roth_search_space = list()
erev_roth_search_space.append(Real(0.001, 1, name="learning_rate"))
erev_roth_search_space.append(Real(0.001, 1, name="epsilon"))
erev_roth_search_space.append(Real(0.001, 1, name="phi"))

egreedy_search_space = list()
egreedy_search_space.append(Real(0, 1, name="learning_rate"))
egreedy_search_space.append(Real(0, 1, name="retention_rate"))
egreedy_search_space.append(Real(0, 0.5, name="initial_epsilon"))
egreedy_search_space.append(Real(0, 0.01, name="epsilon_decay"))
egreedy_search_space.append(Real(0, 0.33, name="final_epsilon"))


def e_greedy_config(params):
    return {"learning_rate": params["learning_rate"],  # Reward multiplier
            "retention_rate": params["retention_rate"],  # Forget some past experience
            "initial_epsilon": params["initial_epsilon"],  # Exploration probability
            "epsilon_decay": params["initial_epsilon"],  # Exploration reduction over time
            "final_epsilon": params["initial_epsilon"]  # Final exploration probability
            }


def run_simulation(params):
    random_changes = capacity_changes.RandomChanges(chance=0.2, limit=0.2)
    return simulation.simulate(visualise=False,
                               agent_type=EGreedyAgent,
                               config=e_greedy_config(params),
                               init_capacities=[70],
                               capacity_change_functions=[random_changes.func])


@use_named_args(egreedy_search_space)
def evaluate_model(**params):
    scores = []

    for x in range(10):
        score = run_simulation(params)
        if score is not None:
            scores.append(score)

    return np.median(scores)


def run():
    initial_params_list = [param for param in hyperparams.e_greedy_optimal.values()]
    result = gp_minimize(
        evaluate_model, egreedy_search_space, x0=initial_params_list, n_jobs=-1, verbose=True
    )

    print("Best Accuracy: %.3f" % (1.0 - result.fun))
    print("Best Parameters: %s" % (result.x))


if __name__ == '__main__':
    run()
