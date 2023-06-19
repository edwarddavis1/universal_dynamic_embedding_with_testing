# %%
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from embedding_functions import *
import gc
from tqdm import tqdm
from experiment_setup import *

# %%

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--methods",
    nargs="+",
    default=["URLSE"],
    help="List of dynamic embedding methods to run. Select from: URLSE, UASE, OMNI, ISE, ISE Procrustes, GloDyNE, Unfolded Node2Vec, Independent Node2Vec",
)
parser.add_argument(
    "--experiments",
    nargs="+",
    default=[
        "moving-static-community",
    ],
    help="Select which dynamic network systems to test on. Select from: static, moving-static-community, moving-community, merge, static-spatial, power-moving, power-static",
)
parser.add_argument(
    "--check-types",
    nargs="+",
    default=["community"],
    help="Run experiments at the community, graph, or node level",
)
parser.add_argument(
    "--n-runs",
    type=int,
    default=200,
    help="Number of p-values to compute for each test",
)
parser.add_argument(
    "--n", type=int, default=200, help="Number of nodes in each (non-power) experiment"
)
parser.add_argument(
    "--n-power",
    type=int,
    default=2000,
    help="Number of nodes in each power experiment (reccomend at least 2000)",
)
parser.add_argument(
    "--show-plots",
    type=bool,
    default=True,
    help="Whether to show the plots or not",
)
parser.add_argument(
    "--all",
    type=bool,
    default=False,
    help="Runs all experiments with all methods at all levels.",
)


args = parser.parse_args()

# Create save directory if it doesn't exist
save_file = True  # Save dataframe of p-values for each method on each experiment
save_dir_dfs = "saved_experiment_dataframes/"  # Directory to save experiment runs in
if not os.path.exists(save_dir_dfs):
    os.makedirs(save_dir_dfs)

save_dir_plots = "saved_experiment_plots/"  # Directory to save plots in
if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

show_plots = args.show_plots
run_all = args.all

# Dynamic embedding methods to run
methods = args.methods

# Experiments to run
experiments_to_run = args.experiments

# Run tests at graph, community or node level
check_type_list = args.check_types

if run_all:
    methods = [
        "URLSE",
        "UASE",
        "OMNI",
        "ISE",
        "ISE Procrustes",
        "GloDyNE",
        "Unfolded Node2Vec",
        "Independent Node2Vec",
    ]
    experiments_to_run = [
        "static",
        "moving-static-community",
        "moving-community",
        "merge",
        "static-spatial",
        "power-moving",
        "power-static",
    ]
    check_type_list = ["community", "graph", "node"]

# Generated network parameters
T = 2  # Number of time points
n_normal = args.n  # Number of nodes in each (non-power) experiment
n_power = args.n_power  # Number of nodes in each power experiment

# If the generated network is moving, control how much it moves
move_prob = 0.53  # For moving system. Prob is initially 0.5
power_move_prob = 0.97  # For power-moving system. Prob is initially 1

n_runs = args.n_runs  # Number of p-values to compute for p-value distribution


# Generates a set of p-values for each method on each experiment and saves them as a dataframe
for check_run_num, check_type in enumerate(check_type_list):
    # If testing at the node level, increase the number of time points
    if check_type == "node":
        T_for_exp = 50
    else:
        T_for_exp = T

    for exp_run_num, exp in enumerate(experiments_to_run):
        # If testing on power-distributed examples, we require more nodes
        if "power" in exp:
            n = n_power
            regulariser = "auto"  # For URLSE. This setting works well on power-distributed examples
        else:
            n = n_normal
            regulariser = 10

        # Select the embedding dimension for each method to be the rank of the embedding matrix
        dim_for_embedding_dict = {}
        for method in methods:
            # Get SBM matrix for experiment
            B = get_B_for_exp(
                exp, T_for_exp, move_prob=move_prob, power_move_prob=power_move_prob
            )

            # Get the rank of the embedding matrix for each method
            d_for_method = get_embedding_dimension(B, method)
            dim_for_embedding_dict[method] = d_for_method

        p_hat_list = []
        print("Experiment: {}\nCheck Type: {}\n".format(exp, check_type))
        for current_run in tqdm(range(n_runs)):
            # Generate the selected system
            As, tau, clust_to_check, changepoint = make_experiment(
                exp, n, T_for_exp, move_prob=move_prob
            )

            # Calculate all embeddings on the system
            embeddings_dict = {}
            for method in methods:
                d = dim_for_embedding_dict[method]
                YA_embedding = embed(
                    As,
                    d,
                    method,
                    q=1,
                    regulariser=regulariser,
                    window=3,
                    walklen=15,
                    num_walks=20,
                )
                embeddings_dict[method] = YA_embedding

            # Select time and node sets for temporal tests
            if exp not in ["merge", "static-spatial"]:
                # Node sets
                if check_type == "community":
                    node_set_1 = np.where(np.tile(tau, changepoint) == clust_to_check)
                elif check_type == "node":
                    node_to_check = np.where(tau == clust_to_check)[0][0]
                    node_set_1 = np.where(
                        np.tile(np.arange(0, n), changepoint) == node_to_check
                    )
                elif check_type in ["graph"]:
                    # Selects all idx available once the time sets have been applied
                    node_set_1 = np.arange(0, n * changepoint)
                else:
                    raise ValueError("Check type not recognised")

                node_set_2 = node_set_1

                # Time sets
                time_set_1 = np.arange(0, changepoint * n)
                time_set_2 = np.arange(n * changepoint, n * T_for_exp)

            # Select time and node sets for spatial tests
            else:
                # Node sets
                if check_type == "community":
                    node_set_1 = np.where(tau == 0)
                    node_set_2 = np.where(tau == 1)
                elif check_type == "node":
                    node_1_to_check = np.where(tau == 0)[0][0]
                    node_set_1 = np.where(
                        np.tile(np.arange(0, n), changepoint) == node_1_to_check
                    )[0]
                    node_2_to_check = np.where(tau == 1)[0][0]
                    node_set_2 = np.where(
                        np.tile(np.arange(0, n), T_for_exp - changepoint)
                        == node_2_to_check
                    )[0]
                elif check_type == "graph":
                    print("Cannot compute spatial tests at graph level... skipping")
                    break
                else:
                    raise ValueError("Check type not recognised")

                # Time sets
                time_set_2 = np.arange(n * changepoint, n * T_for_exp)
                time_set_1 = time_set_2

            # Compute p-values
            for method in methods:
                # Select embedding sets
                YA_embedding = embeddings_dict[method]
                ya1 = YA_embedding[time_set_1, :][node_set_1]
                ya2 = YA_embedding[time_set_2, :][node_set_2]

                # Paired displacement testing
                p_hat = test_temporal_displacement_two_times(
                    np.row_stack([ya1, ya2]), n=ya1.shape[0]
                )

                p_hat_list.append(p_hat)

            # Free up memory
            del YA_embedding
            del embeddings_dict
            gc.collect()

        df = pd.DataFrame()
        df["p_hat"] = p_hat_list
        df["method"] = methods * int(len(p_hat_list) / len(methods))

        if save_file:
            for method in methods:
                df_to_save = df[df["method"] == method]
                df_to_save.to_csv(
                    save_dir_dfs + exp + "_" + check_type + "_" + str(method) + ".csv"
                )


# %%
BLUE = (0.69, 0.8, 1)
GREEN = (0.84, 1, 0.89)
RED = (1, 0.70, 0.70)
GREY = (0.8, 0.8, 0.8)

colour_background = True
save_figs = True

methods_from_save = []
methods_ordering = [
    "ISE",
    "ISE Procrustes",
    "OMNI",
    "UASE",
    "ULSE",
    "URLSE",
    "Independent Node2Vec",
    "Dynamic Skip Gram",
    "Unfolded Node2Vec",
    "GloDyNE",
]

for experiment_to_plot in experiments_to_run:
    for experiment_check_type in check_type_list:
        # Get the saved dataframes from save folder
        df_list = []
        for dirpath, dirnames, filenames in os.walk(save_dir_dfs):
            for df_file in filenames:
                file_check_type = df_file.split("_")[1]
                file_system = df_file.split("_")[0]

                if (
                    file_system == experiment_to_plot in df_file
                    and file_check_type == experiment_check_type
                ):
                    df_list.append(pd.read_csv(save_dir_dfs + df_file))
                    methods_from_save.append(df_file.split("_")[-1].split(".")[0])

        # Plot p-value cumulative distribution
        print(
            "{} network at {} level...".format(
                experiment_to_plot, experiment_check_type
            )
        )
        for i, method in enumerate(methods):
            dfind = df_list[methods_from_save.index(methods[i])]

            roc = []
            alphas = []
            for alpha in np.linspace(0, 1, len(dfind)):
                alphas.append(alpha)
                num_below_alpha = sum(dfind["p_hat"].values < alpha)
                roc_point = num_below_alpha / len(dfind)
                roc.append(roc_point)

            # Get the power at the 5% significance level
            power_significance = 0.05
            power_idx = alphas.index(
                min(alphas, key=lambda x: abs(x - power_significance))
            )
            power = roc[power_idx]
            print("{} had power {}".format(method, power))

            # Colour the plot based on if the power is expected for each experiment
            colour_for_plot = None
            if (
                experiment_to_plot
                in ["moving-community", "static-spatial", "power-moving", "move_power"]
                or experiment_to_plot == "moving-static-community"
                and experiment_check_type == "graph"
            ):
                correct_distribution = "alternative"
            else:
                correct_distribution = "uniform"

            power_threshold = 0.04

            if power >= 0.05 - power_threshold and power <= 0.05 + power_threshold:
                if correct_distribution == "uniform":
                    # if the distribution is approximately uniform when it should be uniform
                    colour_for_plot = GREEN
                else:
                    # if the distribution is approximately uniform when it should be super-uniform
                    colour_for_plot = GREY
            else:
                # If not uniform, decide whether alternative or conservative
                if roc[power_idx] > 0.05:
                    # alternative
                    if correct_distribution == "uniform":
                        colour_for_plot = RED
                    else:
                        colour_for_plot = GREEN

                else:
                    # conservative
                    if correct_distribution == "uniform":
                        colour_for_plot = BLUE
                    else:
                        colour_for_plot = RED

            # Plot the distribution
            fig = plt.figure(figsize=(3, 3))
            plt.plot(
                np.linspace(0, 1, 2),
                np.linspace(0, 1, 2),
                linestyle="--",
                c="grey",
                linewidth=5,
            )
            plt.plot(alphas, roc, linewidth=5)

            if colour_background:
                fig.patch.set_facecolor(colour_for_plot)
            else:
                fig.patch.set_alpha(0.0)

            plt.xticks([])
            plt.yticks([])
            plt.axis("off")

            if show_plots:
                plt.show()

            if save_figs:
                plt.savefig(
                    save_dir_plots
                    + experiment_to_plot
                    + "_"
                    + experiment_check_type
                    + "_"
                    + method
                    + ".png",
                    bbox_inches="tight",
                )

# %%
