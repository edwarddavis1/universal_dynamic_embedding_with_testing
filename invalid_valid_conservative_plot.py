# %%
from tqdm import tqdm
from experiment_setup import *
from embedding_functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
methods = ["UASE", "ISE", "ISE Procrustes"]
n = 200
exp = "moving-static-community"
move_prob = 0.53
T = 2
T_for_exp = T

B = get_B_for_exp(exp, T_for_exp, move_prob=move_prob)

embeddings_dict = {}
tau_dict = {}
p_vals_dict = {method: [] for method in methods}
for method in methods:
    for _ in tqdm(range(200)):
        As, tau, clust_to_check, changepoint = make_experiment(
            exp, n, T_for_exp, move_prob=move_prob
        )
        tau_dict[method] = tau

        # Embed at the rank of the embedding matrix for each method
        d = get_embedding_dimension(B, method)

        YA_embedding = embed(
            As,
            d,
            method,
            q=1,
            window=3,
            walklen=15,
            num_walks=20,
        )
        embeddings_dict[method] = YA_embedding

        # Test over time
        node_set = np.where(tau == 0)
        time_set_1 = np.arange(n)
        time_set_2 = time_set_1 + n
        ya1 = YA_embedding[time_set_1, :][node_set]
        ya2 = YA_embedding[time_set_2, :][node_set]

        p_hat = test_temporal_displacement_two_times(
            np.row_stack([ya1, ya2]), n=ya1.shape[0], n_sim=1000
        )
        p_vals_dict[method].append(p_hat)

# %%
BLUE = (0.69, 0.8, 1)
GREEN = (0.84, 1, 0.89)
RED = (1, 0.70, 0.70)

colours = ["#41b6c4", "#CA054D"]
cmap = mpl.colors.ListedColormap(colours)

titles = ["Valid (UASE)", "Invalid (ISE)", "Conservative (ISE Procrustes)"]
roc_colours = [GREEN, RED, BLUE]

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

for i, method in enumerate(methods):

    node_set = np.where(tau_dict[method] == 0)

    axs[0, i].scatter(
        embeddings_dict[method][time_set_1, :][node_set][:, 0],
        embeddings_dict[method][time_set_1, :][node_set][:, 1],
        c=colours[0],
    )
    axs[0, i].scatter(
        embeddings_dict[method][time_set_2, :][node_set][:, 0],
        embeddings_dict[method][time_set_2, :][node_set][:, 1],
        c=colours[1],
    )
    axs[0, i].set_title(titles[i])

    axs[0, i].grid(alpha=0.2)

    roc = []
    alphas = []
    for alpha in np.linspace(0, 1, 200):
        alphas.append(alpha)
        num_below_alpha = sum(p_vals_dict[method] < alpha)
        roc_point = num_below_alpha / 200
        roc.append(roc_point)

    axs[1, i].plot(alphas, roc, linewidth=5)
    axs[1, i].plot(
        np.linspace(0, 1, 2),
        np.linspace(0, 1, 2),
        linestyle="--",
        c="grey",
        linewidth=5,
    )

    # Colour the background by the roc colour
    axs[1, i].set_facecolor(roc_colours[i])

    axs[1, i].set_xticks([])
    axs[1, i].set_yticks([])


plt.tight_layout()
plt.show()
# %%
