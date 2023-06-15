# %%
"""Originally cloned from https://github.com/iggallagher/Miscellaneous"""
import airportsdata
import pycountry
from matplotlib.patches import Patch
import matplotlib
from sklearn.manifold import MDS
from math import comb
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import time
import numpy as np
import csv
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from embedding_functions import *
# %%


def plot_diss_mat(mat_to_show, title=None, savename=None):
    """
    Plots a dissimilarity matrix with labels for each month
    """
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    labels = np.core.defchararray.add(np.array(months * 3),
                                      np.repeat([" 2019", " 2020", " 2021"], 12))

    # Label every other month for readability
    tick_locations = np.arange(36)[0::2]
    tick_labels = labels[0::2]

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    cax = ax.matshow(mat_to_show)
    _ = ax.set_xticks(tick_locations, tick_labels, rotation=90)
    _ = ax.set_yticks(tick_locations, tick_labels, rotation=0)
    fig.colorbar(cax)
    if title is not None:
        ax.set_title(title)
    plt.grid()

    if savename:
        # Set white background
        fig.patch.set_facecolor('white')
        plt.savefig(savename + '.png', bbox_inches='tight')


# %%
datapath = "../../Datasets/Flight Data/"


# %%
# Load adjacency matrices
As = []
for t in range(36):
    As.append(sparse.load_npz(datapath + 'As_' + str(t) + '.npz'))

As = np.array(As)

n = As[0].shape[0]
T = len(As)

# Load labels
Z = np.load(datapath + 'Z.npy')
nodes = np.load(datapath + 'nodes.npy', allow_pickle=True).item()

# %%
# Compute embedding
d = 5
method = "URLSE"
YAs = regularised_ULSE(As, d, sparse_matrix=True,
                       regulariser='auto', flat=False, verbose=True)
# %%
# Select to look at European airports only
cont_to_look_at = "EU"
continent_nodes = np.where(np.array(Z) == cont_to_look_at)
n_continent = len(continent_nodes[0])
# %%
# Compute dissimilarity or p-value matrix from the embedding


def compute_diss_matrix(YAs, Z, cont_to_look_at, mat_type="dissimilarity"):
    continent_nodes = np.where(np.array(Z) == cont_to_look_at)
    n_continent = len(continent_nodes[0])

    if mat_type == "dissimilarity":
        diss_matrix = np.zeros((T, T))
    else:
        diss_matrix = np.ones((T, T))

    num_p_val_comps = 0
    num_p_vals_to_compute = int(T*(T-1)/2)
    for i in range(diss_matrix.shape[0]):
        for j in range(i+1, diss_matrix.shape[1]):
            num_p_val_comps += 1

            ya_i_euro = YAs[i][continent_nodes]
            ya_j_euro = YAs[j][continent_nodes]

            if mat_type == "dissimilarity":
                p_val = vector_displacement_test(
                    ya_i_euro, ya_j_euro)

                diss_matrix[i, j] = p_val / n_continent
                diss_matrix[j, i] = p_val / n_continent

            else:
                ya_temp = np.concatenate((ya_i_euro, ya_j_euro))
                p_val = test_temporal_displacement_two_times(
                    ya_temp, n=ya_i_euro.shape[0])

                diss_matrix[i, j] = p_val
                diss_matrix[j, i] = p_val

    return diss_matrix


mat_type = "dissimilarity"
# mat_type = "p_value"

# Compute dissimilarity or p-value matrix for each continent
for cont_to_look_at in ["EU", "NA", "AS", "SA", "OC", "AF"]:
    diss_matrix = compute_diss_matrix(
        YAs, Z, cont_to_look_at, mat_type=mat_type)

    if mat_type == "dissimilarity":
        np.save("saved_flight_matrices/{}_{}_matrix".format(
            method, cont_to_look_at), diss_matrix)

        plot_diss_mat(diss_matrix, savename="saved_flight_matrices/{}_{}_matrix".format(
            method, cont_to_look_at))
    else:
        np.save("saved_flight_matrices/{}_{}_p_value_matrix".format(
            method, cont_to_look_at), diss_matrix)
        plot_diss_mat(diss_matrix > 0.05/comb(T, 2))

# %%
# Hierarchical clustering on European airports
cont_to_look_at = "EU"
diss_matrix = compute_diss_matrix(
    YAs, Z, cont_to_look_at="EU", mat_type="dissimilarity")


def plot_dendrogram(model, labels, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs, labels=labels,
               leaf_rotation=90, leaf_font_size=18)


# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(
    distance_threshold=0, compute_full_tree=True, n_clusters=None,
)
model = model.fit(diss_matrix)

plt.rcParams.update({'font.size': 14})
plt.subplots(figsize=(17, 20))
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))
plot_dendrogram(model, truncate_mode="level", p=15,
                labels=labels, orientation="left")
plt.yticks(rotation=0)

# Add red labels for outliers
y_axis = plt.gca().yaxis
for label in y_axis.get_ticklabels():
    if label.get_text() in ['June 2021', 'August 2020', "September 2020"]:
        label.set_color('red')

# Annotations
# First split
plt.annotate("Not Fully-Affected by COVID-19", xy=(0.5, 0.5), xytext=(0.3, 0.5),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             horizontalalignment="center", verticalalignment="center",
             fontsize=20)
plt.annotate("Fully-Affected by COVID-19", xy=(0.5, 0.1), xytext=(0.3, 0.1),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             horizontalalignment="center", verticalalignment="center",
             fontsize=20)

# Second split green
plt.annotate("Unaffected Summer", xy=(0.945, 0.335), xytext=(0.75, 0.335),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             horizontalalignment="center", verticalalignment="center",
             fontsize=20)

# Third split green
plt.annotate("Partially Affected", xy=(0.945, 0.555), xytext=(0.84, 0.555),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             horizontalalignment="center", verticalalignment="center",
             fontsize=20)

# Fourth split green
plt.annotate("Unaffected Non-Summer", xy=(0.86, 0.76), xytext=(0.7, 0.76),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             horizontalalignment="center", verticalalignment="center",
             fontsize=20)

# Second split orange
plt.annotate("COVID-19 Start", xy=(0.88, 0.04), xytext=(0.765, 0.04),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             horizontalalignment="center", verticalalignment="center",
             fontsize=20)
plt.annotate("Other COVID-19", xy=(0.88, 0.15), xytext=(0.765, 0.15),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             horizontalalignment="center", verticalalignment="center",
             fontsize=20)


plt.tight_layout()
plt.show()

# %%
# Plot the dissimilarity matrix for each continent

# change text size
plt.rcParams.update({'font.size': 14})

continents = ["EU", "AS", "NA", "SA", "AF", "OC"]
conts_title = ["Europe", "Asia", "North America",
               "South America", "Africa", "Oceania"]
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))

mats = [compute_diss_matrix(YAs, Z, cont_to_look_at=continents[i])
        for i in range(len(conts_title))]

# plot matrices
fig, axs = plt.subplots(3, 2, figsize=(20, 22))
for i in range(len(conts_title)):
    ax = axs[i % 3, i//3]
    mat_to_show = mats[i]
    ax.set_title(conts_title[i])

    tick_locations = np.arange(36)[0::2]
    tick_labels = labels[0::2]

    cax = ax.matshow(mat_to_show)
    _ = ax.set_xticks(tick_locations, tick_labels, rotation=90)
    _ = ax.set_yticks(tick_locations, tick_labels, rotation=0)
    _ = ax.xaxis.set_ticks_position('bottom')

# Colourbar for all plots
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.80, 0.15, 0.03, 0.7])
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.set_label("Distance")


# # Increase the distance between subplot rows
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.0)

# Add white background
fig.patch.set_facecolor('white')

# Save plot
save_dir = "saved_flight_matrices"
plt.savefig(
    save_dir + "/all_dissimilarity_matrices_d={}_mutligraph.pdf".format(d))

# %%
# Compare individual nodes

# airport_of_interest = "LIML"  # milan
airport_of_interest = "EGLL"  # London Heathrow
# airport_of_interest = "VHHH"  # Hong kong

airport_idx = list(nodes.keys()).index(airport_of_interest)
airport_vecs = YAs[:, airport_idx, :]

dists = np.zeros((T, T))
for i in range(T):
    for j in range(i+1, T):

        dist = np.linalg.norm(airport_vecs[i, :] - airport_vecs[j, :])
        dists[i, j] = dist
        dists[j, i] = dist

plot_diss_mat(dists)

# plt.savefig("saved_flight_matrices/dissimilarity_matrix_{}.pdf".format(
#     airport_of_interest), dpi=300, bbox_inches="tight")

# %%
