# %%
"""Originally cloned from https://github.com/iggallagher/Miscellaneous"""
from scipy.special import softmax
import numba as nb
from scipy.sparse import coo_matrix, find, csc_matrix
# from scipy.sparse import coo_matrix
import scipy.sparse as sp
import airportsdata
import pycountry
from matplotlib.patches import Patch
from sklearn.manifold import Isomap
from matplotlib.animation import FuncAnimation
import matplotlib
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from math import comb
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import time
import csv
import glob
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ian's embedding functions
# import spectral_embedding as se

# Ed's embedding functions
from embedding_functions import *

# %%


def plot_diss_mat(mat_to_show, title=None, savename=None):
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    labels = np.core.defchararray.add(np.array(months * 3),
                                      np.repeat([" 2019", " 2020", " 2021"], 12))

    # remove september 2020
    # mat_to_show = np.delete(mat_to_show, 20, axis=0)
    # mat_to_show = np.delete(mat_to_show, 20, axis=1)
    tick_locations = np.arange(36)[0::2]
    # for i, loc in enumerate(tick_locations):
    #     if loc > 20:
    #         tick_locations[i] = loc - 1

    # tick_locations = np.delete(tick_locations, 10, axis=0)
    tick_labels = labels[0::2]
    # tick_labels = np.delete(tick_labels, 10, axis=0)

    # tick_labels = np.where(tick_labels == "September2020", "", tick_labels)

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

# %% [markdown]
# ### Load airport data

# %%
# Load As - removing September 2020
As = []
for t in range(36):
    # if t == 20:
    #     continue
    # else:
    As.append(sparse.load_npz(datapath + 'As_' + str(t) + '.npz'))

    # As.append(sparse.load_npz(datapath + 'As_' + str(t) + '.npz'))

As = np.array(As)

# Load Z and Z_col
Z = np.load(datapath + 'Z.npy')
Z_col = np.load(datapath + 'Z_col.npy')
nodes = np.load(datapath + 'nodes.npy', allow_pickle=True).item()


# # %%
# airport_lats = {}
# airport_longs = {}
# airport_continents = {}
# first = True

# # airports.csv from https://ourairports.com/data/

# with open(datapath + 'airports.csv', 'r') as csvfile:
# with open(datapath + 'airports.csv', 'r', encoding="UTF-8") as csvfile:

#     reader = csv.reader(csvfile, delimiter=',', quotechar='"')

#     for line in reader:
#         if first:
#             first = False
#             continue

#         airport = line[1]
#         lat = line[4]
#         long = line[5]
#         continent = line[7]

#         if continent not in ['NA', 'AS', 'EU', 'SA', 'OC', 'AF']:
#             airport_continents[airport] = 'XX'
#         else:
#             airport_continents[airport] = continent

#         airport_lats[airport] = float(lat)
#         airport_longs[airport] = float(long)

# # %% [markdown]
# # ### Load network data

# # %%
# nodes = {}
# count = 0

# sources = [[] for i in range(36)]
# targets = [[] for i in range(36)]
# weights = [[] for i in range(36)]

# years = ['2019', '2020', '2021']
# months = ['01', '02', '03', '04', '05',
#           '06', '07', '08', '09', '10', '11', '12']

# for y in range(len(years)):
#     year = years[y]
#     for m in range(len(months)):
#         month = months[m]

#         filename = glob.glob(datapath + 'flightlist_' +
#                              str(year) + str(month) + '*.csv')[0]
#         print(filename)
#         file = open(filename, 'r')
#   file = open(filename, 'r', encoding="UTF-8")

#         t = 12*y + m
#         first = True

#         for line in file:
#             if first:
#                 first_data = np.array(line.strip('\n').split(','))
#                 source_idx = np.where(first_data == 'origin')[0][0]
#                 target_idx = np.where(first_data == 'destination')[0][0]
#                 time_idx = np.where(first_data == 'firstseen')[0][0]

#                 first = False
#                 continue

#             data = line.strip('\n').split(',')
#             source = data[source_idx]
#             target = data[target_idx]

#             # Ignore blank entries
#             if source == '' or target == '':
#                 continue

#             # Ignore self-connections
#             if source == target:
#                 continue

#             # Ignore XX continent flight sources
#             if source not in airport_continents or airport_continents[source] == 'XX':
#                 continue

#             # Ignore XX continent flight targets
#             if target not in airport_continents or airport_continents[target] == 'XX':
#                 continue

#             if source not in nodes:
#                 nodes[source] = count
#                 count += 1
#             if target not in nodes:
#                 nodes[target] = count
#                 count += 1

#             sources[t].append(nodes[source])
#             targets[t].append(nodes[target])

# %%
n = len(nodes)
T = len(As)
print('Number of nodes:', n)

# # %%
# As = []

# for t in range(36):
#     m = len(sources[t])

#     A = sparse.coo_matrix((np.ones(m), (sources[t], targets[t])), shape=(n, n))
#     A = A + A.T

#     As.append(A)

# T = len(As)
# %% [markdown]
# Set continent communities and assign colours.

# %%
# Z = []
# Z_col = []

# for node in nodes:
#     if node in airport_continents:
#         continent = airport_continents[node]
#     else:
#         continent = 'XX'

#     Z.append(continent)
#     if continent == 'NA':
#         Z_col.append('tab:red')
#     elif continent == 'AS':
#         Z_col.append('tab:blue')
#     elif continent == 'EU':
#         Z_col.append('tab:green')
#     elif continent == 'SA':
#         Z_col.append('tab:orange')
#     elif continent == 'OC':
#         Z_col.append('tab:purple')
#     elif continent == 'AF':
#         Z_col.append('tab:pink')
#     else:
#         Z_col.append('black')

# Z = np.array(Z)
# Z_col = np.array(Z_col)

# # Save Z and Z_col
# np.save(datapath + 'Z.npy', Z)
# np.save(datapath + 'Z_col.npy', Z_col)
# np.save(datapath + "nodes.npy", nodes)

# %%
continents = ['NA', 'AS', 'EU', 'SA', 'OC', 'AF']
continent_cols = ['tab:red', 'tab:blue', 'tab:green',
                  'tab:orange', 'tab:purple', 'tab:pink']

# %%
fig = plt.figure()
handles = [plt.scatter([], [], color='tab:red', marker='o', s=8, label=r'North America'),
           plt.scatter([], [], color='tab:blue',
                       marker='o', s=8, label=r'Asia'),
           plt.scatter([], [], color='tab:green',
                       marker='o', s=8, label=r'Europe'),
           plt.scatter([], [], color='tab:orange', marker='o',
                       s=8, label=r'South America'),
           plt.scatter([], [], color='tab:purple',
                       marker='o', s=8, label=r'Oceania'),
           plt.scatter([], [], color='tab:pink', marker='o', s=8, label=r'Africa')]
plt.close()

# %%
# # Save As
# for t in range(T):
#    sparse.save_npz(datapath + 'As_' + str(t) + '.npz', As[t])


# %%
# # Force each A to be binary
# for t in range(T):
#     As[t] = As[t].sign()

# %%
d = 5
%timeit YAs = regularised_ULSE(As, d, sparse_matrix=True, regulariser=264, flat=False)
%timeit YAs = UASE(As, d, sparse_matrix=True, flat=False)

%timeit YAs = online_embed(As, d, impact_times=np.arange(T)[0::2], regulariser=264, method="UASE", sparse_matrix=True)

# %%
# Embed
d = 5
# method = "URLSE"
# t0 = time.time()
%time YAs = regularised_ULSE(As, d, flat=False, regulariser='auto', sparse_matrix=True)
# %timeit YAs = UASE(As, d, flat=False, sparse_matrix=True)
# # YAs = SSE(As, d, flat=False, procrustes=False, consistent_orientation=False)
# t1 = time.time()
# print("Time taken:", t1-t0)

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))
impact_dates = ["October 2019", "April 2020", "June 2020", "May 2019"]
impact_times = [np.where(labels == date)[0][0] for date in impact_dates]
%time YAs = online_embed(As, d, impact_times=impact_times, regulariser=264, method="URLSE", sparse_matrix=True)
# %timeit YAs = regularised_ULSE(As, d, sparse_matrix=True, regulariser=264)

# YAs = online_embed(As, d, impact_times=np.arange(T)[0::5], regulariser=264, method="URLSE", sparse_matrix=True)
# YAs = regularised_ULSE(As, d, sparse_matrix=True, regulariser=264, flat=False)


# YAs = spherical_degree_correct(YAs, T=T)


# %%
# Online
d = 5
# method = "URLSE"


# def ainv(x):
#     return np.linalg.inv(x.T @ x) @ x.T


T = len(As)

# months = ["January", "February", "March", "April", "May", "June",
#           "July", "August", "September", "October", "November", "December"]
# labels = np.core.defchararray.add(np.array(months * 3),
#                                   np.repeat([" 2019", " 2020", " 2021"], 12))
# impact_dates = ["October 2019", "April 2020", "June 2020", "May 2019"]
# impact_times = [np.where(labels == date)[0][0] for date in impact_dates]
# impact_times = [0]

# # impact_times = np.arange(T)[0:2]


# # def func():

# # t0 = time.time()
# I = [As[impact] for impact in impact_times]

# # XA_partial, _ = UASE(I, d, return_left=True, sparse_matrix=True)
# XA_partial, _ = regularised_ULSE(
#     I, d, return_left=True, sparse_matrix=True, regulariser=264)
# # XA_partial, _ = unfolded_n2v(
# #     I, d, sparse_matrix=True, return_left=True)
# # XA_partial = single_spectral(As[15], d)


# # Online estimate of XA

# def ipca_update(u_initial, s_initial, new_As):
#     # new_A = sparse.hstack(new_As)

#     xa_initial = u_initial @ np.diag(np.sqrt(s_initial))

#     usb = sparse.hstack((u_initial @ np.diag(s_initial), new_As))

#     ub, r, _, _ = sparseqr.qr(usb)

#     ut, st, vtt = sparse.linalg.svds(r)

#     u_new = ub @ ut
#     s_new = st

#     XA_new = u_new[:, 0:d] @ np.diag(np.sqrt(s_new[0:d]))

#     return XA_new


# # A1 = to_laplacian(sparse_stack(As[0]), regulariser=264)
# # A2 = to_laplacian(sparse_stack(As[7]), regulariser=264)

# A1 = to_laplacian(As[0], regulariser=264)
# A2 = to_laplacian(As[7], regulariser=264)


# u_initial, s_initial, _ = sparse.linalg.svds(A1)
# # xa_initial = u_initial @ np.diag(np.sqrt(s_initial))

# XA_partial = xa_initial.copy()

# usb = sparse.hstack((u_initial @ np.diag(s_initial), A2))

# ub, r, _, _ = sparseqr.qr(usb)

# ut, st, vtt = sparse.linalg.svds(r)

# u_new = ub @ ut
# s_new = st

# XA_new = u_new[:, 0:d] @ np.diag(np.sqrt(s_new[0:d]))

# XA_new = ipca_update(u_initial, s_initial, A2)

# XA_partial = XA_new.copy()


# # ainv_XA_partial_sparse = sparse.csr_matrix(ainv(XA_partial))
# YA_online = np.zeros((T, n, XA_partial.shape[1]))
# for t in range(T):
#     # YA_online[t] = As[t] @ XA_partial @ np.linalg.inv(
#     #     XA_partial.T @ XA_partial)
#     YA_online[t] = to_laplacian(
#         As[t], regulariser=264) @ XA_partial @ np.linalg.inv(XA_partial.T @ XA_partial)

# # t1 = time.time()
# # print("Time taken:", t1-t0)

# YAs = YA_online.copy()


# %timeit func()

# YAs = spherical_degree_correct(YAs, T=T)

# %%

# method = "UASE"
# YAs = UASE(As, d, flat=False, sparse_matrix=True)

# method = "OMNI"
# YAs = OMNI(As, d, flat=False, sparse_matrix=True)


# np.save("saved_flight_stuff/YAs_UASE_d=5_binary.npy", YAs)
# np.save("saved_flight_stuff/YAs_OMNI_d=5_multigraph.npy", YAs)

# YAs = SSE(As, d, flat=False, procrustes=True)

# d = 5
# t0 = time.time()
# YAs = unfolded_n2v(As, d, num_walks=20, walklen=10,
#                    window=3, sparse_matrix=True, flat=False)
# YAs = unfolded_n2v(As, d, regulariser=0, num_walks=5, walklen=5,
#                    window=2, sparse_matrix=True, flat=False, two_hop=False)
# YAs = regularised_unfolded_n2v(As, d, regulariser=0, num_walks=5, walklen=5,
#                                window=2, sparse_matrix=True, flat=False, two_hop=False)

# YAs = independent_n2v(As, d, num_walks=20, walklen=10,
#                       window=3, sparse_matrix=True, flat=False)

# YAs = GloDyNE(As, d, num_walks=20, walklen=10,
#               window=3, sparse_matrix=True)


#     "saved_flight_stuff/YAs_indep_n2v_numwalks=20_walklen=10_window=3_binary.npy")

# YAs_flat = np.load(
#     "saved_flight_stuff/YAs_GloDyNE_numwalks=20_walklen=10_window=3_binary.npy")
# n = As[0].shape[0]
# YAs = np.zeros((36, n, d))
# for t in range(36):
#     YAs[t] = YAs_flat[t*n:(t+1)*n, :]

# YAs = np.load(
#     "saved_flight_stuff/unfolded_n2v_embedding_YAs_20walks_10walklen_3window.npy")
# YAs = np.load(
#     "saved_flight_stuff/unfolded_n2v_embedding_YAs_20walks_10walklen_3window_binary.npy")
# YAs = np.load(
#     "saved_flight_stuff/YAs_regularised_reg=auto_unfolded_n2v_numwalks=20_walklen=10_window=3_binary_T=35.npy")
# YAs = np.load(
#     "saved_flight_stuff/YAs_regularised_reg=0_unfolded_n2v_numwalks=20_walklen=10_window=3_binary.npy")
# YAs = np.load(
#     "saved_flight_stuff/YAs_unfolded_n2v_numwalks=5_walklen=5_window=2_multigraph.npy")

# %%
cont_to_look_at = "EU"

euro_nodes = np.where(np.array(Z) == cont_to_look_at)
n_euro = len(euro_nodes[0])
T = len(As)
# %%
# print("WARNING: not looking at Eurpoe right now!!")

# airports = pd.read_csv(datapath + "airports.csv")
# list_of_uk_airports = airports[airports["iso_country"]
#                                == "GB"]["iata_code"].values

# Compute the degrees of the nodes we are interested with connecting with
# the nodes we're interested with

# degrees = np.zeros((T, n_euro))
# for t, A in enumerate(As):
#     for i in range(n_euro):
#         degrees[t, i] = np.sum(A[euro_nodes[0][i], euro_nodes[0]])
# %%
# compute p-value matrix for euopean flights

mat_type = "dissimilarity"
# mat_type = "p_value"

mask = False

if mat_type == "dissimilarity":
    p_val_matrix = np.zeros((T, T))
else:
    p_val_matrix = np.ones((T, T))

num_p_val_comps = 0
num_p_vals_to_compute = int(T*(T-1)/2)
t0 = time.time()
# nodes_compared_mat = np.zeros((T, T))
for i in range(p_val_matrix.shape[0]):
    for j in range(i+1, p_val_matrix.shape[1]):
        num_p_val_comps += 1

        if mask:
            degrees_i = degrees[i, :]
            idx_to_remove_i = np.where(degrees_i == 0)[0]
            degrees_j = degrees[j, :]
            idx_to_remove_j = np.where(degrees_j == 0)[0]

            idx_to_remove = np.unique(np.concatenate(
                (idx_to_remove_i, idx_to_remove_j)))

            ya_i_euro = YAs[i][euro_nodes]
            ya_j_euro = YAs[j][euro_nodes]

            ya_i_euro = np.delete(ya_i_euro, idx_to_remove, axis=0)
            ya_j_euro = np.delete(ya_j_euro, idx_to_remove, axis=0)
        else:
            ya_i_euro = YAs[i][euro_nodes]
            ya_j_euro = YAs[j][euro_nodes]

        if mat_type == "dissimilarity":
            p_val = vector_displacement_test(
                ya_i_euro, ya_j_euro)

            # cosine_displacement = np.zeros(ya_i_euro.shape)
            # for ya_euro_node in range(n_euro):
            #     cosine_displacement[ya_euro_node, :] = distance.cosine(
            #         ya_i_euro[ya_euro_node, :], ya_j_euro[ya_euro_node, :])
            # sum_axis = cosine_displacement.sum(axis=0)
            # p_val = np.linalg.norm(sum_axis)

            if mask:
                nodes_compared = n_euro - len(idx_to_remove)
                # p_val_matrix[i, j] = p_val / nodes_compared
                # p_val_matrix[j, i] = p_val / nodes_compared
                p_val_matrix[i, j] = p_val
                p_val_matrix[j, i] = p_val
            else:
                p_val_matrix[i, j] = p_val / n_euro
                p_val_matrix[j, i] = p_val / n_euro

        else:
            ya_temp = np.concatenate((ya_i_euro, ya_j_euro))
            p_val = test_temporal_displacement_two_times(
                ya_temp, n=ya_i_euro.shape[0])

            p_val_matrix[i, j] = p_val
            p_val_matrix[j, i] = p_val

            # # convert t0 from seconds to minutes
            # time_computing = (time.time() - t0) / 60

            # print("{:.2f} min to do {} / {}".format(time_computing,
            #                                         num_p_val_comps, num_p_vals_to_compute))

            # # compute and print time remaining
            # time_per_comp = time_computing / num_p_val_comps
            # time_remaining = time_per_comp * \
            #     (num_p_vals_to_compute - num_p_val_comps)

            # print("Time remaining: {:.2f} minutes".format(time_remaining))


save_dir = "saved_flight_stuff"

if mat_type == "dissimilarity":
    plot_diss_mat(p_val_matrix, savename=None)
    # plot_diss_mat(p_val_matrix, savename=save_dir+"/URLSE_EU_matrix")
    # plot_diss_mat(p_val_matrix, savename=save_dir +
    #               "/online_URLSE_i={}_EU_matrix".format(impact_times))
else:
    plot_diss_mat(p_val_matrix > 0.05/comb(T, 2))

# np.save(save_dir + "/{}_{}_dissimilarity_matrix_d={}_multigraph".format(method,
#                                                                         cont_to_look_at, d), p_val_matrix)

# %%
# normalise dists_copy and dists_full
dists_copy = p_val_matrix.copy()
dists_copy = dists_copy / np.max(dists_copy)
dists_full = dists_full / np.max(dists_full)
print(vector_displacement_test(dists_copy, dists_full))
plot_diss_mat(np.abs(dists_copy - dists_full))

sum_axis = np.sum(np.abs(dists_copy-dists_full), axis=1)
print("Next month: {}".format(labels[np.argmax(sum_axis)]))

sum_axis = np.sum(np.abs(dists_full), axis=1)
impact_months = pd.DataFrame({
    "label": labels,
    "sum_axis": sum_axis
})
# %%
# method = "URLSE"
# save_dir = "saved_flight_stuff"
# cont_to_look_at = "EU"
# p_val_matrix = np.load(save_dir + "/{}_{}_matrix_d={}.npy".format(
#     method, cont_to_look_at, 5))


# %% [markdown]
# Analysis of p-value matrix
# %%
# To use with p-value matrix
# emb = MDS(n_components=2, max_iter=30000, dissimilarity="precomputed").fit_transform(
#     -np.ma.log(p_val_matrix))

# p_val_matrix = np.load("saved_flight_stuff/URLSE_EU_matrix_d=5.npy")
# p_val_matrix = -np.ma.log(p_val_matrix)
# dissim_multi = np.load(
#     "saved_flight_stuff/URLSE_EU_dissimilarity_matrix_d=5_multigraph.npy")
# dissim_binary = np.load(
#     "saved_flight_stuff/URLSE_EU_dissimilarity_matrix_d=5_binary.npy")
# mats = [p_val_matrix, dissim_multi, dissim_binary]

# To use generally
emb = MDS(n_components=5, max_iter=30000,
          dissimilarity="precomputed").fit_transform(p_val_matrix)
# emb = UMAP(n_components=2).fit_transform(emb)

# emb = emb_copy.copy()

mds_thresh_df = pd.DataFrame(emb)
mds_thresh_df.columns = ["Dimension {}".format(
    i+1) for i in range(emb.shape[1])]
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
months_list = months * 3
# remove index 20
# months_list.pop(20)
mds_thresh_df["Month"] = months_list
years = [2019, 2020, 2021]
years_list = list(np.repeat(years, 12).astype(str))
# remove index 20
# years_list.pop(20)
mds_thresh_df["Year"] = years_list

# # Remove 2020 data
# emb = emb[np.where(mds_thresh_df["Year"] != "2020")]
# mds_thresh_df = mds_thresh_df[mds_thresh_df["Year"] != "2020"]
# months = mds_thresh_df["Month"].unique()
# years = mds_thresh_df["Year"].unique()

# Set matplotlib text size to default
plt.rcParams.update({'font.size': 10})

only_2019 = False

# # Select only 2019
# emb_copy = emb.copy()
# only_2019 = True
# emb = emb[np.where(mds_thresh_df["Year"] == "2019")]
# mds_thresh_df = mds_thresh_df[mds_thresh_df["Year"] == "2019"]


# # Remove september 2020
# emb_copy = emb.copy()
# emb = np.delete(emb, 20, axis=0)
# mds_thresh_df = mds_thresh_df.drop(20)


# Using matplotlib plot with one coloured by month, one coloured by year
if only_2019:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for month in mds_thresh_df["Month"].unique():
        ax.scatter(emb[mds_thresh_df["Month"] == month, 0], emb[mds_thresh_df["Month"] == month, 1],
                   label=month, color=dict(zip(months, px.colors.cyclical.Phase))[month], alpha=0.8)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Add a grid
    ax.grid(alpha=0.5)

    # lower the opacity of the outer box
    ax.spines['top'].set_alpha(0.5)
    ax.spines['right'].set_alpha(0.5)
    ax.spines['bottom'].set_alpha(0.5)
    ax.spines['left'].set_alpha(0.5)

else:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for month in mds_thresh_df["Month"].unique():
        ax[0].scatter(emb[mds_thresh_df["Month"] == month, 0], emb[mds_thresh_df["Month"] == month, 1],
                      label=month, color=dict(zip(months, px.colors.cyclical.mygbm))[month], alpha=0.8)
    for year in mds_thresh_df["Year"].unique():
        ax[1].scatter(emb[mds_thresh_df["Year"] == str(year), 0], emb[mds_thresh_df["Year"] == str(year), 1],
                      label=year, color=dict(zip(mds_thresh_df["Year"].unique(), ["#1B9E77", "#D95F02", "#7570B3"]))[year], alpha=0.8)

    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Add a grid
    ax[0].grid(alpha=0.5)
    ax[1].grid(alpha=0.5)

    # lower the opacity of the outer box
    ax[0].spines['top'].set_alpha(0.5)
    ax[0].spines['right'].set_alpha(0.5)
    ax[0].spines['bottom'].set_alpha(0.5)
    ax[0].spines['left'].set_alpha(0.5)
    ax[1].spines['top'].set_alpha(0.5)
    ax[1].spines['right'].set_alpha(0.5)
    ax[1].spines['bottom'].set_alpha(0.5)
    ax[1].spines['left'].set_alpha(0.5)


# Increase the distance between subplots
fig.subplots_adjust(wspace=0.6)

# plt.show()
# # Save figure
# save_dir = "saved_flight_stuff"
# plt.savefig(save_dir + "/mds_of_{}_dissimilarity_matrix_only_2019.pdf".format(
#     title), dpi=300, bbox_inches="tight")
# %%
# List of dates from March 2020 to June 2021 (inclusive)
aff_by_covid = np.array([False] * 36)
aff_by_covid[14:30] = True

mds_thresh_df["Date"] = pd.to_datetime(
    mds_thresh_df["Year"] + "-" + mds_thresh_df["Month"])
mds_thresh_df["Date"] = mds_thresh_df["Date"].dt.strftime("%Y-%m")

# remove index 20 of aff_by_covid
# aff_by_covid = np.delete(aff_by_covid, 20)

mds_thresh_df["Affected by COVID"] = aff_by_covid
# mds_thresh_df["COVID Deaths"] = [0] + list(deaths[:-1])
mds_thresh_df["COVID Deaths"] = np.log(deaths / cases)

fig = px.scatter(mds_thresh_df, x="Dimension 1", y="Dimension 2",
                 #  color="Affected by COVID",
                 color="COVID Deaths",
                 hover_name="Date",
                 color_discrete_sequence=["#1B9E77", "#D95F02"],
                 labels={"Affected by COVID": "Affected by COVID-19"})
fig
# %%
# Plot with plotly
mds_thresh_df["Date"] = pd.to_datetime(
    mds_thresh_df["Year"] + "-" + mds_thresh_df["Month"])
mds_thresh_df["Date"] = mds_thresh_df["Date"].dt.strftime("%Y-%m")
fig = px.scatter(mds_thresh_df, x="Dimension 1", y="Dimension 2",
                 color="Month",
                 facet_col="Year",
                 hover_name="Date",
                 range_x=[min(emb[:, 0]) - 0.1, max(emb[:, 0] + 0.1)],
                 range_y=[min(emb[:, 1]) - 0.1, max(emb[:, 1] + 0.1)],
                 color_discrete_sequence=px.colors.cyclical.Phase,
                 )
# fig.update_layout(
#     xaxis_title="MDS 1",
#     yaxis_title="MDS 2",
#     font=dict(
#         family="Courier New, monospace",
#         size=18,
#         color="#7f7f7f"
#     )
# )
fig.show()


# %%
# Animation
# %matplotlib qt


def update(frame):
    x = emb[0:frame, 0]
    y = emb[0:frame, 1]
    plt.cla()

    plt.scatter(x, y, c=range(frame))
    plt.xlim(min(emb[:, 0]) - 0.1, max(emb[:, 0] + 0.1))
    plt.ylim(min(emb[:, 1]) - 0.1, max(emb[:, 1] + 0.1))
    plt.title(f"t = {labels[frame]}")


months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=range(1, T), interval=300)
plt.show()


# %%
# Hierarchical clustering


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
    # affinity="precomputed",
    # linkage="average",
)

# Try average linkage cluster

# model = model.fit(-np.ma.log(p_val_matrix))

model = model.fit(p_val_matrix)
# model = model.fit(emb)
# model = model.fit(dissim_multi)

# emb = embed(dissim_binary, 5, "Unfolded Node2Vec", num_walks=200, walklen=10)
# emb = PCA(n_components=5).fit_transform(emb)

# emb = MDS(n_components=5, dissimilarity="precomputed").fit_transform(dissim_multi)
# model = model.fit(p_val_matrix)

# increase the size of the x axis labels
plt.rcParams.update({'font.size': 14})

plt.subplots(figsize=(17, 20))
# plt.title("Hierarchical Clustering")
# plot the top three levels of the dendrogram
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))
# labels_no_sept = np.delete(labels_no_sept, 20)
plot_dendrogram(model, truncate_mode="level", p=15,
                labels=labels, orientation="left")

# Rotate the y axis labels
plt.yticks(rotation=0)

# Add red labels for outliers
y_axis = plt.gca().yaxis
for label in y_axis.get_ticklabels():
    # print(label.get_text())
    if label.get_text() in ['June 2021', 'August 2020', "September 2020"]:
        label.set_color('red')

# Annotations
# First split
# plt.annotate("Not Fully-Affected by COVID-19", xy=(0.5, 0.5), xytext=(0.3, 0.5),
#              xycoords="axes fraction", textcoords="axes fraction",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
#              horizontalalignment="center", verticalalignment="center",
#              fontsize=20)
# plt.annotate("Fully-Affected by COVID-19", xy=(0.5, 0.1), xytext=(0.3, 0.1),
#              xycoords="axes fraction", textcoords="axes fraction",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
#              horizontalalignment="center", verticalalignment="center",
#              fontsize=20)

# # Second split green
# plt.annotate("Unaffected Summer", xy=(0.945, 0.335), xytext=(0.75, 0.335),
#              xycoords="axes fraction", textcoords="axes fraction",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
#              horizontalalignment="center", verticalalignment="center",
#              fontsize=20)

# # Third split green
# plt.annotate("Partially Affected", xy=(0.945, 0.555), xytext=(0.84, 0.555),
#              xycoords="axes fraction", textcoords="axes fraction",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
#              horizontalalignment="center", verticalalignment="center",
#              fontsize=20)

# # Fourth split green
# plt.annotate("Unaffected Non-Summer", xy=(0.86, 0.76), xytext=(0.7, 0.76),
#              xycoords="axes fraction", textcoords="axes fraction",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
#              horizontalalignment="center", verticalalignment="center",
#              fontsize=20)

# # Second split orange
# plt.annotate("COVID-19 Start", xy=(0.88, 0.04), xytext=(0.765, 0.04),
#              xycoords="axes fraction", textcoords="axes fraction",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
#              horizontalalignment="center", verticalalignment="center",
#              fontsize=20)
# plt.annotate("Other COVID-19", xy=(0.88, 0.15), xytext=(0.765, 0.15),
#              xycoords="axes fraction", textcoords="axes fraction",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
#              horizontalalignment="center", verticalalignment="center",
#              fontsize=20)


plt.tight_layout()
# plt.show()
# plt.savefig(
#     "saved_flight_stuff/EU_hierarchical_clustering_multigraph_distance_matrix.pdf")

# %%
fig, axs = plt.subplots(2, 3, figsize=(20, 15))

file_names = ["NA", "SA", "EU", "AS", "OC", "AF"]
titles = ["North America", "South America",
          "Europe", "Asia", "Oceania", "Africa"]

method = "URLSE"
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))
for i in range(len(file_names)):
    axs[i % 2, np.repeat(np.arange(3), 2)[i]].set_title(titles[i])
    axs[i % 2, np.repeat(np.arange(3), 2)[i]].matshow(
        np.load("saved_flight_stuff{}_{}_matrix.npy".format(method, file_names[i])) > 0.001)
    axs[i % 2, np.repeat(np.arange(3), 2)[i]].set_xticks(
        np.arange(36)[0::5], labels[0::5], rotation=45)
    axs[i % 2, np.repeat(np.arange(3), 2)[i]].set_yticks(
        np.arange(36)[0::5], labels[0::5], rotation=45)

fig.tight_layout()
# plt.savefig("saved_flight_stuff/{}_tests_by_continent_binary.pdf".format(method))

# %%

# %%
# %%
# compute dissimilarity matrix for each continent
conts = ["EU", "AS", "NA", "SA", "AF", "OC"]
mats = []
for cont_to_look_at in conts:
    p_val_matrix = np.zeros((T, T))
    euro_nodes = np.where(np.array(Z) == cont_to_look_at)

    num_p_val_comps = 0
    num_p_vals_to_compute = int(T*(T-1)/2)
    t0 = time.time()
    for i in range(p_val_matrix.shape[0]):
        for j in range(i+1, p_val_matrix.shape[1]):

            # ya_temp = np.row_stack(
            #     [YAs[i][euro_nodes], YAs[j][euro_nodes]])
            # p_val = test_temporal_displacement_two_times(
            #     ya_temp, n=YAs[i][euro_nodes].shape[0])

            p_val = vector_displacement_test(
                YAs[i][euro_nodes], YAs[j][euro_nodes])

            p_val_matrix[i, j] = p_val
            p_val_matrix[j, i] = p_val

            num_p_val_comps += 1

        # # convert t0 from seconds to minutes
        # time_computing = (time.time() - t0) / 60

        # print("{:.2f} min to do {} / {}".format(time_computing,
        #                                         num_p_val_comps, num_p_vals_to_compute))

        # # compute and print time remaining
        # time_per_comp = time_computing / num_p_val_comps
        # time_remaining = time_per_comp * \
        #     (num_p_vals_to_compute - num_p_val_comps)

        # print("Time remaining: {:.2f} minutes".format(time_remaining))

    # plt.matshow(p_val_matrix)
    # save_dir = "saved_flight_stuff"
    # np.save(save_dir + "/{}_{}_dissimilarity_matrix_d={}".format(method,
        #  cont_to_look_at, d), p_val_matrix)

    # remove row and column 20
    p_val_matrix = np.delete(p_val_matrix, 20, axis=0)
    p_val_matrix = np.delete(p_val_matrix, 20, axis=1)

    mats.append(p_val_matrix)

# %%
# Plot each dissimilarity matrix

# change text size
plt.rcParams.update({'font.size': 14})

conts = ["Europe", "Asia", "North America",
         "South America", "Africa", "Oceania"]
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))

# plot p-value matrices
fig, axs = plt.subplots(3, 2, figsize=(20, 22))
for i in range(len(conts)):
    # ax = axs[i//3, i % 3]
    ax = axs[i % 3, i//3]
    mat_to_show = mats[i]
    # thresh = 0.05 / comb(T, 2)
    # mat_to_show = np.where(mat_to_show > thresh, 1, 0)
    # np.fill_diagonal(mat_to_show, 1)
    ax.set_title(conts[i])

    tick_locations = np.arange(36)[0::2]
    # for i, loc in enumerate(tick_locations):
    #     if loc > 20:
    #         tick_locations[i] = loc - 1

    # tick_locations = np.delete(tick_locations, 10, axis=0)
    tick_labels = labels[0::2]
    # tick_labels = np.delete(tick_labels, 10, axis=0)

    # tick_labels = np.where(tick_labels == "September2020", "", tick_labels)

    cax = ax.matshow(mat_to_show)
    _ = ax.set_xticks(tick_locations, tick_labels, rotation=90)
    _ = ax.set_yticks(tick_locations, tick_labels, rotation=0)
    _ = ax.xaxis.set_ticks_position('bottom')
    # fig.colorbar(cax)

# Colourbar for all plots
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.80, 0.15, 0.03, 0.7])
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.set_label("Distance")


# # Increase the distance between subplot rows
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.0)

# # Add title to figure
# # fig.suptitle(
# #     "Dissimilarity matrices for different continents from multigraph adjacency matrices")
# # add space between plot and title
# fig.subplots_adjust(top=0.85)

# Add white background
fig.patch.set_facecolor('white')

# plt.tight_layout()

# Save plot
save_dir = "saved_flight_stuff"
plt.savefig(
    save_dir + "/all_dissimilarity_matrices_d={}_mutligraph.pdf".format(d))

# %%
# For each matrix compute the column sums and plot the results
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for i in range(len(conts)):
    ax = axs[i//3, i % 3]
    mat_to_show = mats[i]
    ax.plot(np.sum(mat_to_show, axis=0))
    ax.set_title(conts[i])
    ax.axvline(14, label="March 2020", c="red")
    ax.axvline(30, label="July 2021", c="red")

# %%
# Comparing embeddings from permutation test, multigraph, and binary graphs
p_val_matrix = np.load("saved_flight_stuff/URLSE_EU_matrix_d=5.npy")
p_val_matrix = -np.ma.log(p_val_matrix)
dissim_multi = np.load(
    "saved_flight_stuff/URLSE_EU_dissimilarity_matrix_d=5_multigraph.npy")
dissim_binary = np.load(
    "saved_flight_stuff/URLSE_EU_dissimilarity_matrix_d=5_binary.npy")

mats = [p_val_matrix, dissim_multi, dissim_binary]
titles = ["Permutation test", "Multigraph", "Binary graph"]
embs = []
for i in range(len(mats)):
    if titles[i] == "Permutation test":
        emb = MDS(
            n_components=2, dissimilarity="precomputed").fit_transform(-np.ma.log(mats[i]))
    else:
        emb = MDS(n_components=2,
                  dissimilarity="precomputed").fit_transform(mats[i])

    embs.append(emb)


# %%
# Just the europe plot
# increase text size in plot
plt.rcParams.update({'font.size': 15})
method = "Unfolded Node2Vec"

i = 0
mat_to_show = p_val_matrix
title = "binary"

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))
# remove september 2020
tick_locations = np.arange(36)[0::2]
tick_labels = labels[0::2]

fig, ax = plt.subplots(1, 1, figsize=(9, 9))

cax = ax.matshow(mat_to_show)
_ = ax.set_xticks(tick_locations, tick_labels, rotation=90)
_ = ax.set_yticks(tick_locations, tick_labels, rotation=0)
fig.colorbar(cax)
plt.grid()

# Set white background
fig.patch.set_facecolor('white')

plt.savefig("saved_flight_stuff/{}_EU_matrix_{}.png".format(method, title), dpi=300,
            bbox_inches="tight")

# %%
i = 0
emb = MDS(n_components=2, max_iter=30000,
          dissimilarity="precomputed").fit_transform(p_val_matrix)

mds_thresh_df = pd.DataFrame(emb)
mds_thresh_df.columns = ["Dimension {}".format(
    i+1) for i in range(emb.shape[1])]
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
mds_thresh_df["Month"] = months*3
years = [2019, 2020, 2021]
mds_thresh_df["Year"] = list(np.repeat(years, 12).astype(str))

# emb = emb[np.where(mds_thresh_df["Year"] != "2020")]
# mds_thresh_df = mds_thresh_df[mds_thresh_df["Year"] != "2020"]
# months = mds_thresh_df["Month"].unique()
# years = mds_thresh_df["Year"].unique()

# Set matplotlib text size to default
plt.rcParams.update({'font.size': 10})

for color_by in ["month", "year"]:
    # Using matplotlib plot with one coloured by month, one coloured by year
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if color_by == "month":
        for month in months:
            ax.scatter(emb[mds_thresh_df["Month"] == month, 0], emb[mds_thresh_df["Month"] == month, 1],
                       label=month, color=dict(zip(months, px.colors.cyclical.Phase))[month], alpha=0.8)
    if color_by == "year":
        for year in years:
            ax.scatter(emb[mds_thresh_df["Year"] == str(year), 0], emb[mds_thresh_df["Year"] == str(year), 1],
                       label=year, color=dict(zip(years, ["#1B9E77", "#D95F02", "#7570B3"]))[year], alpha=0.8)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Add a grid
    ax.grid(alpha=0.5)

    # lower the opacity of the outer box
    ax.spines['top'].set_alpha(0.5)
    ax.spines['right'].set_alpha(0.5)
    ax.spines['bottom'].set_alpha(0.5)
    ax.spines['left'].set_alpha(0.5)

    # Increase the distance between subplots
    fig.subplots_adjust(wspace=0.6)

    # plt.show()

    # Save figure
    save_dir = "saved_flight_stuff"
    # plt.savefig(save_dir + "/mds_of_{}_dissimilarity_matrix_{}_no_september.pdf".format(
    # title, color_by), dpi=300, bbox_inches="tight")

# %%


def plot_diss_mat(mat_to_show, title=None):
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    labels = np.core.defchararray.add(np.array(months * 3),
                                      np.repeat([" 2019", " 2020", " 2021"], 12))
    # remove september 2020
    mat_to_show = np.delete(mat_to_show, 20, axis=0)
    mat_to_show = np.delete(mat_to_show, 20, axis=1)
    tick_locations = np.arange(36)[0::2]
    for i, loc in enumerate(tick_locations):
        if loc > 20:
            tick_locations[i] = loc - 1

    tick_locations = np.delete(tick_locations, 10, axis=0)
    tick_labels = labels[0::2]
    tick_labels = np.delete(tick_labels, 10, axis=0)

    # tick_labels = np.where(tick_labels == "September2020", "", tick_labels)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    cax = ax.matshow(mat_to_show)
    _ = ax.set_xticks(tick_locations, tick_labels, rotation=90)
    _ = ax.set_yticks(tick_locations, tick_labels, rotation=0)
    fig.colorbar(cax)
    if title is not None:
        ax.set_title(title)

    plt.grid()


# plot_diss_mat(dists)

# %%
# Compare individual nodes
# Differences based on individual nodes

# for node in nodes:
#     if node in airport_continents:
#         continent = airport_continents[node]

# YAs_flat = np.load(
#     "saved_flight_stuff/YAs_unfolded_n2v_numwalks=20_walklen=10_window=3_binary.npy")
# YAs = np.zeros((T, len(nodes), 5))
# for t in range(T):
#     YAs[t, :, :] = YAs_flat[t*len(nodes):(t+1)*len(nodes), :]

# airport_of_interest = "EGGD"  # bristol
# airport_of_interest = "UUEE"  # moscow (slow covid recovery)
# airport_of_interest = "LEAL"  # alicante (very weird)
# airport_of_interest = "LIMJ" #genoa
# airport_of_interest = "LIRP" #pisa (super weird)
# airport_of_interest = "LIML"  # milan (VERY touristy in summer)
# airport_of_interest = "LDZA"  # zagreb (christmas?)
# airport_of_interest = "EGNX"  # East middlands (small UK)
# airport_of_interest = "EGLL"  # London Heathrow (big UK) - again slow recovery
# airport_of_interest = "LFLL" # Lyon (ski destination)
# airport_of_interest = "LSGG"  # Geneva (ski destination)
# airport_of_interest = "LOWI"  # Innsbruck (ski destination)
# airport_of_interest = "VHHH"  # Hong kong (can see the protests from June 2019)
# airport_of_interest = "LEPA"
# remove_sept2020 = True

# airport_of_interest = "EGBT"

# YAs = np.load("saved_flight_stuff/YAs_URLSE_d=5_degreecorr_binary.npy")
# YAs = np.load("saved_flight_stuff/YAs_UASE_d=5_degreecorr_binary.npy")

# YAs = np.load(
#     "saved_flight_stuff/unfolded_n2v_embedding_YAs_20walks_10walklen_3window_binary.npy")
# YAs = np.load("saved_flight_stuff/YAs_indep_n2v_numwalks=20_walklen=10_window=3_binary.npy")

# YA_flat = np.load("saved_flight_stuff/YAs_GloDyNE_numwalks=20_walklen=10_window=3_binary.npy")
# YAs = np.zeros((T, len(nodes), 5))
# for t in range(T):
#     YAs[t, :, :] = YA_flat[t*len(nodes):(t+1)*len(nodes), :]

remove_sept2020 = False

airport_idx = list(nodes.keys()).index(airport_of_interest)
airport_vecs = YAs[:, airport_idx, :]

dists = np.zeros((T, T))
for i in range(T):
    for j in range(i+1, T):

        dist = np.linalg.norm(airport_vecs[i, :] - airport_vecs[j, :])
        # dist = distance.cosine(airport_vecs[i, :], airport_vecs[j, :])

        # dot product difference
        # dist = np.dot(airport_vecs[i, :], airport_vecs[j, :])

        # Spearman distance
        # dist = spearmanr(airport_vecs[i, :], airport_vecs[j, :])[0]

        dists[i, j] = dist
        dists[j, i] = dist

plot_diss_mat(dists)

# save plot
# plt.savefig("saved_flight_stuff/dissimilarity_matrix_{}.pdf".format(
#     airport_of_interest), dpi=300, bbox_inches="tight")

# %%
remove_sept2020 = False

emb = MDS(n_components=2, max_iter=30000,
          dissimilarity="precomputed").fit_transform(dists)

mds_thresh_df = pd.DataFrame(emb)
mds_thresh_df.columns = ["Dimension {}".format(
    i+1) for i in range(emb.shape[1])]
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
years = ["2019", "2020", "2021"]

if remove_sept2020:
    months_wo_sept = list(months*3)
    months_wo_sept.pop(20)
    mds_thresh_df["Month"] = months_wo_sept
    years_wo_sept = list(np.repeat([2019, 2020, 2021], 12).astype(str))
    years_wo_sept.pop(20)
    mds_thresh_df["Year"] = years_wo_sept
else:
    mds_thresh_df["Month"] = list(months*3)
    years = [2019, 2020, 2021]
    mds_thresh_df["Year"] = list(np.repeat(years, 12).astype(str))

# # Remove 2020 data
# emb = emb[np.where(mds_thresh_df["Year"] != "2020")]
# mds_thresh_df = mds_thresh_df[mds_thresh_df["Year"] != "2020"]
# months = mds_thresh_df["Month"].unique()
# years = mds_thresh_df["Year"].unique()

# Set matplotlib text size to default
plt.rcParams.update({'font.size': 10})


# Using matplotlib plot with one coloured by month, one coloured by year
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for month in months:
    ax[0].scatter(emb[mds_thresh_df["Month"] == month, 0], emb[mds_thresh_df["Month"] == month, 1],
                  label=month, color=dict(zip(months, px.colors.cyclical.mygbm))[month], alpha=0.8)
for year in years:
    ax[1].scatter(emb[mds_thresh_df["Year"] == str(year), 0], emb[mds_thresh_df["Year"] == str(year), 1],
                  label=year, color=dict(zip(years, ["#1B9E77", "#D95F02", "#7570B3"]))[year], alpha=0.8)

ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Add a grid
ax[0].grid(alpha=0.5)
ax[1].grid(alpha=0.5)

# lower the opacity of the outer box
ax[0].spines['top'].set_alpha(0.5)
ax[0].spines['right'].set_alpha(0.5)
ax[0].spines['bottom'].set_alpha(0.5)
ax[0].spines['left'].set_alpha(0.5)
ax[1].spines['top'].set_alpha(0.5)
ax[1].spines['right'].set_alpha(0.5)
ax[1].spines['bottom'].set_alpha(0.5)
ax[1].spines['left'].set_alpha(0.5)


# Increase the distance between subplots
fig.subplots_adjust(wspace=0.6)

plt.show()

# # Save figure
# save_dir = "saved_flight_stuff"
# plt.savefig(save_dir + "/mds_of_{}_dissimilarity_matrix.pdf".format(
#     titles[i]), dpi=300, bbox_inches="tight")

# %%


# def embed_airport(airport_of_interest, remove_sept2020=True):
#     # airport_of_interest = "EGNX" # East middlands (small UK)
#     # airport_of_interest = "EGLL"  # London Heathrow (big UK) - again slow recovery

#     airport_idx = list(nodes.keys()).index(airport_of_interest)
#     airport_vecs = YAs[:, airport_idx, :]

#     dists = np.zeros((T, T))
#     for i in range(T):
#         for j in range(i+1, T):
#             dist = np.linalg.norm(airport_vecs[i, :] - airport_vecs[j, :])
#             # dist = vector_displacement_test(airport_vecs[i, :], airport_vecs[j, :])
#             # dist = distance.cosine(airport_vecs[i, :], airport_vecs[j, :])

#             dists[i, j] = dist
#             dists[j, i] = dist

#     if remove_sept2020:
#         dists_reduced = dists.copy()
#         # remove row and column 20
#         dists_reduced = np.delete(dists_reduced, 20, axis=0)
#         dists_reduced = np.delete(dists_reduced, 20, axis=1)
#         dists = dists_reduced
#         # plt.matshow(dists)
#     # else:
#     #     plt.matshow(dists)

#     emb = MDS(n_components=2, max_iter=30000,
#               dissimilarity="precomputed").fit_transform(dists)

#     mds_thresh_df = pd.DataFrame(emb)
#     mds_thresh_df.columns = ["Dimension {}".format(
#         i+1) for i in range(emb.shape[1])]
#     months = ["January", "February", "March", "April", "May", "June",
#               "July", "August", "September", "October", "November", "December"]

#     if remove_sept2020:
#         months_wo_sept = list(months*3)
#         months_wo_sept.pop(20)
#         mds_thresh_df["Month"] = months_wo_sept
#         years_wo_sept = list(np.repeat([2019, 2020, 2021], 12).astype(str))
#         years_wo_sept.pop(20)
#         mds_thresh_df["Year"] = years_wo_sept
#     else:
#         mds_thresh_df["Month"] = list(months*3)
#         years = [2019, 2020, 2021]
#         mds_thresh_df["Year"] = list(np.repeat(years, 12).astype(str))

#     return(mds_thresh_df)


# # Estimate covid recovery month
# # Compute the distances between 2019 and 2021 months
# plt.figure(figsize=(12, 6))
# for aiport in ["EGNX", "EGLL", "LIMJ", "LIRP"]:
#     month_dist = []
#     mds_thresh_df = embed_airport(aiport)
#     for month in months:
#         # Get YA for month 2019
#         month_2019 = mds_thresh_df[(mds_thresh_df["Month"] == month) & (
#             mds_thresh_df["Year"] == "2019")][["Dimension {}".format(i+1) for i in range(emb.shape[1])]].values[0]
#         # month_2020 = mds_thresh_df[(mds_thresh_df["Month"] == month) & (
#         #     mds_thresh_df["Year"] == "2020")][["Dimension {}".format(i+1) for i in range(emb.shape[1])]].values[0]
#         month_2021 = mds_thresh_df[(mds_thresh_df["Month"] == month) & (
#             mds_thresh_df["Year"] == "2021")][["Dimension {}".format(i+1) for i in range(emb.shape[1])]].values[0]

#         dist = np.linalg.norm(month_2019 - month_2021)
#         month_dist.append(dist)

#     # normalise month_dist
#     month_dist = np.array(month_dist)
#     month_dist = month_dist / np.max(month_dist)

#     plt.plot(months, month_dist, marker="o", label=aiport)

# plt.legend()
# # plt.figure(figsize=(12, 6))
# # plt.plot(months, month_dist, marker="o")

# %%
# For each topic get the degree of each node
n = len(nodes)
degrees = np.zeros((T, n))
for t, A in enumerate(As):
    for i in range(n):
        degrees[t, i] = np.sum(A[i, :])
# %%
idx_to_airport = nodes.copy()
airports = list(nodes)
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
dates = np.core.defchararray.add(np.array(months * 3),
                                 np.repeat(["2019", "2020", "2021"], 12))


date_1 = []
date_2 = []
all_dists = []
airport_compared = []
for i in tqdm(range(T)):
    for j in range(i+1, T):
        mask = np.logical_and(degrees[i, :] > 0, degrees[j, :] > 0)
        dists = []
        for airport_idx in np.where(mask == True)[0]:
            airport_vecs = YAs[:, airport_idx, :]
            dist = distance.cosine(airport_vecs[i, :], airport_vecs[j, :])
            dists.append(dist)

        # Sort dists and corresponding airports
        sorted_dists, sorted_airports = zip(
            *sorted(zip(dists, np.array(airports)[mask])))

        date_1.extend([dates[i]] * np.sum(mask))
        date_2.extend([dates[j]] * np.sum(mask))
        all_dists.extend(sorted_dists)
        airport_compared.extend(sorted_airports)

        # for K in range(20):
        #     print("{} vs {} | {} : {}".format(
        #         idx_to_topic[i], idx_to_topic[j], sorted_airports[-K-1], sorted_dists[-K-1]))

df = pd.DataFrame({
    "Date_1": date_1,
    "Date_2": date_2,
    "Airport": airport_compared,
    "Distance": all_dists,
    # "degree_1": degrees[[list(dates).index(t) for t in date_1], [airports.index(w) for w in airport_compared]],
    # "degree_2": degrees[[list(dates).index(t) for t in date_2], [airports.index(w) for w in airport_compared]],
})
df = df.sort_values(by="Distance", ascending=False)
df[0:20]

# %%
# Plot of the dissimilarity matrix with the p-value matrix
# diss_matrix = np.load(
#     "saved_flight_stuff/URLSE_EU_dissimilarity_matrix_d=5_multigraph.npy")
# p_val_matrix = np.load("saved_flight_stuff/URLSE_EU_matrix_d=5.npy")

# Apply Bonferroni correction to the p-value matrix and control type 1 error at 5%
thresh = 0.05 / comb(T, 2)
p_val_matrix_corr = ~ (p_val_matrix > thresh)

# Plot matrices, removing september 2020 from both

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))


plt.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(2, 1, figsize=(16, 20), sharex=True)
for idx, mat_to_show in enumerate([diss_matrix, p_val_matrix_corr]):

    # remove september 2020
    # mat_to_show = np.delete(mat_to_show, 20, axis=0)
    # mat_to_show = np.delete(mat_to_show, 20, axis=1)
    tick_locations = np.arange(36)[0::2]
    # for i, loc in enumerate(tick_locations):
    #     if loc > 20:
    #         tick_locations[i] = loc - 1

    # tick_locations = np.delete(tick_locations, 10, axis=0)
    tick_labels = labels[0::2]
    # tick_labels = np.delete(tick_labels, 10, axis=0)

    cax = ax[idx].matshow(mat_to_show, cmap=matplotlib.cm.viridis)
    _ = ax[idx].set_xticks(tick_locations, tick_labels, rotation=90)
    _ = ax[idx].set_yticks(tick_locations, tick_labels, rotation=0)
    ax[idx].grid()

    if idx == 0:
        # Place colourbar below subplot first
        cbar = fig.colorbar(
            cax, ax=ax[idx], orientation="vertical", panchor=False, pad=0.058)
        cbar.ax.set_xlabel("Distance")
    else:
        # # binary legend to show significance
        # cbar = fig.colorbar(cax, ax=ax, orientation="horizontal", pad=0.03, ticks=[0, 1])
        # cbar.ax[idx].set_xticklabels(["Not significant", "Significant"])

        cmap = matplotlib.cm.get_cmap('viridis')
        col_1 = cmap(0)
        col_2 = cmap(255)
        legend_elements = [Patch(facecolor=col_1,
                                 label='Not Significant'),
                           Patch(facecolor=col_2,
                                 label='Significant'),
                           ]

        # Create the figure
        ax[idx].legend(handles=legend_elements, loc="upper right",
                       title="Distance Significance", fancybox=True, frameon=False, ncol=1, bbox_to_anchor=(1.4, 1))

    # increase matplotlib font size

    plt.tight_layout()

    # set white background
    fig.patch.set_facecolor('white')

    # # save
    # if idx == 0:
    #     plt.savefig(
    #         "saved_flight_stuff/URLSE_EU_matrix_multigraph_no_september.png")
    # else:
    #     plt.savefig("saved_flight_stuff/URLSE_EU_matrix_perm_test.png")

# plt.savefig("saved_flight_stuff/EU_matrix_with_perm_test.png")

# %%
degrees = []
for t in range(T):
    degrees.append(np.sum(np.sum(As[t], axis=0)))
    times.append(t)

plt.figure(figsize=(16, 8))
_ = plt.plot(labels, degrees)

# rotate x axis
_ = plt.xticks(rotation=90)

# %%
# Plot of the dissimilarity matrix with the p-value matrix
diss_matrix = np.load(
    "saved_flight_stuff/URLSE_EU_dissimilarity_matrix_d=5_multigraph.npy")
p_val_matrix = np.load("saved_flight_stuff/URLSE_EU_matrix_d=5.npy")

# Apply Bonferroni correction to the p-value matrix and control type 1 error at 5%
thresh = 0.05 / comb(T, 2)
p_val_matrix_corr = ~ (p_val_matrix > thresh)

# Plot matrices, removing september 2020 from both

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))


plt.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(2, 2, figsize=(16, 20), sharex=True, sharey=True)
for idx, mat_to_show in enumerate([diss_matrix, p_val_matrix_corr]):

    # remove september 2020
    # mat_to_show = np.delete(mat_to_show, 20, axis=0)
    # mat_to_show = np.delete(mat_to_show, 20, axis=1)
    tick_locations = np.arange(36)[0::2]
    # for i, loc in enumerate(tick_locations):
    #     if loc > 20:
    #         tick_locations[i] = loc - 1

    # tick_locations = np.delete(tick_locations, 10, axis=0)
    tick_labels = labels[0::2]
    # tick_labels = np.delete(tick_labels, 10, axis=0)

    cax = ax[idx, 0].matshow(mat_to_show, cmap=matplotlib.cm.viridis)
    _ = ax[idx, 0].set_xticks(tick_locations, tick_labels, rotation=90)
    _ = ax[idx, 0].set_yticks(tick_locations, tick_labels, rotation=0)
    ax[idx, 0].grid()

    if idx == 0:
        # Place colourbar below subplot first
        # cbar = fig.colorbar(
        #     cax, ax=ax[idx], orientation="horizontal", panchor=False, pad=0.058)
        # cbar.ax.set_xlabel("Distance")
        continue
    else:
        # # binary legend to show significance
        # cbar = fig.colorbar(cax, ax=ax, orientation="horizontal", pad=0.03, ticks=[0, 1])
        # cbar.ax[idx].set_xticklabels(["Not significant", "Significant"])

        cmap = matplotlib.cm.get_cmap('viridis')
        col_1 = cmap(0)
        col_2 = cmap(255)
        legend_elements = [Patch(facecolor=col_1,
                                 label='Not Significant'),
                           Patch(facecolor=col_2,
                                 label='Significant'),
                           ]

        # Create the figure
        ax[idx, 0].legend(handles=legend_elements, loc="upper right",
                          title="Distance Significance", fancybox=True, frameon=False, ncol=1, bbox_to_anchor=(1.4, 1))

    # increase matplotlib font size

# covid plot
covid_all = pd.read_csv("covid_data/covid_eu.csv")

# remove dates after december 2021
covid_all = covid_all[covid_all["year"] <= 2021]

covid_avg = covid_all.groupby(["month", "year"]).mean().reset_index()

covid_avg = covid_avg.sort_values(by=["year", "month"])
cases = np.zeros((T))
cases[12:] = covid_avg["cases"].values
deaths = np.zeros((T))
deaths[12:] = covid_avg["deaths"].values

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))


# plot deaths and labels such that the y axis is shared with the first subplot - making sure they line up
ax[0, 1].plot(deaths, labels, color="red")
ax[0, 1].set_xlabel("Deaths")
ax[0, 1].set_xticks(tick_locations, tick_labels, rotation=90)
ax[0, 1].grid()


# hide subplot 2,2
ax[idx, 1].axis('off')

# plt.tight_layout()

# set white background
fig.patch.set_facecolor('white')

# # save
# if idx == 0:
#     plt.savefig(
#         "saved_flight_stuff/URLSE_EU_matrix_multigraph_no_september.png")
# else:
#     plt.savefig("saved_flight_stuff/URLSE_EU_matrix_perm_test.png")

# plt.savefig("saved_flight_stuff/EU_matrix_with_perm_test.png")

# %%
# Work out the difference between network states pre-covid
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))
year_matrix = dists[0:12, 0:12]
tick_labels = labels[0:12]
tick_locations = np.arange(12)

fig, ax = plt.subplots(1, 1, figsize=(9, 9))

cax = ax.matshow(year_matrix)
_ = ax.set_xticks(tick_locations, tick_labels, rotation=90)
_ = ax.set_yticks(tick_locations, tick_labels, rotation=0)
fig.colorbar(cax)
# %%

# Difference between Auguest and January
date1 = "July 2019"
date2 = "July 2020"
emb1 = YAs[np.where(labels == date1)[0][0]][euro_nodes]
emb2 = YAs[np.where(labels == date2)[0][0]][euro_nodes]

diff = np.linalg.norm(emb1 - emb2, axis=1)

airport_labels = np.array(list(nodes.keys()))[euro_nodes]
airports = pd.read_csv(datapath+"airports.csv")
diff_df = pd.DataFrame({
    "diff": diff,
    "airport": airport_labels,
    "airport_name": [airports[airports["ident"] == airport_code]["name"].values[0] for airport_code in airport_labels],
    "avg_degree": np.mean(degrees, axis=0),
    "deg_diff": [np.abs(np.sum(As[np.where(labels == date1)[
        0][0]][:, nodes[code]]) - np.sum(As[np.where(labels == date2)[
            0][0]][:, nodes[code]])) for code in airport_labels]
})
diff_df = diff_df.sort_values(by="diff", ascending=False)

diff_df.head(20)
# %%
degrees_to_airport1 = As[np.where(labels == "April 2019")[
    0][0]][:, nodes["LGRP"]].todense()

# flatten
degrees_to_airport1 = np.array(degrees_to_airport1).flatten()

df1 = pd.DataFrame({
    "degree": degrees_to_airport1[euro_nodes],
    "airport": airport_labels,
    "airport_name": [airports[airports["ident"] == airport_code]["name"].values[0] for airport_code in airport_labels]
})
df1 = df1.sort_values(by="degree", ascending=False)

degrees_to_airport2 = As[np.where(labels == "June 2019")[
    0][0]][:, nodes["LGRP"]].todense()

# flatten
degrees_to_airport2 = np.array(degrees_to_airport2).flatten()

df2 = pd.DataFrame({
    "degree": degrees_to_airport2[euro_nodes],
    "airport": airport_labels,
    "airport_name": [airports[airports["ident"] == airport_code]["name"].values[0] for airport_code in airport_labels]

})
df2 = df2.sort_values(by="degree", ascending=False)

# %%
df1["degree"].sum()
df2["degree"].sum()

len(set(df1[df1["degree"] > 0]["airport"].values) -
    set(df2[df2["degree"] > 0]["airport"].values))

airports_which_dont_fly_both = set(
    df1[df1["degree"] > 0]["airport"].values) - set(df2[df2["degree"] > 0]["airport"].values)

[airports[airports["ident"] == airport_code]["name"].values[0]
    for airport_code in airports_which_dont_fly_both]

# %%

# Select airport codes which start with the letter Z
airport_codes = [code for code in nodes.keys() if code[0] == "Z"]

airports = airportsdata.load("ICAO")

for code in airport_codes:
    try:
        print(code + ": " + airports[code]["name"], ", " +
              pycountry.countries.get(alpha_2=airports[code]["country"]).name)
    except:
        print("No airport found for code: ", code)
# %%


# %%
