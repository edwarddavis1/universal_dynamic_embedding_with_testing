# %%
"""C5 Data from: https://github.com/OxCGRT/covid-policy-tracker"""
import datetime
import pycountry
from matplotlib.patches import Patch
import matplotlib
from sklearn.manifold import Isomap
from matplotlib.animation import FuncAnimation
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
# # Covid policy tracker
# # C5m close public transport
# c5 = pd.read_csv("covid_data/c5m_close_public_transport.csv")

# ukc5 = c5[c5["country_name"] == "Germany"]

# plt.figure(figsize=(15, 10))
# _ = plt.plot(ukc5.columns.values[6:], ukc5.iloc[0].values[6:])

# # Have tick every 10 datapoints
# _ = plt.xticks(np.arange(0, len(ukc5.columns.values[6:]), 30),
#                ukc5.columns.values[6::30], rotation=90)

# %%
datapath = "../../Datasets/Flight Data/"

# Load As
As = []
T = 36
for t in range(T):
    As.append(sparse.load_npz(datapath + 'As_' + str(t) + '.npz'))

# Load Z and Z_col
Z = np.load(datapath + 'Z.npy')
Z_col = np.load(datapath + 'Z_col.npy')
nodes = np.load(datapath + 'nodes.npy', allow_pickle=True).item()

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))
# Get european nodes
cont_to_look_at = "EU"
euro_nodes = np.where(np.array(Z) == cont_to_look_at)
# %%
# TODO this could definitely be faster
# Get data on each airport for each month
airports = pd.read_csv(datapath + "airports.csv")
country_codes_in_data = []
airports_not_found = []
flights_for_each_country_month = []
flights_for_each_country_number = []
flights_for_each_country_airport = []
flights_for_each_country_code = []
flights_for_each_country_country = []
for code in np.array(list(nodes.keys()))[euro_nodes]:
    try:
        country_of_airport = airports[airports["ident"]
                                      == code]["iso_country"].values[0]
        country_codes_in_data.append(country_of_airport)

        for t in range(T):
            flights_for_each_country_month.append(labels[t])
            flights_for_each_country_number.append(
                np.sum(As[t][:, nodes[code]]))
            flights_for_each_country_airport.append(code)
            flights_for_each_country_code.append(country_of_airport)
            try:
                country_name = pycountry.countries.get(
                    alpha_2=country_of_airport).name
                flights_for_each_country_country.append(country_name)
            except:
                if country_of_airport == "XK":
                    flights_for_each_country_country.append("Kosovo")
                else:
                    print("can't find country for code: ", country_of_airport)
                    flights_for_each_country_country.append("Unknown")
    except:
        airports_not_found.append(code)

len(airports_not_found)
# %%
flights_for_each_country = pd.DataFrame({
    "month": flights_for_each_country_month,
    "airport": flights_for_each_country_airport,
    "number_of_flights": flights_for_each_country_number,
    "country_code": flights_for_each_country_code,
    "country": flights_for_each_country_country
})

# Replace country "Unknown" with "Kosovo"
flights_for_each_country.loc[flights_for_each_country["country"]
                             == "Unknown", "country"] = "Kosovo"
flights_for_each_country.loc[flights_for_each_country["country"]
                             == "Czechia", "country"] = "Czech Republic"
flights_for_each_country.loc[flights_for_each_country["country"]
                             == "Slovakia", "country"] = "Slovak Republic"
flights_for_each_country.loc[flights_for_each_country["country"]
                             == "Russian Federation", "country"] = "Russia"
flights_for_each_country.loc[flights_for_each_country["country"]
                             == "Moldova, Republic of", "country"] = "Moldova"

# %%
# Get country name for each country code
countries_in_data = []
for code in country_codes_in_data:
    try:
        countries_in_data.append(pycountry.countries.get(alpha_2=code).name)
    except:
        if code == "XK":
            countries_in_data.append("Kosovo")
        else:
            print(code)
# %%

# Match the names of the countries in the data to those in the C5 data
c5 = pd.read_csv("covid_data/c5m_close_public_transport.csv")
euro_countries_in_flight_data_and_c5 = []
not_in_data = []
for country in np.unique(countries_in_data):
    if country not in c5["country_name"].unique():

        not_in_data.append(country)

        if country == "Czechia":
            euro_countries_in_flight_data_and_c5.append("Czech Republic")
        elif country == "Slovakia":
            euro_countries_in_flight_data_and_c5.append("Slovak Republic")
        elif country == "Russian Federation":
            euro_countries_in_flight_data_and_c5.append("Russia")
        elif country == "Moldova, Republic of":
            euro_countries_in_flight_data_and_c5.append("Moldova")
        else:
            print("no data for " + country)
    else:
        euro_countries_in_flight_data_and_c5.append(country)

# %%
# Map months to their three-letter abbreviations
month_to_abbrev = {
    "January": "Jan",
    "February": "Feb",
    "March": "Mar",
    "April": "Apr",
    "May": "May",
    "June": "Jun",
    "July": "Jul",
    "August": "Aug",
    "September": "Sep",
    "October": "Oct",
    "November": "Nov",
    "December": "Dec"
}
abbrev_to_month = {v: k for k, v in month_to_abbrev.items()}


# %%
euro_c5 = c5[c5["country_name"].isin(euro_countries_in_flight_data_and_c5)]

# remove row if region_code is not NaN (remove england, scotland, etc)
euro_c5 = euro_c5[pd.isnull(euro_c5["region_code"])]


total_flights_from_country = flights_for_each_country[[
    "country", "number_of_flights"]].groupby("country").sum().reset_index()

euro_c5["total_flights"] = [total_flights_from_country[total_flights_from_country["country"]
                                                       == country]["number_of_flights"].values[0] for country in euro_c5["country_name"]]

euro_c5_mat = euro_c5[[
    col for col in euro_c5.columns if "20" in col]].values

# binaryize
euro_c5_mat[euro_c5_mat > 0] = 1

# Weight the c5 of each country by the number of flights from that country
# weighted_c5 = euro_c5_mat * euro_c5["total_flights"].values[:, np.newaxis]
weighted_c5 = euro_c5_mat

weighted_c5_sum = np.sum(weighted_c5, axis=0)

weighted_c5_dates = [col for col in euro_c5.columns if "20" in col]

# Add 2019 months
months_2021 = weighted_c5_dates[weighted_c5_dates.index(
    "01Jan2021"):weighted_c5_dates.index("31Dec2021")+1]
months_2019 = [month.replace("2021", "2019") for month in months_2021]

weighted_c5_dates = months_2019 + weighted_c5_dates
weighted_c5_sum = np.concatenate(
    (np.zeros(len(months_2019)), weighted_c5_sum))

# Remove months after 31Dec2021
weighted_c5_sum = weighted_c5_sum[:weighted_c5_dates.index("01Jan2022")]
weighted_c5_dates = weighted_c5_dates[:weighted_c5_dates.index("01Jan2022")]

# %%
# Increase font size
plt.rcParams.update({'font.size': 14})


plt.figure(figsize=(15, 10))

weighted_c5_months = [abbrev_to_month[date[2:5]] +
                      " " + date[5:] for date in weighted_c5_dates]

# convert to date time using pandas
weighted_c5_dates_datetime = pd.to_datetime(
    weighted_c5_dates)
plt.plot(weighted_c5_dates_datetime, weighted_c5_sum, c="black")

# Manually set ticks
c5_tick_labels = []
for i in range(len(weighted_c5_months)):
    if str(weighted_c5_dates_datetime[i])[8:10] == "01":
        c5_tick_labels.append(i)

c5_tick_labels = np.array(c5_tick_labels)

plt.xticks(np.array(weighted_c5_dates_datetime)[c5_tick_labels],
           np.array(weighted_c5_months)[c5_tick_labels],
           rotation=90)


# plt.grid()

# Found using hierarchical clustering from the flight data script
cluster_1 = ["April 2020", "May 2020"]
cluster_2 = ["January 2021", "June 2020", "February 2021",
             "April 2021", "November 2020", "December 2020", "March 2021"]
cluster_5 = ["July 2021", "August 2021", "September 2021", "July 2019",
             "June 2019", "October 2021", "October 2019", "August 2019", "September 2019"]
cluster_3 = ["May 2021", "July 2020", "March 2020", "October 2020"]
cluster_4 = ["September 2020", "January 2019", "February 2019", "March 2019", "August 2020", "May 2019", "February 2020",
             "June 2021", "November 2021", "January 2020", "April 2019", "November 2019", "December 2019", "December 2021"]

# Convert each to date time
cluster_1 = pd.to_datetime(cluster_1)
cluster_2 = pd.to_datetime(cluster_2)
cluster_3 = pd.to_datetime(cluster_3)
cluster_4 = pd.to_datetime(cluster_4)
cluster_5 = pd.to_datetime(cluster_5)


colors = ["#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]

# plt.scatter(cluster_1, np.zeros((len(cluster_1),))+32,
#             label="COVID-19 Start", color=colors[0])
# plt.scatter(cluster_2, np.zeros((len(cluster_2),))+32,
#             label="Other COVID-19", color=colors[1])
# plt.scatter(cluster_3, np.zeros((len(cluster_3),)) +
#             32, label="Unaffected Summer", color=colors[2])
# plt.scatter(cluster_4, np.zeros((len(cluster_4),)) +
#             32, label="Partially Affected", color=colors[3])
# plt.scatter(cluster_5, np.zeros((len(cluster_5),))+32,
#             label="Unaffected Non-summer", color=colors[4])

# all_clusters = [cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]
# for i, cluster in enumerate(all_clusters):
#     # plot thick vertical line for each cluster
#     for date in cluster:
#         plt.axvline(date, color=colors[i], linewidth=1, alpha=1)

all_clusters = [cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]
datetime_labels = pd.to_datetime(labels)
for month_num, month in enumerate(datetime_labels):
    for clust_num, cluster in enumerate(all_clusters):
        if month in cluster:
            # plt.axvline(month, color=colors[clust_num], linewidth=1, alpha=1)

            # color area between months
            if month_num == 0:
                plt.axvspan(
                    month, datetime_labels[month_num+1], color=colors[clust_num], alpha=0.5)
            elif month_num == len(labels)-1:
                # For December, go up until new year
                plt.axvspan(month,
                            pd.to_datetime("January 2022"), color=colors[clust_num], alpha=0.5)
            else:
                plt.axvspan(
                    datetime_labels[month_num], datetime_labels[month_num+1], color=colors[clust_num], alpha=0.5)


plt.xlabel("Date")
plt.ylabel("Number of European Countries with Public Transport Regulations")


# Define legend for the axvspans
legend_elements = [Patch(facecolor=colors[0], edgecolor='black', label='COVID-19 Start', alpha=0.5),
                   Patch(
                       facecolor=colors[1], edgecolor='black', label='Other COVID-19', alpha=0.5),
                   Patch(facecolor=colors[2], edgecolor='black',
                         label='Partially Affected', alpha=0.5),
                   Patch(facecolor=colors[3], edgecolor='black',
                         label='Unaffected Non-summer', alpha=0.5),
                   Patch(facecolor=colors[4], edgecolor='black',
                         label='Unaffected Summer', alpha=0.5), ]
# Create the figure
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.13),
           fancybox=False, shadow=False, ncol=5, title="Clustering of Network Embedding")

# Add annotations
plt.annotate("COVID-19 Start", xy=(pd.to_datetime("April 2020"), 31), xytext=(pd.to_datetime("April 2020"), 34.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
plt.annotate("Second COVID-19 Wave", xy=(pd.to_datetime("November 2020"), 17), xytext=(pd.to_datetime("November 2020"), 20),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")

# Add label "normal periodic behaviour" with arrows pointing from the label to January 2019, June 2019 and November 2019
plt.annotate("", xy=(pd.to_datetime("January 2019"), 31), xytext=(pd.to_datetime("April 2019"), 34),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
plt.annotate("Normal Periodic Behaviour", xy=(pd.to_datetime("June 2019"), 31), xytext=(pd.to_datetime("June 2019"), 34.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
plt.annotate("", xy=(pd.to_datetime("November 2019"), 31), xytext=(pd.to_datetime("August 2019"), 34),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")

# Save again but "Periodic Behaviour Resumes" with arrows pointing from the label to July 2021 and November 2021
plt.annotate("Periodic Behaviour Resumes", xy=(pd.to_datetime("July 2021"), 31), xytext=(pd.to_datetime("September 2021"), 34.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
plt.annotate("", xy=(pd.to_datetime("November 2021"), 31), xytext=(pd.to_datetime("September 2021"), 34.18),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")

# "Partially Affected Months on the Edge of COVID-19 Waves"
plt.annotate("Partially Affected Months\non the Edge of COVID-19 Waves", xy=(pd.to_datetime("July 2020"), 10), xytext=(pd.to_datetime("September 2020"), 2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
# Second arrow pointing from the label to October 2020
plt.annotate("", xy=(pd.to_datetime("November 2020"), 10), xytext=(pd.to_datetime("September 2020"), 3.93),
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")


# set ylim
plt.ylim(0, 36)


# Save figure as pdf
# plt.savefig("saved_flight_stuff/hierarchical_clustering_on_C5_data.pdf", bbox_inches='tight')


# %%

# %%
#############################
######## Main plot ##########
#############################


# increase font size
plt.rcParams.update({'font.size': 14})

# curvwe ofnnumber of lockdowns?
diss_matrix = np.load(
    "saved_flight_stuff/URLSE_EU_dissimilarity_matrix_d=5_multigraph.npy")
p_val_matrix = np.load("saved_flight_stuff/URLSE_EU_matrix_d=5.npy")

# Apply Bonferroni correction to the p-value matrix and control type 1 error at 5%
T = 36
thresh = 0.05 / comb(T, 2)
p_val_matrix = ~ (p_val_matrix > thresh)

mat_to_show = diss_matrix

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
labels = np.core.defchararray.add(np.array(months * 3),
                                  np.repeat([" 2019", " 2020", " 2021"], 12))

# convert to datetime
# dates = pd.to_datetime(labels, format="%B %Y")
dates = pd.to_datetime(labels, format="%B %Y").strftime("%B %Y")

# remove september 2020
# mat_to_show = np.delete(mat_to_show, 20, axis=0)
# mat_to_show = np.delete(mat_to_show, 20, axis=1)
tick_locations = np.arange(36)[0::2]
# for i, loc in enumerate(tick_locations):
#     if loc > 20:
#         tick_locations[i] = loc - 1

# tick_locations = np.delete(tick_locations, 10, axis=0)
tick_labels = dates[0::2]
# tick_labels = np.delete(tick_labels, 10, axis=0)

# tick_labels = np.where(tick_labels == "September2020", "", tick_labels)

fig, ax = plt.subplots(2, 2, figsize=(12, 15), sharey=False, sharex=False,
                       gridspec_kw={'width_ratios': [3, 1]}, constrained_layout=True)

# have subplots share the x axis
# fig.subplots_adjust(hspace=0)
# ax[0].get_shared_x_axes().join(ax[0], ax[1])


cax = ax[0, 0].matshow(mat_to_show)
_ = ax[0, 0].set_xticks(tick_locations, tick_labels, rotation=90)

# hide x axis labels

_ = ax[0, 0].set_yticks(tick_locations, tick_labels, rotation=0)
_ = ax[0, 0].xaxis.set_ticks_position('top')
# plt.setp(ax[0,0].get_xticklabels(), visible=False)
_ = ax[0, 0].grid()

# Add colourbar above the subplot
cbar = fig.colorbar(
    cax, ax=ax[1, 0], orientation='horizontal', location="top", shrink=0.6598, anchor=(0.9837, 4.3))
cbar.ax.set_xlabel("Dissimilarity")

# Set colorbar ticks to be below
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.xaxis.set_label_position('bottom')

# get rid of outer box
for edge, spine in ax[0, 0].spines.items():
    spine.set_visible(False)


# C5
weighted_c5_months = [date[0:2] + " " + abbrev_to_month[date[2:5]] +
                      " " + date[5:] for date in weighted_c5_dates][::-1]

# # convert to date time
# weighted_c5_months = pd.to_datetime(
#     weighted_c5_months, format="%d %B %Y").strftime("%d %B %Y")

# repeated_months = []
# for month in weighted_c5_months:
#     if month[0:2] == "01":
#         month = " ".join(month.split(" ")[1:])
#         repeated_months.append(month)
#     else:
#         repeated_months.append(month)

# _ = ax[0, 1].plot(weighted_c5_sum, repeated_months)
_ = ax[0, 1].plot(weighted_c5_sum, weighted_c5_months)
ax[0, 1].set_xlabel(
    "Number of European\nCountries with Public\nTransport Restrictions")

# Manually set ticks
c5_tick_labels = []
for i in range(len(weighted_c5_months)):
    if weighted_c5_months[i][0:2] == "01":
        c5_tick_labels.append(i)

# ax[0, 1].set_ylim(ax[0, 0].get_ylim())

# yticks = ax[0, 0].get_yticks() - 0.1
# ax[0, 1].set_yticks(yticks)

_ = ax[0, 1].set_yticks(np.array(c5_tick_labels[::2]),
                        np.array(weighted_c5_months)[c5_tick_labels][::-1][::2], rotation=0)
# _ = ax[0, 1].set_yticks(np.arange(0, len(weighted_c5_months))[0::20],
#                         weighted_c5_months[0::20][::-1], rotation=0)

# Hide tick labels
_ = ax[0, 1].set_yticklabels([])

# set y limits
ax[0, 1].set_ylim((1110, 14))


ax[0, 1].grid()


# hide subplot 1,1
_ = ax[1, 1].axis('off')


_ = ax[1, 0].grid()
_ = ax[1, 0].matshow(p_val_matrix)
_ = ax[1, 0].set_xticks(tick_locations, tick_labels, rotation=90)
_ = ax[1, 0].set_yticks(tick_locations, tick_labels, rotation=0)
_ = ax[1, 0].xaxis.set_ticks_position('top')

cmap = matplotlib.cm.get_cmap('viridis')
col_1 = cmap(0)
col_2 = cmap(255)
legend_elements = [Patch(facecolor=col_1,
                         label='Not Significant'),
                   Patch(facecolor=col_2,
                         label='Significant'),
                   ]

# Create the figure
ax[1, 1].legend(handles=legend_elements, loc="upper right",
                title="Dissimilarity Significance", fancybox=True, frameon=False, ncol=1, bbox_to_anchor=(0.92, 1.03))


# reduce distance between subplot columns
plt.subplots_adjust(wspace=-0.35)
plt.subplots_adjust(hspace=0.7)


# plt.tight_layout()

# Set background color
# fig.patch.set_facecolor('white')

# plt.gca().set_axis_off()
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
#             hspace = 0, wspace = 0)
# plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())

# save
# plt.savefig("saved_flight_stuff/eu_matrix_with_covid_regulations.pdf",
#             dpi=300, bbox_inchdes='tight', pad_inches=0)

# %%
################################
#### With automatic clusters ###
################################
kmeans = KMeans(n_clusters=5, random_state=0).fit(p_val_matrix)
kmeansdf = pd.DataFrame({
    "date": labels,
    "cluster": kmeans.labels_
})
# %%


# Increase font size
plt.rcParams.update({'font.size': 14})


plt.figure(figsize=(15, 10))

weighted_c5_months = [abbrev_to_month[date[2:5]] +
                      " " + date[5:] for date in weighted_c5_dates]

# convert to date time using pandas
weighted_c5_dates_datetime = pd.to_datetime(
    weighted_c5_dates)
plt.plot(weighted_c5_dates_datetime, weighted_c5_sum, c="black")

# Manually set ticks
c5_tick_labels = []
for i in range(len(weighted_c5_months)):
    if str(weighted_c5_dates_datetime[i])[8:10] == "01":
        c5_tick_labels.append(i)

c5_tick_labels = np.array(c5_tick_labels)

plt.xticks(np.array(weighted_c5_dates_datetime)[c5_tick_labels],
           np.array(weighted_c5_months)[c5_tick_labels],
           rotation=90)


# plt.grid()

# Found using hierarchical clustering from the flight data script
# cluster_1 = ["April 2020", "May 2020"]
# cluster_2 = ["January 2021", "June 2020", "February 2021",
#              "April 2021", "November 2020", "December 2020", "March 2021"]
# cluster_5 = ["July 2021", "August 2021", "September 2021", "July 2019",
#              "June 2019", "October 2021", "October 2019", "August 2019", "September 2019"]
# cluster_3 = ["May 2021", "July 2020", "March 2020", "October 2020"]
# cluster_4 = ["September 2020", "January 2019", "February 2019", "March 2019", "August 2020", "May 2019", "February 2020",
#              "June 2021", "November 2021", "January 2020", "April 2019", "November 2019", "December 2019", "December 2021"]


cluster_1 = kmeansdf[kmeansdf["cluster"] == 0]["date"].values
cluster_2 = kmeansdf[kmeansdf["cluster"] == 1]["date"].values
cluster_3 = kmeansdf[kmeansdf["cluster"] == 2]["date"].values
cluster_4 = kmeansdf[kmeansdf["cluster"] == 3]["date"].values
cluster_5 = kmeansdf[kmeansdf["cluster"] == 4]["date"].values

cluster_label = ["0", "1", "2", "3", "4"]

# Convert each to date time
cluster_1 = pd.to_datetime(cluster_1)
cluster_2 = pd.to_datetime(cluster_2)
cluster_3 = pd.to_datetime(cluster_3)
cluster_4 = pd.to_datetime(cluster_4)
cluster_5 = pd.to_datetime(cluster_5)


colors = ["#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8", "#253494"]

# plt.scatter(cluster_1, np.zeros((len(cluster_1),))+32,
#             label="COVID-19 Start", color=colors[0])
# plt.scatter(cluster_2, np.zeros((len(cluster_2),))+32,
#             label="Other COVID-19", color=colors[1])
# plt.scatter(cluster_3, np.zeros((len(cluster_3),)) +
#             32, label="Unaffected Summer", color=colors[2])
# plt.scatter(cluster_4, np.zeros((len(cluster_4),)) +
#             32, label="Partially Affected", color=colors[3])
# plt.scatter(cluster_5, np.zeros((len(cluster_5),))+32,
#             label="Unaffected Non-summer", color=colors[4])

# all_clusters = [cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]
# for i, cluster in enumerate(all_clusters):
#     # plot thick vertical line for each cluster
#     for date in cluster:
#         plt.axvline(date, color=colors[i], linewidth=1, alpha=1)

all_clusters = [cluster_1, cluster_2, cluster_3, cluster_4, cluster_5]
datetime_labels = pd.to_datetime(labels)
for month_num, month in enumerate(datetime_labels):
    for clust_num, cluster in enumerate(all_clusters):
        if month in cluster:
            # plt.axvline(month, color=colors[clust_num], linewidth=1, alpha=1)

            # color area between months
            if month_num == 0:
                plt.axvspan(
                    month, datetime_labels[month_num+1], color=colors[clust_num], alpha=0.5)

                # label the cluster above the oclor area
                plt.text(
                    month, 32, cluster_label[clust_num], color=colors[clust_num])

            elif month_num == len(labels)-1:
                # For December, go up until new year
                plt.axvspan(month,
                            pd.to_datetime("January 2022"), color=colors[clust_num], alpha=0.5)

                # label the cluster above the oclor area
                plt.text(
                    month, 32, cluster_label[clust_num], color=colors[clust_num])

            else:
                plt.axvspan(
                    datetime_labels[month_num], datetime_labels[month_num+1], color=colors[clust_num], alpha=0.5)

                # label the cluster above the oclor area
                plt.text(
                    month, 32, cluster_label[clust_num], color=colors[clust_num])


plt.xlabel("Date")
plt.ylabel("Number of European Countries with Public Transport Regulations")


# Define legend for the axvspans
legend_elements = [Patch(facecolor=colors[0], edgecolor='black', label=cluster_label[0], alpha=0.5),
                   Patch(
                       facecolor=colors[1], edgecolor='black', label=cluster_label[1], alpha=0.5),
                   Patch(facecolor=colors[2], edgecolor='black',
                         label=cluster_label[2], alpha=0.5),
                   Patch(facecolor=colors[3], edgecolor='black',
                         label=cluster_label[3], alpha=0.5),
                   Patch(facecolor=colors[4], edgecolor='black',
                         label=cluster_label[4], alpha=0.5), ]
# Create the figure
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.13),
           fancybox=False, shadow=False, ncol=5, title="Clustering of Network Embedding")

# # Add annotations
# plt.annotate("COVID-19 Start", xy=(pd.to_datetime("April 2020"), 31), xytext=(pd.to_datetime("April 2020"), 34.5),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
# plt.annotate("Second COVID-19 Wave", xy=(pd.to_datetime("November 2020"), 17), xytext=(pd.to_datetime("November 2020"), 20),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")

# # Add label "normal periodic behaviour" with arrows pointing from the label to January 2019, June 2019 and November 2019
# plt.annotate("", xy=(pd.to_datetime("January 2019"), 31), xytext=(pd.to_datetime("April 2019"), 34),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
# plt.annotate("Normal Periodic Behaviour", xy=(pd.to_datetime("June 2019"), 31), xytext=(pd.to_datetime("June 2019"), 34.5),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
# plt.annotate("", xy=(pd.to_datetime("November 2019"), 31), xytext=(pd.to_datetime("August 2019"), 34),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")

# # Save again but "Periodic Behaviour Resumes" with arrows pointing from the label to July 2021 and November 2021
# plt.annotate("Periodic Behaviour Resumes", xy=(pd.to_datetime("July 2021"), 31), xytext=(pd.to_datetime("September 2021"), 34.5),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
# plt.annotate("", xy=(pd.to_datetime("November 2021"), 31), xytext=(pd.to_datetime("September 2021"), 34.18),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")

# # "Partially Affected Months on the Edge of COVID-19 Waves"
# plt.annotate("Partially Affected Months\non the Edge of COVID-19 Waves", xy=(pd.to_datetime("July 2020"), 10), xytext=(pd.to_datetime("September 2020"), 2),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")
# # Second arrow pointing from the label to October 2020
# plt.annotate("", xy=(pd.to_datetime("November 2020"), 10), xytext=(pd.to_datetime("September 2020"), 3.93),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8), fontsize=14, ha="center")


# set ylim
plt.ylim(0, 36)


# Save figure as pdf
# plt.savefig("saved_flight_stuff/hierarchical_clustering_on_C5_data.pdf",
#             bbox_inches='tight')

# %%
