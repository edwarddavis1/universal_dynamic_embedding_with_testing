# %%
"""C5 Data from: https://github.com/OxCGRT/covid-policy-tracker"""
import pycountry
from matplotlib.patches import Patch
import matplotlib
from math import comb
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import pandas as pd
from embedding_functions import *

# %%
datapath = "Flight Data/"

# Load As
As = []
T = 36
for t in range(T):
    As.append(sparse.load_npz(datapath + "As_" + str(t) + ".npz"))

# Load Z
Z = np.load(datapath + "Z.npy")
nodes = np.load(datapath + "nodes.npy", allow_pickle=True).item()

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
labels = np.core.defchararray.add(
    np.array(months * 3), np.repeat([" 2019", " 2020", " 2021"], 12)
)
# Get european nodes
cont_to_look_at = "EU"
euro_nodes = np.where(np.array(Z) == cont_to_look_at)
# %%
# Get data on each airport for each month

# airports.csv from https://ourairports.com/data/
airports = pd.read_csv(datapath + "airports.csv")

country_codes_in_data = []
airports_not_found = []
flights_for_each_country_month = []
flights_for_each_country_number = []
flights_for_each_country_airport = []
flights_for_each_country_code = []
flights_for_each_country_country = []

# For each european airport in the flight network, get the number of flights in each time period
for code in np.array(list(nodes.keys()))[euro_nodes]:
    try:
        country_of_airport = airports[airports["ident"] == code]["iso_country"].values[
            0
        ]
        country_codes_in_data.append(country_of_airport)

        for t in range(T):
            flights_for_each_country_month.append(labels[t])
            flights_for_each_country_number.append(np.sum(As[t][:, nodes[code]]))
            flights_for_each_country_airport.append(code)
            flights_for_each_country_code.append(country_of_airport)
            try:
                country_name = pycountry.countries.get(alpha_2=country_of_airport).name
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
flights_for_each_country = pd.DataFrame(
    {
        "month": flights_for_each_country_month,
        "airport": flights_for_each_country_airport,
        "number_of_flights": flights_for_each_country_number,
        "country_code": flights_for_each_country_code,
        "country": flights_for_each_country_country,
    }
)

# Consistent Naming
flights_for_each_country.loc[
    flights_for_each_country["country"] == "Unknown", "country"
] = "Kosovo"
flights_for_each_country.loc[
    flights_for_each_country["country"] == "Czechia", "country"
] = "Czech Republic"
flights_for_each_country.loc[
    flights_for_each_country["country"] == "Slovakia", "country"
] = "Slovak Republic"
flights_for_each_country.loc[
    flights_for_each_country["country"] == "Russian Federation", "country"
] = "Russia"
flights_for_each_country.loc[
    flights_for_each_country["country"] == "Moldova, Republic of", "country"
] = "Moldova"
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

# C5 data (transport restriction data)
# 0 - no measures
# 1 - recommend closing (or significantly reduce volume/route/means of transport available)
# 2 - require closing (or prohibit most citizens from using it)
# Blank - no data
c5 = pd.read_csv("covid_data/c5m_close_public_transport.csv")

# Match the names of the countries in the c5 data
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
    "December": "Dec",
}
abbrev_to_month = {v: k for k, v in month_to_abbrev.items()}


# %%
# Select european countries (in the flight netowrk) from the C5 data
euro_c5 = c5[c5["country_name"].isin(euro_countries_in_flight_data_and_c5)]

# remove row if region_code is not NaN (remove england, scotland, etc)
euro_c5 = euro_c5[pd.isnull(euro_c5["region_code"])]

# Get the total number of flights from each country
total_flights_from_country = (
    flights_for_each_country[["country", "number_of_flights"]]
    .groupby("country")
    .sum()
    .reset_index()
)
euro_c5["total_flights"] = [
    total_flights_from_country[total_flights_from_country["country"] == country][
        "number_of_flights"
    ].values[0]
    for country in euro_c5["country_name"]
]

# Hacky way of selecting columns with a date
euro_c5_mat = euro_c5[[col for col in euro_c5.columns if "20" in col]].values

# binaryize as we only care about if there are restrictions or not
euro_c5_mat[euro_c5_mat > 0] = 1

# Get the number of european countries with restrictions for each day
euro_c5_sum = np.sum(euro_c5_mat, axis=0)
euro_c5_dates = [col for col in euro_c5.columns if "20" in col]

# Add 2019 months
months_2021 = euro_c5_dates[
    euro_c5_dates.index("01Jan2021") : euro_c5_dates.index("31Dec2021") + 1
]
months_2019 = [month.replace("2021", "2019") for month in months_2021]
euro_c5_dates = months_2019 + euro_c5_dates
euro_c5_sum = np.concatenate((np.zeros(len(months_2019)), euro_c5_sum))

# Remove months after 31Dec2021
# Now have the number of european countries with restrictions for each day between 01Jan2019 and 31Dec2021
euro_c5_sum = euro_c5_sum[: euro_c5_dates.index("01Jan2022")]
euro_c5_dates = euro_c5_dates[: euro_c5_dates.index("01Jan2022")]

# %%
#############################
######## Main plot ##########
#############################


# increase font size
plt.rcParams.update({"font.size": 14})

# Load dissimilarity and p-value matrices
diss_matrix = np.load("saved_flight_matrices/URLSE_EU_matrix.npy")
p_val_matrix = np.load("saved_flight_matrices/URLSE_EU_p_value_matrix.npy")

# Apply Bonferroni correction to the p-value matrix and control type 1 error at 5%
T = 36
thresh = 0.05 / comb(T, 2)
p_val_matrix = ~(p_val_matrix > thresh)

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
labels = np.core.defchararray.add(
    np.array(months * 3), np.repeat([" 2019", " 2020", " 2021"], 12)
)

# convert to datetime
dates = pd.to_datetime(labels, format="%B %Y").strftime("%B %Y")


tick_locations = np.arange(36)[0::2]
tick_labels = dates[0::2]
fig, ax = plt.subplots(
    2,
    2,
    figsize=(12, 15),
    sharey=False,
    sharex=False,
    gridspec_kw={"width_ratios": [3, 1]},
    constrained_layout=True,
)
cax = ax[0, 0].matshow(diss_matrix)
_ = ax[0, 0].set_xticks(tick_locations, tick_labels, rotation=90)
_ = ax[0, 0].set_yticks(tick_locations, tick_labels, rotation=0)
_ = ax[0, 0].xaxis.set_ticks_position("top")
_ = ax[0, 0].grid()

# Add colourbar above the subplot
cbar = fig.colorbar(
    cax,
    ax=ax[1, 0],
    orientation="horizontal",
    location="top",
    shrink=0.6598,
    anchor=(0.9837, 4.3),
)
cbar.ax.set_xlabel("Dissimilarity")

# Set colorbar ticks to be below
cbar.ax.xaxis.set_ticks_position("bottom")
cbar.ax.xaxis.set_label_position("bottom")

# get rid of outer box
for edge, spine in ax[0, 0].spines.items():
    spine.set_visible(False)

# C5 plot (match x-axis with that of the dissimilarity plot)
euro_c5_months = [
    date[0:2] + " " + abbrev_to_month[date[2:5]] + " " + date[5:]
    for date in euro_c5_dates
][::-1]
_ = ax[0, 1].plot(euro_c5_sum, euro_c5_months)
ax[0, 1].set_xlabel("Number of European\nCountries with Public\nTransport Restrictions")

# Manually set ticks
c5_tick_labels = []
for i in range(len(euro_c5_months)):
    if euro_c5_months[i][0:2] == "01":
        c5_tick_labels.append(i)


_ = ax[0, 1].set_yticks(
    np.array(c5_tick_labels[::2]),
    np.array(euro_c5_months)[c5_tick_labels][::-1][::2],
    rotation=0,
)
_ = ax[0, 1].set_yticklabels([])
_ = ax[0, 1].set_ylim((1110, 14))
_ = ax[0, 1].grid()
_ = ax[1, 1].axis("off")
_ = ax[1, 0].grid()
_ = ax[1, 0].matshow(p_val_matrix)
_ = ax[1, 0].set_xticks(tick_locations, tick_labels, rotation=90)
_ = ax[1, 0].set_yticks(tick_locations, tick_labels, rotation=0)
_ = ax[1, 0].xaxis.set_ticks_position("top")

cmap = matplotlib.cm.get_cmap("viridis")
col_1 = cmap(0)
col_2 = cmap(255)
legend_elements = [
    Patch(facecolor=col_1, label="Not Significant"),
    Patch(facecolor=col_2, label="Significant"),
]

ax[1, 1].legend(
    handles=legend_elements,
    loc="upper right",
    title="Dissimilarity Significance",
    fancybox=True,
    frameon=False,
    ncol=1,
    bbox_to_anchor=(0.92, 1.03),
)


# reduce distance between subplot columns
_ = plt.subplots_adjust(wspace=-0.35)
_ = plt.subplots_adjust(hspace=0.7)

plt.gca().xaxis.set_major_locator(plt.NullLocator())

# save
plt.savefig(
    "saved_flight_plots/eu_matrix_with_covid_regulations.pdf",
    dpi=300,
    # bbox_inchdes="tight",
    pad_inches=0,
)

# %%
