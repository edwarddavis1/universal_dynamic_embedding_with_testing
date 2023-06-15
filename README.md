# Universal Stable Dynamic Graph Embedding with Robust Hypothesis Testing


This repository contains code for analyzing flight data and C5 regulations in Europe during the COVID-19 pandemic. The code is written in Python and uses various libraries such as pandas, numpy, and pycountry.

## Getting Started

This code has been tested using python 3.8.10. To install the required packages for this repo use the following command.

```pip install -r requirements.txt```

We implement the [GloDyNE](https://ieeexplore.ieee.org/abstract/document/9302718) embedding, which can be found at the [following](https://github.com/houchengbin/GloDyNE) GitHub.

## Data

The data used in this analysis comes from various sources, including:

- Flight data from [OpenFlights](https://openflights.org/data.html)
- C5 regulations data from [ECDC](https://www.ecdc.europa.eu/en/publications-data/download-data-response-measures-covid-19)

The flight data is stored in a CSV file called `airports.csv`, which contains information about airports around the world. The C5 regulations data is stored in a CSV file called `c5m_close_public_transport.csv`, which contains information about transport restrictions in various countries.

