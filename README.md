# Universal Stable Dynamic Graph Embedding with Robust Hypothesis Testing

This repository implements a collection of spectral and skip-gram dynamic network embedding methods. This repo compares how well different methods can encode planted structure in simulated dynamic networks through the use of a hypothesis test. The methods are also applied to a dynamic flight network to see whether they can encode both the periodic nature of the network, as well as represent the disruption caused by the COVID-19 pandemic on air traffic.

## Getting Started

This code has been tested using python 3.8.10. To install the required packages for this repo use the following command.

```pip install -r requirements.txt```

We implement the [GloDyNE](https://ieeexplore.ieee.org/abstract/document/9302718) embedding, which can be found at the [following](https://github.com/houchengbin/GloDyNE) GitHub.

## Data

We consider the following datasets in this analysis.

- The [OpenSky](https://zenodo.org/record/5815448#.Y1_ydy-l1hD) network, which is a 17,388 node dynamic network of flights which occurred from the start of 2019 to the end of 2021.
- ```airports.csv``` from [OurAirports](https://ourairports.com/data/), which contains information about airports around the world.
- The [Oxford COVID-19 Government Response Tracker](https://github.com/OxCGRT/covid-policy-tracker), from which we use the [C5 regulations](https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/timeseries/c5m_close_public_transport.csv) data which lists the travel restrictions in place per country in response to the COVID-19 pandemic.

## Dynamic Embedding Methods

This repository implements a collection of spectral and skip-gram dynamic embedding methods. Here, we consider the problem of embedding discrete-time dynamic networks, i.e. those that can be represented as a series of adjacency matrix ``snapshots" over time, $\mathbf{A}^{(1)}, \dots, \mathbf{A}^{(T)}$. A dynamic embedding is then a low-dimensional representation for each of snapshots in the series, $\hb{Y}^{(1)}, \dots, \hb{Y}^{(T)} \in \R^{n \times d}$, which we refer to as embedding time points.

### Spectral
- Independent spectral embedding (ISE): Each $\hb{Y}^{(t)}$ is the independent spectral embedding of each $\mathbf{A}^{(t)}$. Here, we add the option to ensure that the eigenvector orientation is consisten to remove random flipping between embedding time points. We additioanlly add the option of aligning subsequent embedding time points via a procrustes rotation.
- [Omnibus embedding (OMNI)](https://arxiv.org/abs/1705.09355): Constructs a single block matrix containing pairwise-averaged adjacency snapshots, $\mathbf{M}_{s,t} = (\mathbf{A}^{(s)} + \mathbf{A}^{(t)})/2$, and then computes a spectral embedding on this matrix to achieve a dynamic embedding [1].
- [Unfolded adjacency spectral embedding (UASE)](https://arxiv.org/abs/2007.10455): Constructs a column-concatenated adjacency matrix, $\mathcal{A} = \left[\mathbf{A}^{(1)}, \dots, \mathbf{A}^{(T)} \right]$, called the rectangular unfolded adjacency matrix. The dynamic embedding is then achieved through an right SVD embedding [2,3].

# References
[1] Keith Levin, Avanti Athreya, Minh Tang, Vince Lyzinski, Youngser Park, and Carey E
Priebe. A central limit theorem for an omnibus embedding of multiple random graphs
and implications for multiscale network inference. arXiv preprint arXiv:1705.09355, 2017.
```@article{omni,
  title={A central limit theorem for an omnibus embedding of multiple random graphs and implications for multiscale network inference},
  author={Levin, Keith and Athreya, Avanti and Tang, Minh and Lyzinski, Vince and Park, Youngser and Priebe, Carey E},
  journal={arXiv preprint arXiv:1705.09355},
  year={2017}
}```
[2] Ian Gallagher, Andrew Jones, and Patrick Rubin-Delanchy. Spectral embedding for dynamic
networks with stability guarantees. Advances in Neural Information Processing Systems,
34:10158–10170, 2021.

[3]
