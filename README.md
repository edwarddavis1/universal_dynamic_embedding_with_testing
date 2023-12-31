# Universal Stable Dynamic Graph Embedding with Robust Hypothesis Testing

This repository contains methods an experiments from the paper: [A Simple and Powerful Framework for Stable Dynamic Network Embedding](https://arxiv.org/abs/2311.09251). 

This repository implements a collection of spectral and skip-gram dynamic network embedding methods. This repo compares how well different methods can encode planted structure in simulated dynamic networks through the use of a hypothesis test. The methods are also applied to a dynamic flight network to see whether they can encode both the periodic nature of the network, as well as represent the disruption caused by the COVID-19 pandemic on air traffic.

## Getting Started

This code has been tested using Python 3.8.10. To install the required packages for this repo use the following command.

```
pip install -r requirements.txt
```

We also include the [GloDyNE](https://github.com/houchengbin/GloDyNE) embedding, for which the [METIS package](https://github.com/networkx/networkx-metis) is required. If you do not want to install the METIS package, you can use the flag ```--no-glodyne``` when running experiments to avoid using GloDyNE.



## Data

We consider the following datasets in this analysis.

- The [OpenSky](https://zenodo.org/record/5815448#.Y1_ydy-l1hD) network, which is a 17,388 node dynamic network of flights which occurred from the start of 2019 to the end of 2021.
- ```airports.csv``` from [OurAirports](https://ourairports.com/data/), which contains information about airports around the world.
- The [C5 regulations](https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/timeseries/c5m_close_public_transport.csv) data, from the [Oxford COVID-19 Government Response Tracker](https://github.com/OxCGRT/covid-policy-tracker). This data appears as ```c5m_close_public_transport.csv```. This lists the travel restrictions in place per country in response to the COVID-19 pandemic.


## Scripts
### Dynamic Embedding Functions:
- ```embedding_functions.py```: Contains functions for several spectral and skip-gram dynamic embedding methods (listed below). Also contains functions to perform hypothesis testing on a dynamic embedding.

#### Example Usage
```python
>>> from embedding_functions import *
>>> from experiment_setup import *
>>> 
>>> # Generate random dynamic network where community one moves half way through
>>> n = 100
>>> T = 10
>>> As, tau, _ = make_temporal_simple(n=n, T=T, move_prob=0.9)
>>> 
>>> # Compute a dynamic embedding
>>> ya = embed(As, d=2, method="URLSE")
>>> 
>>> ya.shape
(1000, 2)
```

### Simulated experiments and testing
- ```experiment_setup.py```: Contains functions to generate various dynamic networks with planted structure. 
- ```experiments.py```: Checks if various dynamic embedding methods can encode planted structure through the use of a hypothesis test. For each, a cumulative p-value distribution is plot and coloured based on if the dynamic embedding correctly encoded the planted structure or not. The plots are coloured green if correct, red if incorrect and blue if correct but the test was conservative.

#### Example Usage
```
python experiments.py --methods "URLSE" --experiments "moving-static-community" --check-types "community"

> URLSE had power 0.035

python experiments.py --methods "URLSE" --experiments "moving-community" --check-types "graph"

> URLSE had power 0.57

```

```
optional arguments:
  -h, --help            show this help message and exit
  --methods METHODS [METHODS ...]
                        List of dynamic embedding methods to run. Select from:
                        URLSE, UASE, OMNI, ISE, ISE Procrustes, GloDyNE,
                        Unfolded Node2Vec, Independent Node2Vec
  --experiments EXPERIMENTS [EXPERIMENTS ...]
                        Select which dynamic network systems to test on.
                        Select from: static, moving-static-community, moving-
                        community, merge, static-spatial, power-moving, power-
                        static
  --check-types CHECK_TYPES [CHECK_TYPES ...]
                        Run experiments at the community, graph, or node level
  --n-runs N_RUNS       Number of p-values to compute for each test
  --n N                 Number of nodes in each (non-power) experiment
  --n-power N_POWER     Number of nodes in each power experiment (reccomend at
                        least 2000)
  --all                 Runs all experiments with all methods at all check
                        types.
  --no-save             Use to bypass saving an experiment.
  --no-plots            Use to bypass plotting an experiment.
  --plot-only           Plot the result of a previously saved experiment.
                        Don't compute another test.
  --no-glodyne          In case you don't have the METIS package installed,
                        you can run the code without the GloDyNE method
                        (overrides the --all flag).
```


### Flight Network Analysis
- ```prep_flight_data.py```: Generates a sparse adjacency series and labelled node set from the [OpenSky](https://zenodo.org/record/5815448#.Y1_ydy-l1hD) network. Note that we have not included the raw ```flightlist_.csv``` files due to their size, but the sparse adjacency series that are generated by this script are small enough to include.
- ```flight_network_analysis.py```: Computes a dynamic embedding of the [OpenSky](https://zenodo.org/record/5815448#.Y1_ydy-l1hD) network. Follow-up analysis on this embedding includes the generation of dissimilarity and p-value matrices to study the encoded temporal structure of the network and hierarchical clustering. 
- ```flight_and_c5_regulations.py```: Produces a neat plot of the dissimilarity and p-value matrices to compare the encoded temporal structure of the network to a plot of travel restrictions in Europe in response to the COVID-19 pandemic. There are two waves of dissimilarity in the dissimilarity matrix which line up with the two main European waves of the pandemic.

## Dynamic Embedding Methods

This repository implements a collection of spectral and skip-gram dynamic embedding methods. Here, we consider the problem of embedding discrete-time dynamic networks, i.e. those that can be represented as a series of adjacency matrix ``snapshots" over time, $\mathbf{A}^{(1)}, \dots, \mathbf{A}^{(T)}$. A dynamic embedding is then a low-dimensional representation for each of the snapshots in the series, $\hat{\mathbf{Y}}^{(1)}, \dots, \hat{\mathbf{Y}}^{(T)} \in \mathbb{R}^{n \times d}$, which we refer to as embedding time points. Methods in **bold** are stable dynamic embedding methods.

### Spectral Embedding Methods
- [**Unfolded adjacency spectral embedding (UASE)**](https://arxiv.org/abs/2007.10455): Constructs a column-concatenated adjacency matrix, $\mathcal{A} = \left[\mathbf{A}^{(1)}, \dots, \mathbf{A}^{(T)} \right]$, called the rectangular unfolded adjacency matrix. The dynamic embedding is then achieved through the right SVD embedding [1,2].
- **Unfolded Regularised Laplacian Spectral Embedding (URLSE)**: computes a spectral embedding the matrix, 

```math
\begin{equation}\mathbf{L} = \left(\mathbf{D}_{\text{L}} - \gamma \right)^{-1/2} \mathcal{A} \left(\mathbf{D}_{\text{R}} - \gamma \right)^{-1/2} \end{equation}
```

where  
```math
\mathbf{D}_{\text{L}} = \text{diag}\left(\sum_i^{n}{\mathcal{A}^{\top}_i}\right), \mathbf{D}_{\text{R}} = \text{diag}\left(\sum_i^{nT}{\mathcal{A}_i}\right),
```
are the left and right degree matrices and $\gamma$ is a regularisation parameter.
- [Omnibus embedding (OMNI)](https://arxiv.org/abs/1705.09355): Constructs a single block matrix containing pairwise-averaged adjacency snapshots, $\mathbf{M}_{s,t} = (\mathbf{A}^{(s)} + \mathbf{A}^{(t)})/2$, and then computes a spectral embedding on this matrix to achieve a dynamic embedding [3].
- Independent spectral embedding (ISE): Each $\hat{\mathbf{Y}}^{(t)}$ is the independent spectral embedding of each $\mathbf{A}^{(t)}$. Here, we add the option to ensure that the eigenvector orientation is consistent to remove random flipping between embedding time points. We additionally add the option of aligning subsequent embedding time points via a Procrustes rotation.

### Skip-gram Embedding Methods
- **Unfolded node2vec**: Computes a node2vec embedding on the $(n + nT) \times (n + n T)$ dilated unfolded adjacency matrix,
```math
\begin{equation}
\mathbf{A} = \begin{bmatrix}
\mathbf{0} & \mathcal{A} \\ \mathcal{A}^\top & \mathbf{0}
\end{bmatrix}.
\end{equation}
```
This computation gives both a time-invariant anchor embedding (first $n$ rows) and a time-varying dynamic embedding (remaining $nT$ rows).
- [GloDyNE](https://ieeexplore.ieee.org/abstract/document/9302718): The method aims to preserve global topology by only updating a subset of nodes which accumulate the largest topological changes over subsequent network snapshots. Then, to compute an embedding at $t$, its training is initialised using the pre-trained weights of the SGNS at $t-1$ in order to keep the time points similar [4].
- [Independent node2vec](https://dl.acm.org/doi/abs/10.1145/2939672.2939754): Computes each  $\hat{\mathbf{Y}}^{(t)}$ as an independent static node2vec embedding [5].

## Simulated Dynamic Network Systems

| System | Description | Ideal Testing Power at 5% Level |
| --- | --- | --- |
| ```static``` | Two static communities. | 0.05 |
| ```moving-community``` | Community 2 changes, community 1 is static. The moving community is selected for testing. | 1 |
| ```moving-static-community``` | Community 2 changes, community 1 is static. The static community is selected for testing. | 0.05 |
| ```merge``` | Community 1 and 2 merge over time. | 0.05 |
| ```power-static``` | Static sparse network with a single community. | 0.05 |
| ```power-moving``` | Changing sparse network with a single community. | 1 |

# References
[1] Ian Gallagher, Andrew Jones, and Patrick Rubin-Delanchy. Spectral embedding for dynamic
networks with stability guarantees. Advances in Neural Information Processing Systems,
34:10158–10170, 2021.
```
@article{gallagher2021spectral,
  title={Spectral embedding for dynamic networks with stability guarantees},
  author={Gallagher, Ian and Jones, Andrew and Rubin-Delanchy, Patrick},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={10158--10170},
  year={2021}
}
```
[2] Andrew Jones and Patrick Rubin-Delanchy. The multilayer random dot product graph.
arXiv preprint arXiv:2007.10455, 2020.
```
@article{jones2020multilayer,
  title={The multilayer random dot product graph},
  author={Jones, Andrew and Rubin-Delanchy, Patrick},
  journal={arXiv preprint arXiv:2007.10455},
  year={2020}
}
```
[3] Keith Levin, Avanti Athreya, Minh Tang, Vince Lyzinski, Youngser Park, and Carey E
Priebe. A central limit theorem for an omnibus embedding of multiple random graphs
and implications for multiscale network inference. arXiv preprint arXiv:1705.09355, 2017.
```
@article{levin2017central,
  title={A central limit theorem for an omnibus embedding of multiple random graphs and implications for multiscale network inference},
  author={Levin, Keith and Athreya, Avanti and Tang, Minh and Lyzinski, Vince and Park, Youngser and Priebe, Carey E},
  journal={arXiv preprint arXiv:1705.09355},
  year={2017}
}
```

[4] Chengbin Hou, Han Zhang, Shan He, and Ke Tang. Glodyne: Global topology preserving
dynamic network embedding. IEEE Transactions on Knowledge and Data Engineering,
2020.
```
@article{hou2020glodyne,
  title={Glodyne: Global topology preserving dynamic network embedding},
  author={Hou, Chengbin and Zhang, Han and He, Shan and Tang, Ke},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={34},
  number={10},
  pages={4826--4837},
  year={2020},
  publisher={IEEE}
}
```

[5] Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. In
Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery
and data mining, pages 855–864, 2016.
```
@inproceedings{grover2016node2vec,
  title={node2vec: Scalable feature learning for networks},
  author={Grover, Aditya and Leskovec, Jure},
  booktitle={Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={855--864},
  year={2016}
}
```

