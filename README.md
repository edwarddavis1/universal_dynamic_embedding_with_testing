# Universal Stable Dynamic Graph Embedding with Robust Hypothesis Testing

This repository implements a collection of spectral and skip-gram dynamic network embedding methods. This repo compares how well different methods can encode planted structure in simulated dynamic networks through the use of a hypothesis test. The methods are also applied to a dynamic flight network to see whether they can encode both the periodic nature of the network, as well as represent the disruption caused by the COVID-19 pandemic on air traffic.

## Getting Started

This code has been tested using python 3.8.10. To install the required packages for this repo use the following command.

```
pip install -r requirements.txt
```

## Data

We consider the following datasets in this analysis.

- The [OpenSky](https://zenodo.org/record/5815448#.Y1_ydy-l1hD) network, which is a 17,388 node dynamic network of flights which occurred from the start of 2019 to the end of 2021.
- ```airports.csv``` from [OurAirports](https://ourairports.com/data/), which contains information about airports around the world.
- The [C5 regulations](https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/timeseries/c5m_close_public_transport.csv) data, from the [Oxford COVID-19 Government Response Tracker](https://github.com/OxCGRT/covid-policy-tracker). This data appears as ```c5m_close_public_transport.csv```. This lists the travel restrictions in place per country in response to the COVID-19 pandemic.


## Scripts
**Dynamic Embedding Functions**: 
- ```embedding_functions.py```: Contains functions for several spectral and skip-gram dynamic embedding methods (listed below). Also contains functions to perform hypothesis testing on a dynamic embedding.

**Simulated experiments and testing**
- ```experiment_setup.py```: Contains functions to generate of various dynamic networks with planted structure. 
- ```experiments.py```: Checks if various dynamic embedding methods can encode planted structure through the use of a hypothesis test. For each, a cumulative p-value distribution is plot and coloured based on if the dynamic embedding correctly encoded the planted structure or not. The plots are coloured green if correct, red if incorrect and blue if correct but the test was conservative.


**Flight Network Analysis**:
- ```prep_flight_data.py```: Generates a sparse adjacency series and labelled node set from the [OpenSky](https://zenodo.org/record/5815448#.Y1_ydy-l1hD) network. 
- ```flight_network_analysis.py```: Computes a dynamic embedding of the [OpenSky](https://zenodo.org/record/5815448#.Y1_ydy-l1hD) network. Follow-up analysis on this embedding includes the generation of dissimilarity and p-value matrices to study the encoded temporal structure of the network, and heirarchical clustering. 
- ```flight_and_c5_regulations.py```: Produces a neat plot of the dissimilarity and p-value matrices to compare the encoded temporal structure of the network to a plot of travel restrictions in Europe in response to the COVID-19 pandemic. There are two waves of dissimilarity in the dissimilarity matrix which line up with the two main European waves of the pandemic.

## Dynamic Embedding Methods

This repository implements a collection of spectral and skip-gram dynamic embedding methods. Here, we consider the problem of embedding discrete-time dynamic networks, i.e. those that can be represented as a series of adjacency matrix ``snapshots" over time, $\mathbf{A}^{(1)}, \dots, \mathbf{A}^{(T)}$. A dynamic embedding is then a low-dimensional representation for each of snapshots in the series, $\hat{\mathbf{Y}}^{(1)}, \dots, \hat{\mathbf{Y}}^{(T)} \in \mathbb{R}^{n \times d}$, which we refer to as embedding time points. Methods in **bold** are stable dynamic embedding methods.

### Spectral Embedding Methods
- [**Unfolded adjacency spectral embedding (UASE)**](https://arxiv.org/abs/2007.10455): Constructs a column-concatenated adjacency matrix, $\mathcal{A} = \left[\mathbf{A}^{(1)}, \dots, \mathbf{A}^{(T)} \right]$, called the rectangular unfolded adjacency matrix. The dynamic embedding is then achieved through an right SVD embedding [2,3].
- **Unfolded Regularised Laplacian Spectral Embedding (URLSE)**: computes a spectral embedding the matrix, 
$ \begin{equation}\mathbf{L} = \left(\mathbf{D}_{\text{L}} - \gamma \right)^{-1/2} \mathcal{A} \left(\mathbf{D}_{\text{R}} - \gamma \right)^{-1/2} \end{equation} $,
where $\mathbf{D}_{\text{L}} = \text{diag}\left(\sum_i^{n}{\mathcal{A}^{\top}_i}\right)$ and $\mathbf{D}_{\text{R}} = \text{diag}\left(\sum_i^{nT}{\mathcal{A}_i}\right)$ are the left and right degree matrices and $\gamma$ is a regularisation parameter.
- [Omnibus embedding (OMNI)](https://arxiv.org/abs/1705.09355): Constructs a single block matrix containing pairwise-averaged adjacency snapshots, $\mathbf{M}_{s,t} = (\mathbf{A}^{(s)} + \mathbf{A}^{(t)})/2$, and then computes a spectral embedding on this matrix to achieve a dynamic embedding [1].
- Independent spectral embedding (ISE): Each $\hat{\mathbf{Y}}^{(t)}$ is the independent spectral embedding of each $\mathbf{A}^{(t)}$. Here, we add the option to ensure that the eigenvector orientation is consistent to remove random flipping between embedding time points. We additionally add the option of aligning subsequent embedding time points via a Procrustes rotation.

### Skip-gram Embedding Methods
- **Unfolded node2vec**: Computes a node2vec embedding on the $(n + nT) \times (n + n T)$ dilated unfolded adjacency matrix,
$
\begin{equation}
\mathbf{A} = \begin{bmatrix}
\mathbf{0} & \mathcal{A} \\ \mathcal{A}^\top & \mathbf{0}
\end{bmatrix}.
\end{equation}
$
This computation gives both a time-invariant anchor embedding (first $n$ rows) and a time-varying dynamic embedding (remaining $nT$ rows).
- [GloDyNE](https://ieeexplore.ieee.org/abstract/document/9302718): The method aims to preserve global topology by only updating a subset of nodes which accumulate the largest topological changes over subsequent network snapshots. Then, to compute an embedding at $t$, its training is initialised using the pre-trained weights of the SGNS at $t-1$ in order to keep the time points similar [4].
- [Independent node2vec](https://dl.acm.org/doi/abs/10.1145/2939672.2939754): Computes each  $\hat{\mathbf{Y}}^{(t)}$ as an independent static node2vec embedding [5].

# References
[1] Keith Levin, Avanti Athreya, Minh Tang, Vince Lyzinski, Youngser Park, and Carey E
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
[2] Ian Gallagher, Andrew Jones, and Patrick Rubin-Delanchy. Spectral embedding for dynamic
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
[3] Andrew Jones and Patrick Rubin-Delanchy. The multilayer random dot product graph.
arXiv preprint arXiv:2007.10455, 2020.
```
@article{jones2020multilayer,
  title={The multilayer random dot product graph},
  author={Jones, Andrew and Rubin-Delanchy, Patrick},
  journal={arXiv preprint arXiv:2007.10455},
  year={2020}
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
