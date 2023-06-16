# import networkx as nx
import numpy as np
from scipy import sparse
import nodevectors
import numba as nb
from GloDyNE_src.libne.DynWalks import DynWalks
import networkx as nx


def GloDyNE(
    As, d, limit=0.1, num_walks=20, walklen=30, window=10, scheme=4, sparse_matrix=False
):
    """
    Computes a dynamic embedding using GloDyNE
    https://ieeexplore.ieee.org/abstract/document/9302718

    Requires following the below instructions to install GloDyNE
    https://github.com/houchengbin/GloDyNE

    Inputs
    As: numpy array of an adjacency matrix series of shape (T, n, n)
    d: embedding dimension
    limit: alpha (defined in the paper)
    num_walks: number of walks per node
    walklen: length of each walk
    window: window size for skipgram
    scheme: scheme 4 for METIS based node selecting approach; scheme 1-3 for other approaches
    sparse_matrix: whether the adjacency matrices are sparse

    Output
    ya: dynamic embedding of shape (n*T, d)
    """
    T = len(As)
    n = As[0].shape[0]
    graphs = []
    for t in range(T):
        if sparse_matrix:
            graphs.append(nx.from_scipy_sparse_matrix(As[t]))
        else:
            graphs.append(nx.from_numpy_array(As[t]))

    model = DynWalks(
        G_dynamic=graphs,
        limit=limit,
        num_walks=num_walks,
        walk_length=walklen,
        window=window,
        emb_dim=d,
        negative=5,
        workers=32,
        seed=2019,
        scheme=scheme,
    )
    ya_raw = model.sampling_traning()

    ya = np.zeros((n * T, d))
    for t in range(T):
        for i in range(n):
            ya[i + n * t, :] = ya_raw[t][i]

    return ya


def ISE(As, d, flat=True, procrustes=False, consistent_orientation=True):
    """
    Computes independent specrtal embedding, ISE for each adjacency snapshot

    Inputs
    As: numpy array of an adjacency matrix series of shape (T, n, n)
    d: embedding dimension
    flat: whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d)
    procrustes: whether to align each embedding with the previous embedding
    consistent_orientation: whether to ensure the eigenvector orientation is consistent

    Output
    YA: dynamic embedding of shape (n*T, d) or (T, n, d)
    """

    n = As[0].shape[0]
    T = len(As)

    if not isinstance(d, list):
        d_list = [d] * T
    else:
        d_list = d

    # Compute embeddings for each time
    YA_list = []
    for t in range(T):
        UA, SA, _ = sparse.linalg.svds(As[t], d_list[t])
        idx = SA.argsort()[::-1]
        UA = UA[:, idx]
        SA = SA[idx]

        # Make sure the eigenvector orientation choice is consistent
        if consistent_orientation:
            sum_of_ev = np.sum(UA, axis=0)
            for i in range(sum_of_ev.shape[0]):
                if sum_of_ev[i] < 0:
                    UA[:, i] = -1 * UA[:, i]

        embed = UA @ np.diag(np.sqrt(SA))

        if embed.shape[1] < max(d_list):
            empty_cols = np.zeros((n, max(d_list) - embed.shape[1]))
            embed = np.column_stack([embed, empty_cols])

        if procrustes and t > 0:
            # Align with previous embedding
            w1, s, w2t = np.linalg.svd(previous_embed.T @ embed)
            w2 = w2t.T
            w = w1 @ w2.T
            embed_rot = embed @ w.T
            embed = embed_rot

        YA_list.append(embed)
        previous_embed = embed

    # Format the output
    if flat:
        YA = np.row_stack(YA_list)
    else:
        YA = np.zeros((T, n, max(d_list)))
        for t in range(T):
            YA[t, :, :] = YA_list[t]

    return YA


def UASE(As, d, flat=True, sparse_matrix=False, return_left=False):
    """
    Computes the unfolded adjacency spectral embedding (UASE)
    https://arxiv.org/abs/2007.10455
    https://arxiv.org/abs/2106.01282

    Inputs
    As: numpy array of an adjacency matrix series of shape (T, n, n)
    d: embedding dimension
    flat: whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d)
    sparse_matrix: whether the adjacency matrices are sparse
    return_left: whether to return the left (anchor) embedding as well as the right (dynamic) embedding

    Output
    YA: dynamic embedding of shape (n*T, d) or (T, n, d)
    (optional) XA: anchor embedding of shape (n, d)
    """
    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        A = As[0]
        for t in range(1, T):
            A = np.hstack((A, As[t]))

    # SVD spectral embedding
    UA, SA, VAt = sparse.linalg.svds(A, d)
    VA = VAt.T
    idx = SA.argsort()[::-1]
    VA = VA[:, idx]
    UA = UA[:, idx]
    SA = SA[idx]
    YA_flat = VA @ np.diag(np.sqrt(SA))
    XA = UA @ np.diag(np.sqrt(SA))
    if flat:
        YA = YA_flat
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t, :, :] = YA_flat[n * t : n * (t + 1), :]

    if not return_left:
        return YA
    else:
        return XA, YA


def independent_n2v(
    As, d, p=1, q=1, flat=True, num_walks=20, window=10, walklen=30, verbose=False
):
    """
    Computes an independent node2vec embedding for each adjacency snapshot

    Requires nodevectors: https://github.com/VHRanger/nodevectors

    Inputs
    As: numpy array of an adjacency matrix series of shape (T, n, n)
    d: embedding dimension
    p: return parameter for node2vec
    q: in-out parameter for node2vec
    flat: whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d)
    num_walks: number of random walks per node
    window: window size for word2vec
    walklen: length of each random walk
    verbose: whether to print progress

    Output
    YA: dynamic embedding of shape (n*T, d) or (T, n, d)
    """

    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    if not isinstance(d, list):
        d_list = [d] * T
    else:
        d_list = d

    YA_list = []
    for t in range(T):
        n2v_obj = nodevectors.Node2Vec(
            n_components=d_list[t],
            epochs=num_walks,
            walklen=walklen,
            return_weight=1 / p,
            neighbor_weight=1 / q,
            verbose=verbose,
            w2vparams={"window": window, "negative": 5, "iter": 10, "batch_words": 128},
        )
        n2v_embed = n2v_obj.fit_transform(As[t])

        if n2v_embed.shape[1] < max(d_list):
            empty_cols = np.zeros((n, max(d_list) - n2v_embed.shape[1]))
            n2v_embed = np.column_stack([n2v_embed, empty_cols])

        YA_list.append(n2v_embed)

    if flat:
        YA = np.row_stack(YA_list)
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t] = YA_list[t]

    return YA


def unfolded_n2v(
    As,
    d,
    p=1,
    q=1,
    sparse_matrix=False,
    flat=True,
    two_hop=False,
    num_walks=20,
    window=10,
    walklen=30,
    verbose=False,
    return_left=False,
):
    """
    Computes the unfolded node2vec embedding (node2vec computed on the dialted unfolded matrix)

    Requires nodevectors: https://github.com/VHRanger/nodevectors

    Inputs
    As: numpy array of an adjacency matrix series of shape (T, n, n)
    d: embedding dimension
    p: return parameter for node2vec
    q: in-out parameter for node2vec
    sparse_matrix: whether the adjacency matrices are sparse
    flat: whether to return a flat embedding (n*T, d) or a 3D embedding (T, n, d)
    two_hop: whether to use a two-hop dilated unfolded adjacency matrix or not
    num_walks: number of random walks per node
    window: window size for word2vec
    walklen: length of each random walk
    verbose: whether to print progress
    return_left: whether to return the left (anchor) embedding as well as the right (dynamic) embedding

    Output
    YA: dynamic embedding of shape (n*T, d) or (T, n, d)
    (optional) XA: anchor embedding of shape (n, d)
    """

    n = As[0].shape[0]
    T = len(As)

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        if len(As.shape) == 2:
            As = np.array([As[:, :]])
        if len(As.shape) == 3:
            T = len(As)
            A = As[0, :, :]
            for t in range(1, T):
                A = np.block([A, As[t]])

    # Construct the dilated unfolded adjacency matrix
    DA = sparse.bmat([[None, A], [A.T, None]])
    DA = sparse.csr_matrix(DA)

    # Compute node2vec
    n2v_obj = nodevectors.Node2Vec(
        n_components=d,
        epochs=num_walks,
        walklen=walklen,
        return_weight=1 / p,
        neighbor_weight=1 / q,
        verbose=verbose,
        w2vparams={"window": window, "negative": 5, "iter": 10, "batch_words": 128},
    )
    if two_hop:
        DA = DA @ DA.T

    n2v = n2v_obj.fit_transform(DA)

    # Take the rows of the embedding corresponding to the right embedding
    # ([0:n] will be left embedding)
    right_n2v = n2v[n:, :]
    XA = n2v[0:n, :]

    if flat:
        YA = right_n2v
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t] = right_n2v[t * n : (t + 1) * n, 0:d]

    if not return_left:
        return YA
    else:
        return XA, YA


def safe_inv_sqrt(a, tol=1e-12):
    """Computes the inverse square root, but returns zero if the result is either infinity
    or below a tolerance"""
    with np.errstate(divide="ignore"):
        b = 1 / np.sqrt(a)
    b[np.isinf(b)] = 0
    b[a < tol] = 0
    return b


def to_laplacian(A, regulariser=0, verbose=False):
    """Constructs the (regularised) symmetric Laplacian."""
    left_degrees = np.reshape(np.asarray(A.sum(axis=1)), (-1,))
    right_degrees = np.reshape(np.asarray(A.sum(axis=0)), (-1,))
    if regulariser == "auto":
        regulariser = np.mean(np.concatenate((left_degrees, right_degrees)))
        if verbose:
            print("Auto regulariser: {}".format(regulariser))
    left_degrees_inv_sqrt = safe_inv_sqrt(left_degrees + regulariser)
    right_degrees_inv_sqrt = safe_inv_sqrt(right_degrees + regulariser)
    L = sparse.diags(left_degrees_inv_sqrt) @ A @ sparse.diags(right_degrees_inv_sqrt)
    return L


def regularised_ULSE(
    As,
    d,
    regulariser=0,
    flat=True,
    sparse_matrix=False,
    return_left=False,
    verbose=False,
):
    """Compute the unfolded (regularlised) Laplacian Spectral Embedding
    1. Construct dilated unfolded adjacency matrix
    2. Compute the symmetric (regularised) graph Laplacian
    3. Spectral Embed
    As: adjacency matrices of shape (T, n, n)
    d: embedding dimension
    regulariser: regularisation parameter. 0 for no regularisation, 'auto' for automatic regularisation (this often isn't the best).
    flat: True outputs embedding of shape (nT, d), False outputs shape (T, n, d)
    """

    # Assume fixed n over time
    n = As[0].shape[0]
    T = len(As)

    # Construct the rectangular unfolded adjacency
    if sparse_matrix:
        A = As[0]
        for t in range(1, T):
            A = sparse.hstack((A, As[t]))
    else:
        A = As[0]
        for t in range(1, T):
            A = np.hstack((A, As[t]))

    # Construct (regularised) Laplacian matrix
    L = to_laplacian(A, regulariser=regulariser, verbose=verbose)

    # Compute spectral embedding
    U, S, Vt = sparse.linalg.svds(L, d)
    idx = np.abs(S).argsort()[::-1]
    YA_flat = Vt.T[:, idx] @ np.diag((np.sqrt(S[idx])))
    XA = U[:, idx] @ np.diag(np.sqrt(S[idx]))

    if flat:
        YA = YA_flat
    else:
        YA = np.zeros((T, n, d))
        for t in range(T):
            YA[t] = YA_flat[t * n : (t + 1) * n, 0:d]

    if not return_left:
        return YA
    else:
        return XA, YA


@nb.njit()
def form_omni_matrix(As, n, T):
    """
    Forms the embedding matrix for the omnibus embedding
    """
    A = np.zeros((T * n, T * n))

    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                A[t1 * n : (t1 + 1) * n, t1 * n : (t1 + 1) * n] = As[t1]
            else:
                A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] = (As[t1] + As[t2]) / 2

    return A


def form_omni_matrix_sparse(As, n, T):
    """
    Forms embedding matrix for the omnibus embedding using sparse matrices
    """
    A = sparse.lil_matrix((T * n, T * n))

    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                A[t1 * n : (t1 + 1) * n, t1 * n : (t1 + 1) * n] = As[t1]
            else:
                A[t1 * n : (t1 + 1) * n, t2 * n : (t2 + 1) * n] = (As[t1] + As[t2]) / 2

    return A


def OMNI(As, d, flat=True, sparse_matrix=False):
    """
    Computes the omnibus spectral embedding
    https://arxiv.org/abs/1705.09355

    Inputs
    As: adjacency matrices of shape (T, n, n)
    d: embedding dimension
    flat: True outputs embedding of shape (nT, d), False outputs shape (T, n, d)
    sparse_matrix: True uses sparse matrices, False uses dense matrices

    Outputs
    XA: dynamic embedding of shape (T, n, d) or (nT, d)

    """

    n = As[0].shape[0]
    T = len(As)

    # Construct omnibus matrices
    if sparse_matrix:
        A = form_omni_matrix_sparse(As, n, T)
    else:
        A = form_omni_matrix(As, n, T)

    # Compute spectral embedding
    UA, SA, _ = sparse.linalg.svds(A, d)
    idx = SA.argsort()[::-1]
    UA = np.real(UA[:, idx][:, 0:d])
    SA = np.real(SA[idx][0:d])
    XA_flat = UA @ np.diag(np.sqrt(np.abs(SA)))

    if flat:
        XA = XA_flat
    else:
        XA = np.zeros((T, n, d))
        for t in range(T):
            XA[t] = XA_flat[t * n : (t + 1) * n, 0:d]

    return XA


def embed(
    As,
    d,
    method,
    regulariser=0,
    p=1,
    q=1,
    two_hop=False,
    verbose=False,
    num_walks=20,
    window=10,
    walklen=30,
):
    """
    Computes a dynamic embedding using a specified method
    """
    if method.upper() == "ISE":
        YA = ISE(As, d)
    elif method.upper() == "ISE PROCRUSTES":
        YA = ISE(As, d, procrustes=True)
    elif method.upper() == "UASE":
        YA = UASE(As, d)
    elif method.upper() == "OMNI":
        YA = OMNI(As, d)
    elif method.upper() == "ULSE":
        YA = regularised_ULSE(As, d, regulariser=0)
    elif method.upper() == "URLSE":
        YA = regularised_ULSE(As, d, regulariser=regulariser)
    elif method.upper() == "INDEPENDENT NODE2VEC":
        YA = independent_n2v(
            As,
            d,
            p=p,
            q=q,
            num_walks=num_walks,
            window=window,
            walklen=walklen,
            verbose=verbose,
        )
    elif method.upper() == "UNFOLDED NODE2VEC":
        YA = unfolded_n2v(
            As,
            d,
            p=p,
            q=q,
            two_hop=two_hop,
            num_walks=num_walks,
            window=window,
            walklen=walklen,
            verbose=verbose,
        )
    elif method.upper() == "GLODYNE":
        YA = GloDyNE(As, d, num_walks=num_walks, window=window, walklen=walklen)
    else:
        raise ValueError(
            "Method given is not a recognised embedding method\n- Please select from:\n\t> ISE\n\t> ISE PROCRUSTES\n\t> OMNI\n\t> UASE\n\t> ULSE\n\t> URLSE\n\t> INDEPENDENT NODE2VEC\n\t> UNFOLDED NODE2VEC\n\t> GLODYNE\n"
        )

    return YA


@nb.njit()
def vector_displacement_test(ya1, ya2):
    """
    Computes the vector displacement between two embedding sets

    ya1 & ya2 are numpy arrays of equal shape (n, d)
    """

    displacement = ya2 - ya1
    sum_axis = displacement.sum(axis=0)

    # Magnitude of average displacement vector
    t_obs = np.linalg.norm(sum_axis)

    return t_obs


@nb.njit()
def test_temporal_displacement(ya, n, T, changepoint, n_sim=1000):
    """Computes vector displacement permutation test with temporal permutations
    Vector displacement is expected to be approximately zero if two sets are from the same distribution and non-zero otherwise.
    Temporal permutations only permutes a node embedding at t with its representation at other times.

    This function should only be used when comparing more than two time points across a single changepoint.

    ya: (numpy array (nT, d) Entire dynamic embedding.
    n: (int) number of nodes
    T: (int) number of time points
    changepoint: (int) time point at which to split the embedding
    n_sim: (int) number of permuted test statistics computed.
    """

    if changepoint < 1:
        raise Exception("Changepoint must be at least 1.")
    elif changepoint > T:
        raise Exception("Changepoint must be less than or equal T")

    # Select time point embedding just before and after the changepoint
    ya1 = ya[n * (changepoint - 1) : n * (changepoint), :]
    ya2 = ya[n * changepoint : n * (changepoint + 1), :]

    # Get observed value of the test
    t_obs = vector_displacement_test(ya1, ya2)

    # Permute the sets
    t_obs_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        ya_star = ya.copy()
        for j in range(n):
            # For each node get its position at each time - permute over these positions
            possible_perms = ya[j::n, :]
            ya_star[j::n, :] = possible_perms[np.random.choice(T, T, replace=False), :]

        # Get permuted value of the test
        ya_star_1 = ya_star[n * (changepoint - 1) : n * changepoint, :]
        ya_star_2 = ya_star[n * changepoint : n * (changepoint + 1), :]
        t_obs_star = vector_displacement_test(ya_star_1, ya_star_2)
        t_obs_stars[sim_iter] = t_obs_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_obs_stars >= t_obs)
    return p_hat


@nb.njit()
def test_temporal_displacement_two_times(ya, n, n_sim=1000):
    """
    Much faster version of test_temporal_displacement for when there are only two time points

    Computes vector displacement permutation test with temporal permutations
    Vector displacement is expected to be approximately zero if two sets are from the same distribution and non-zero otherwise.
    Temporal permutations only permutes a node embedding at t with its representation at other times.

    ya: (numpy array (nT, d) Entire dynamic embedding.
    n: (int) number of nodes
    changepoint (int > radius, <= T-radius)
    n_sim: (int) number of permuted test statistics computed.
    """

    # Select time point embedding just before and after the changepoint
    ya1 = ya[0:n, :]
    ya2 = ya[n : 2 * n, :]

    # Get observed value of the test
    displacement = ya2 - ya1
    sum_axis = displacement.sum(axis=0)
    t_obs = np.linalg.norm(sum_axis)

    # Permute the sets
    t_stars = np.zeros((n_sim))
    for sim_iter in range(n_sim):
        # Randomly swap the signs of each row of displacement
        signs = np.random.randint(0, 2, size=displacement.shape) * 2 - 1
        displacement_permuted = displacement * signs
        sum_axis_permuted = displacement_permuted.sum(axis=0)
        t_star = np.linalg.norm(sum_axis_permuted)
        t_stars[sim_iter] = t_star

    # Compute permutation test p-value
    p_hat = 1 / n_sim * np.sum(t_stars >= t_obs)
    return p_hat
