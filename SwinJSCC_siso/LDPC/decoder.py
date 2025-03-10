# """Decoding module."""
# import numpy as np
# import warnings

# import torch

# from . import utils

# from numba import njit, int64, types, float64


# def fc(LLR, rho, LLR_limit=50, Lp_target=None):
#     # LLR_limit = abs(Lp_target[:LLR.shape[0], :]) * 0.9
#     # idx = np.where(abs(LLR) - LLR_limit > 0)
#     # LLR[idx] = np.sign(LLR[idx]) * LLR_limit[idx]
#     # scale = min(1, (Lp_target ** 2).sum() / (Lp_target ** 2).sum()/2)
#     # LLR *= scale
#     # LLR_limit =abs(Lp_target)/2
#     # LLR_limit =min(100, max(abs(Lp_target)))
#     LLR_limit=torch.tensor(LLR_limit, dtype=torch.float32)
#     LLR = torch.tensor(LLR, dtype=torch.float32)
#     # idx = np.where(LLR < -LLR_limit)
#     # LLR[idx] = -LLR_limit
#     # print("idx1:",LLR)
#     idx = np.where(LLR > LLR_limit)
#     LLR[idx] = LLR_limit
#     # print("idx2:",idx,LLR)
#     a = (1 - rho) * np.exp(LLR) + rho
#     b = (1 - rho) + rho * np.exp(LLR)
#     idx = np.where(a > 1e-3)
#     LLR[idx] = torch.tensor(np.log(a[idx] / b[idx]), dtype=torch.float32)
#     # print("idx3:",idx,LLR)

#     return LLR


# def interleaver(x, patten=None, seed=None):
#     if patten is None:
#         rng = utils.check_random_state(seed)
#         patten = np.arange(x.size)
#         rng.shuffle(patten)
#         out_patten = True
#     else:
#         out_patten = False

#     y = np.zeros(x.shape)
#     y[patten] = x

#     if out_patten:
#         return y, patten
#     else:
#         return y


# def deinterleaver(x, patten):
#     return x[patten]


# def BER(x, y):
#     # print(x.shape,y.shape,np.array(x - y).shape)
#     x = np.array(x)
#     y = np.array(y)
#     return abs(x - y).sum() / x.size


# def decode(H, y, snr, maxiter=1000):
#     """Decode a Gaussian noise corrupted n bits message using BP algorithm.

#     Decoding is performed in parallel if multiple codewords are passed in y.

#     Parameters
#     ----------
#     H: array (n_equations, n_code). Decoding matrix H.
#     y: array (n_code, n_messages) or (n_code,). Received message(s) in the
#         codeword space.
#     maxiter: int. Maximum number of iterations of the BP algorithm.

#     Returns
#     -------
#     x: array (n_code,) or (n_code, n_messages) the solutions in the
#         codeword space.

#     """
#     m, n = H.shape

#     bits_hist, bits_values, nodes_hist, nodes_values = utils._bitsandnodes(H)

#     _n_bits = np.unique(H.sum(0))
#     _n_nodes = np.unique(H.sum(1))

#     if _n_bits * _n_nodes == 1:
#         solver = _logbp_numba_regular
#         bits_values = bits_values.reshape(n, -1)
#         nodes_values = nodes_values.reshape(m, -1)

#     else:
#         solver = _logbp_numba

#     var = 10 ** (-snr / 10)

#     if y.ndim == 1:
#         y = y[:, None]
#     # step 0: initialization

#     Lc = 2 * y / var
#     _, n_messages = y.shape

#     Lq = np.zeros(shape=(m, n, n_messages))

#     Lr = np.zeros(shape=(m, n, n_messages))
#     for n_iter in range(maxiter):
#         Lq, Lr, L_posteriori = solver(bits_hist, bits_values, nodes_hist,
#                                       nodes_values, Lc, Lq, Lr, n_iter)
#         x = np.array(L_posteriori <= 0).astype(int)
#         product = utils.incode(H, x)
#         if product:
#             break
#     if n_iter == maxiter - 1:
#         warnings.warn("""Decoding stopped before convergence. You may want
#                        to increase maxiter""")
#     return x.squeeze()


# def decoder_init(H, y, snr):
#     bits_hist, bits_values, nodes_hist, nodes_values = utils._bitsandnodes(H)

#     _n_bits = np.unique(H.sum(0))
#     _n_nodes = np.unique(H.sum(1))

#     if _n_bits * _n_nodes == 1:
#         solver = _logbp_numba_regular
#         bits_values = bits_values.reshape(n, -1)
#         nodes_values = nodes_values.reshape(m, -1)

#     else:
#         solver = _logbp_numba

#     if y.ndim == 1:
#         y = y[:, None]
#     # step 0: initialization

#     if snr is not None:
#         var = 10 ** (-snr / 10)
#         y *= 2 / var

#     return y, {"H": H, "solver": solver, "bits_values": bits_values, "nodes_values": nodes_values,
#                "nodes_hist": nodes_hist, "bits_hist": bits_hist}


# def decode_LLR(Lc, H, solver, bits_values, nodes_values, nodes_hist, bits_hist, La=None, maxiter=10):
#     """Decode a Gaussian noise corrupted n bits message using BP algorithm.

#     Decoding is performed in parallel if multiple codewords are passed in y.

#     Parameters
#     ----------
#     H: array (n_equations, n_code). Decoding matrix H.
#     Lc: a priori LLR of codewords.
#     maxiter: int. Maximum number of iterations of the BP algorithm.

#     """

#     m, n = H.shape

#     if La is not None:
#         k = La.shape[1]
#         # print(Lc.shape,La.shape)
#         Lc[:k, :] = np.add(Lc[:k, :], np.array(La).T)
#         # print(Lc.shape)

#     _, n_messages = Lc.shape

#     Lq = np.zeros(shape=(m, n, n_messages))

#     Lr = np.zeros(shape=(m, n, n_messages))
#     for n_iter in range(maxiter):
#         Lq, Lr, L_posteriori = solver(bits_hist, bits_values, nodes_hist,
#                                       nodes_values, Lc, Lq, Lr, n_iter)
#         x = np.array(L_posteriori <= 0).astype(int)
#         product = utils.incode(H, x)
#         if product:
#             break
#     return L_posteriori


# output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :],
#                                 float64[:, :]))


# @njit(output_type_log2(int64[:], int64[:], int64[:], int64[:], float64[:, :],
#                        float64[:, :, :], float64[:, :, :], int64), cache=True)
# def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr,
#                  n_iter):
#     """Perform inner ext LogBP solver."""
#     m, n, n_messages = Lr.shape
#     # step 1 : Horizontal

#     bits_counter = 0
#     nodes_counter = 0
#     for i in range(m):
#         # ni = bits[i]
#         ff = bits_hist[i]
#         ni = bits_values[bits_counter: bits_counter + ff]
#         bits_counter += ff
#         for j in ni:
#             nij = ni[:]

#             X = np.ones(n_messages)
#             if n_iter == 0:
#                 for kk in range(len(nij)):
#                     if nij[kk] != j:
#                         X *= np.tanh(0.5 * Lc[nij[kk]])
#             else:
#                 for kk in range(len(nij)):
#                     if nij[kk] != j:
#                         X *= np.tanh(0.5 * Lq[i, nij[kk]])
#             num = 1 + X
#             denom = 1 - X
#             for ll in range(n_messages):
#                 if num[ll] == 0:
#                     Lr[i, j, ll] = -1
#                 elif denom[ll] == 0:
#                     Lr[i, j, ll] = 1
#                 else:
#                     Lr[i, j, ll] = np.log(num[ll] / denom[ll])

#     # step 2 : Vertical
#     for j in range(n):
#         # mj = nodes[j]
#         ff = nodes_hist[j]
#         mj = nodes_values[nodes_counter: nodes_counter + ff]
#         nodes_counter += ff
#         for i in mj:
#             mji = mj[:]
#             Lq[i, j] = Lc[j]

#             for kk in range(len(mji)):
#                 if mji[kk] != i:
#                     Lq[i, j] += Lr[mji[kk], j]

#     # LLR a posteriori:
#     L_posteriori = np.zeros((n, n_messages))
#     nodes_counter = 0
#     for j in range(n):
#         ff = nodes_hist[j]
#         mj = nodes_values[nodes_counter: nodes_counter + ff]
#         nodes_counter += ff
#         L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

#     return Lq, Lr, L_posteriori


# @njit(output_type_log2(int64[:], int64[:, :], int64[:], int64[:, :],
#                        float64[:, :], float64[:, :, :], float64[:, :, :],
#                        int64), cache=True)
# def _logbp_numba_regular(bits_hist, bits_values, nodes_hist, nodes_values, Lc,
#                          Lq, Lr, n_iter):
#     """Perform inner ext LogBP solver."""
#     m, n, n_messages = Lr.shape
#     # step 1 : Horizontal
#     for i in range(m):
#         ni = bits_values[i]
#         for j in ni:
#             nij = ni[:]

#             X = np.ones(n_messages)
#             if n_iter == 0:
#                 for kk in range(len(nij)):
#                     if nij[kk] != j:
#                         X *= np.tanh(0.5 * Lc[nij[kk]])
#             else:
#                 for kk in range(len(nij)):
#                     if nij[kk] != j:
#                         X *= np.tanh(0.5 * Lq[i, nij[kk]])
#             num = 1 + X
#             denom = 1 - X
#             for ll in range(n_messages):  # arctanh
#                 if num[ll] == 0:
#                     Lr[i, j, ll] = -1
#                 elif denom[ll] == 0:
#                     Lr[i, j, ll] = 1
#                 else:
#                     Lr[i, j, ll] = np.log(num[ll] / denom[ll])

#     # step 2 : Vertical
#     for j in range(n):
#         mj = nodes_values[j]
#         for i in mj:
#             mji = mj[:]
#             Lq[i, j] = Lc[j]

#             for kk in range(len(mji)):
#                 if mji[kk] != i:
#                     Lq[i, j] += Lr[mji[kk], j]

#     # LLR a posteriori:
#     L_posteriori = np.zeros((n, n_messages))
#     for j in range(n):
#         mj = nodes_values[j]
#         L_posteriori[j] = Lc[j] + Lr[mj, j].sum(axis=0)

#     return Lq, Lr, L_posteriori


# def get_message(tG, x):
#     """Compute the original `n_bits` message from a `n_code` codeword `x`.

#     Parameters
#     ----------
#     tG: array (n_code, n_bits) coding matrix tG.
#     x: array (n_code,) decoded codeword of length `n_code`.

#     Returns
#     -------
#     message: array (n_bits,). Original binary message.

#     """
#     n, k = tG.shape

#     rtG, rx = utils.gausselimination(tG, x)

#     message = np.zeros(k).astype(int)

#     message[k - 1] = rx[k - 1]
#     for i in reversed(range(k - 1)):
#         message[i] = rx[i]
#         message[i] -= utils.binaryproduct(rtG[i, list(range(i + 1, k))],
#                                           message[list(range(i + 1, k))])

#     return abs(message)

"""Decoding module."""
import numpy as np
import warnings

import torch

from . import utils

from numba import njit, int64, types, float64

# ------------------------------------------------------------------
# Modified fc: now using fully vectorized torch operations
# ------------------------------------------------------------------
def fc(LLR, rho, LLR_limit=50, Lp_target=None):
    # Use torch tensors throughout.
    LLR = torch.as_tensor(LLR, dtype=torch.float32)
    # Only clamp the upper bound (as in the original code)
    LLR_limit_tensor = torch.tensor(LLR_limit, dtype=torch.float32, device=LLR.device)
    LLR = torch.where(LLR > LLR_limit_tensor, LLR_limit_tensor, LLR)
    # Compute a and b in a vectorized way.
    a = (1 - rho) * torch.exp(LLR) + rho
    b = (1 - rho) + rho * torch.exp(LLR)
    mask = a > 1e-3
    LLR[mask] = torch.log(a[mask] / b[mask])
    return LLR

# ------------------------------------------------------------------
# interleaver, deinterleaver, BER (unchanged)
# ------------------------------------------------------------------
def interleaver(x, patten=None, seed=None):
    if patten is None:
        rng = utils.check_random_state(seed)
        patten = np.arange(x.size)
        rng.shuffle(patten)
        out_patten = True
    else:
        out_patten = False

    y = np.zeros(x.shape)
    y[patten] = x

    if out_patten:
        return y, patten
    else:
        return y

def deinterleaver(x, patten):
    return x[patten]

def BER(x, y):
    x = np.array(x.cpu())
    y = np.array(y.cpu())
    return abs(x - y).sum() / x.size

# ------------------------------------------------------------------
# Main decode functions (unchanged API)
# ------------------------------------------------------------------
def decode(H, y, snr, maxiter=1000):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm.

    Decoding is performed in parallel if multiple codewords are passed in y.

    Parameters
    ----------
    H: array (n_equations, n_code). Decoding matrix H.
    y: array (n_code, n_messages) or (n_code,). Received message(s) in the
        codeword space.
    maxiter: int. Maximum number of iterations of the BP algorithm.

    Returns
    -------
    x: array (n_code,) or (n_code, n_messages) the solutions in the
        codeword space.

    """
    m, n = H.shape

    bits_hist, bits_values, nodes_hist, nodes_values = utils._bitsandnodes(H)

    _n_bits = np.unique(H.sum(0))
    _n_nodes = np.unique(H.sum(1))

    if _n_bits * _n_nodes == 1:
        solver = _logbp_numba_regular
        bits_values = bits_values.reshape(n, -1)
        nodes_values = nodes_values.reshape(m, -1)
    else:
        solver = _logbp_numba

    var = 10 ** (-snr / 10)

    if y.ndim == 1:
        y = y[:, None]
    # Initialization:
    Lc = 2 * y / var
    _, n_messages = y.shape

    Lq = np.zeros((m, n, n_messages), dtype=np.float64)
    Lr = np.zeros((m, n, n_messages), dtype=np.float64)
    for n_iter in range(maxiter):
        Lq, Lr, L_posteriori = solver(bits_hist, bits_values, nodes_hist,
                                      nodes_values, Lc, Lq, Lr, n_iter)
        x = (L_posteriori <= 0).astype(int)
        product = utils.incode(H, x)
        if product:
            break
    if n_iter == maxiter - 1:
        warnings.warn("""Decoding stopped before convergence. You may want
                       to increase maxiter""")
    return x.squeeze()

def decoder_init(H, y, snr):
    bits_hist, bits_values, nodes_hist, nodes_values = utils._bitsandnodes(H)

    _n_bits = np.unique(H.sum(0))
    _n_nodes = np.unique(H.sum(1))

    if _n_bits * _n_nodes == 1:
        solver = _logbp_numba_regular
        bits_values = bits_values.reshape(n, -1)
        nodes_values = nodes_values.reshape(m, -1)
    else:
        solver = _logbp_numba

    if y.ndim == 1:
        y = y[:, None]
    if snr is not None:
        var = 10 ** (-snr / 10)
        y *= 2 / var

    return y, {"H": H, "solver": solver, "bits_values": bits_values,
               "nodes_values": nodes_values, "nodes_hist": nodes_hist,
               "bits_hist": bits_hist}

def decode_LLR(Lc, H, solver, bits_values, nodes_values, nodes_hist,
               bits_hist, La=None, maxiter=10):
    """Decode a Gaussian noise corrupted n bits message using BP algorithm.

    Parameters
    ----------
    H: array (n_equations, n_code). Decoding matrix H.
    Lc: a priori LLR of codewords.
    maxiter: int. Maximum number of iterations of the BP algorithm.

    """
    m, n = H.shape

    if La is not None:
        k = La.shape[1]
        Lc[:k, :] = np.add(Lc[:k, :], np.array(La).T)

    _, n_messages = Lc.shape

    Lq = np.zeros((m, n, n_messages), dtype=np.float64)
    Lr = np.zeros((m, n, n_messages), dtype=np.float64)
    for n_iter in range(maxiter):
        Lq, Lr, L_posteriori = solver(bits_hist, bits_values, nodes_hist,
                                      nodes_values, Lc, Lq, Lr, n_iter)
        x = (L_posteriori <= 0).astype(int)
        product = utils.incode(H, x)
        if product:
            break
    return L_posteriori

# ------------------------------------------------------------------
# Numba helper type (unchanged)
# ------------------------------------------------------------------
output_type_log2 = types.Tuple((float64[:, :, :], float64[:, :, :],
                                float64[:, :]))

# ------------------------------------------------------------------
# Modified Numba functions with pre-computed products and vectorized vertical step.
# Note: We also enable fastmath=True.
# ------------------------------------------------------------------
@njit(output_type_log2(int64[:], int64[:], int64[:], int64[:],
                       float64[:, :], float64[:, :, :], float64[:, :, :],
                       int64), cache=True, fastmath=True)
def _logbp_numba(bits_hist, bits_values, nodes_hist, nodes_values, Lc, Lq, Lr,
                 n_iter):
    m, n, n_messages = Lr.shape
    bits_counter = 0
    for i in range(m):
        ff = bits_hist[i]
        ni = bits_values[bits_counter: bits_counter + ff]
        bits_counter += ff
        # Precompute tanh values for all connected variable nodes.
        T = np.empty((ff, n_messages))
        for k in range(ff):
            idx = ni[k]
            if n_iter == 0:
                for ll in range(n_messages):
                    T[k, ll] = np.tanh(0.5 * Lc[idx, ll])
            else:
                for ll in range(n_messages):
                    T[k, ll] = np.tanh(0.5 * Lq[i, idx, ll])
        # Compute the elementwise product P over all connections.
        P = np.ones(n_messages)
        for k in range(ff):
            for ll in range(n_messages):
                P[ll] *= T[k, ll]
        # For each connection, compute the message by "dividing out" the current term.
        for k in range(ff):
            j = ni[k]
            X = np.empty(n_messages)
            for ll in range(n_messages):
                if T[k, ll] != 0.0:
                    X[ll] = P[ll] / T[k, ll]
                else:
                    prod = 1.0
                    for kk in range(ff):
                        if kk != k:
                            prod *= T[kk, ll]
                    X[ll] = prod
            for ll in range(n_messages):
                num = 1.0 + X[ll]
                den = 1.0 - X[ll]
                if num == 0.0:
                    Lr[i, j, ll] = -1.0
                elif den == 0.0:
                    Lr[i, j, ll] = 1.0
                else:
                    Lr[i, j, ll] = np.log(num / den)
    # Vertical step: vectorize by computing a sum over all check-node messages
    nodes_counter = 0
    for j in range(n):
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        sum_Lr = np.zeros(n_messages)
        for idx in range(ff):
            i = mj[idx]
            for ll in range(n_messages):
                sum_Lr[ll] += Lr[i, j, ll]
        for idx in range(ff):
            i = mj[idx]
            for ll in range(n_messages):
                Lq[i, j, ll] = Lc[j, ll] + (sum_Lr[ll] - Lr[i, j, ll])
    # Compute a posteriori LLR for each variable node.
    L_posteriori = np.empty((n, n_messages))
    nodes_counter = 0
    for j in range(n):
        ff = nodes_hist[j]
        mj = nodes_values[nodes_counter: nodes_counter + ff]
        nodes_counter += ff
        for ll in range(n_messages):
            temp = Lc[j, ll]
            for idx in range(ff):
                i = mj[idx]
                temp += Lr[i, j, ll]
            L_posteriori[j, ll] = temp
    return Lq, Lr, L_posteriori

@njit(output_type_log2(int64[:], int64[:, :], int64[:], int64[:, :],
                       float64[:, :], float64[:, :, :], float64[:, :, :],
                       int64), cache=True, fastmath=True)
def _logbp_numba_regular(bits_hist, bits_values, nodes_hist, nodes_values, Lc,
                         Lq, Lr, n_iter):
    m, n, n_messages = Lr.shape
    for i in range(m):
        ni = bits_values[i]
        ff = bits_hist[i]
        T = np.empty((ff, n_messages))
        for k in range(ff):
            j = ni[k]
            if n_iter == 0:
                for ll in range(n_messages):
                    T[k, ll] = np.tanh(0.5 * Lc[j, ll])
            else:
                for ll in range(n_messages):
                    T[k, ll] = np.tanh(0.5 * Lq[i, j, ll])
        P = np.ones(n_messages)
        for k in range(ff):
            for ll in range(n_messages):
                P[ll] *= T[k, ll]
        for k in range(ff):
            j = ni[k]
            X = np.empty(n_messages)
            for ll in range(n_messages):
                if T[k, ll] != 0.0:
                    X[ll] = P[ll] / T[k, ll]
                else:
                    prod = 1.0
                    for kk in range(ff):
                        if kk != k:
                            prod *= T[kk, ll]
                    X[ll] = prod
            for ll in range(n_messages):
                num = 1.0 + X[ll]
                den = 1.0 - X[ll]
                if num == 0.0:
                    Lr[i, j, ll] = -1.0
                elif den == 0.0:
                    Lr[i, j, ll] = 1.0
                else:
                    Lr[i, j, ll] = np.log(num / den)
    for j in range(n):
        mj = nodes_values[j]
        ff = nodes_hist[j]
        sum_Lr = np.zeros(n_messages)
        for idx in range(ff):
            i = mj[idx]
            for ll in range(n_messages):
                sum_Lr[ll] += Lr[i, j, ll]
        for idx in range(ff):
            i = mj[idx]
            for ll in range(n_messages):
                Lq[i, j, ll] = Lc[j, ll] + (sum_Lr[ll] - Lr[i, j, ll])
    L_posteriori = np.empty((n, n_messages))
    for j in range(n):
        mj = nodes_values[j]
        ff = nodes_hist[j]
        for ll in range(n_messages):
            temp = Lc[j, ll]
            for idx in range(ff):
                i = mj[idx]
                temp += Lr[i, j, ll]
            L_posteriori[j, ll] = temp
    return Lq, Lr, L_posteriori

def get_message(tG, x):
    """Compute the original `n_bits` message from a `n_code` codeword `x`.

    Parameters
    ----------
    tG: array (n_code, n_bits) coding matrix tG.
    x: array (n_code,) decoded codeword of length `n_code`.

    Returns
    -------
    message: array (n_bits,). Original binary message.

    """
    n, k = tG.shape

    rtG, rx = utils.gausselimination(tG, x)

    message = np.zeros(k, dtype=int)

    message[k - 1] = rx[k - 1]
    for i in reversed(range(k - 1)):
        message[i] = rx[i]
        message[i] -= utils.binaryproduct(rtG[i, list(range(i + 1, k))],
                                          message[list(range(i + 1, k))])

    return abs(message)
