#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Divide DANSE algorithm into two parts: initialize and filter. This enables to filter using different techniques (VAD,
oracle masks, predicted masks, etc) and to avoid repeating the lines in the scripts using DANSE.


[1] A. Bertrand, J. Callebaut and M. Moonen, "Adaptive distributed noise
    reduction for speech enhancement in wireless acoustic sensor networks",
    Proc. of the International Workshop on Acoustic Echo and Noise Control
    (IWAENC), Tel Aviv, Israel, Aug. 2010.

[2] Szurley J., Bertrand A., Moonen M., "Improved Tracking Performance for
    Distributed node-specific signal enhancement in wireless acoustic sensor
    networks", 2013.

    matlab script available on the website
      http://homes.esat.kuleuven.be/~abertran

"""


import numpy as np
from code_utils.se_utils import intern_filter


def get_node_params(Y, nb_nodes):
    """
    Number of mics/sensors in node k and available signals at node k (nb_mics_node + compressed signals)
    :param Y: Signal, only two first dims are important
    :return: nb_mics_node
    :return: dim_node
    """
    nb_mics_node, dim_node = [], []
    for k in range(nb_nodes):
        nb_mics_node.append(np.size(Y[k], 1))
        dim_node.append(nb_mics_node[k] + nb_nodes - 1)

    return nb_mics_node, dim_node


def initialize(Y, fs, L=None, hop_length=None):
    """
    Initialize local constants used in DANSE algorithm
    :param Y:
    :param fs:
    :param L:
    :param hop_length:
    :return:
    """

    if L is None:
        L = np.int(512 * fs / 16000)  # block length of 512 if sampling frequency is 16000
    if hop_length is None:
        hop_length = np.int(L / 2)

    nb_samples_per_sec = fs / (L / 2)  # number of DFT blocks per second
    # Forgetting factors
    lambda_cor = np.exp(np.log(0.5) / (2 * nb_samples_per_sec))
    lambda_ext = np.exp(np.log(0.5) / (0.2 * nb_samples_per_sec))

    # Update rate
    min_nb_samples = 3 * nb_samples_per_sec

    nb_nodes = len(Y)  # number of nodes
    length_signal = np.size(Y[0], 0)  # length of the audio signal
    Wint, Wext, Wext_target, Y_out = [], [], [], []

    S_out, N_out = [], []
    Rnn, Rxx = list([] for i in np.arange(0, nb_nodes)), \
               list([] for i in np.arange(0, nb_nodes)), \
 \
    # Estimated freq signals
    Yest = np.zeros((np.int(L / 2 + 1), 1), 'complex')  # signal estimate of current block
    Yest_S = np.zeros((np.int(L / 2 + 1), 1), 'complex')  # Target speech output (for phantom path)
    Yest_N = np.zeros((np.int(L / 2 + 1), 1), 'complex')  # Noise output (for phantom path)

    nb_mics_node, dim_node = get_node_params(Y, nb_nodes)

    for k in np.arange(0, nb_nodes):
        # filters
        Wint.append(np.zeros((dim_node[k], np.int(L / 2 + 1)), 'complex'))
        Wext.append(np.ones((nb_mics_node[k], np.int(L / 2 + 1)), 'complex'))
        Wext_target.append(np.ones((nb_mics_node[k], np.int(L / 2 + 1)), 'complex'))
        # Correlation matrices
        for u in np.arange(0, L / 2 + 1):
            Rnn[k].append(np.zeros((dim_node[k], dim_node[k]), 'complex'))  # Noise covariance matrix
            Rxx[k].append(np.zeros((dim_node[k], dim_node[k]), 'complex'))  # Speech covariance matrix

        # Output time signals
        Y_out.append(np.zeros((length_signal, 1)))
        S_out.append(np.zeros((length_signal, 1)))
        N_out.append(np.zeros((length_signal, 1)))

    # Hanning window
    Han = np.hanning(L).reshape((np.int(L), 1)) * np.ones((1, np.max(nb_mics_node)))

    return L, length_signal, hop_length, min_nb_samples, nb_nodes, nb_mics_node, Han, lambda_cor, lambda_ext, \
           Rxx, Rnn, Wint, Wext, Wext_target, Yest, Yest_S, Yest_N, Y_out, S_out, N_out


def compute_stft_at_iter(sig, L, Han, i_iter):
    """
    Compute the STFT frame of sig at iteration iter (length L)
    :param sig:         Signal to compute the STFT of
    :param L:           Length of FFT window
    :param Han:         Window to apply to signal before FFT computation
    :param i_iter:      iteration number / frame
    :return sig_block:  FFT of sig at frame iter
    """
    nb_nodes = len(sig)
    nb_mics_node, _ = get_node_params(sig, nb_nodes)

    sig_block = []
    for k in np.arange(nb_nodes):
        sig_block.append(np.fft.fft(np.sqrt(Han[:, 0:nb_mics_node[k]])
                                    * sig[k][i_iter:np.int(i_iter + L), :], axis=0).T)
        sig_block[k] = sig_block[k][:, 0:np.int(L / 2 + 1)]

    return sig_block


def filter_with_mask(i_iter, L, alpha, mu, f_type, rank, i_frame, i_samples, min_nb_samples,
                     Han, lambda_cor, lambda_ext,
                     Yblock, Sblock, Nblock,
                     M,
                     Rxx, Rnn, Wint, Wext, Wext_target,
                     Yest, Yest_S, Yest_N, Y_out, S_out, N_out,
                     nb_updates, simultaneous, update_token):

    nb_nodes = len(Yblock)
    nb_mics_node, _ = get_node_params(np.swapaxes(np.array(Wext), 1, 2), nb_nodes)

    # Stack masks in nodes; Compute compressed signals
    Mblock = list([] for i in np.arange(0, nb_nodes))  # Block of mask values at frame i_iter
    Zblock = np.zeros((np.int(nb_nodes), np.int(np.int(L / 2 + 1))), dtype=complex)
    Zblock_S = np.zeros((np.int(nb_nodes), np.int(np.int(L / 2 + 1))), dtype=complex)
    Zblock_N = np.zeros((np.int(nb_nodes), np.int(np.int(L / 2 + 1))), dtype=complex)
    In = []
    In_S, In_N = [], []
    In_s, In_n = [], []

    for k in range(nb_nodes):
        Mblock[k] = np.tile(M[k], [nb_mics_node[k], 1])  # Same mask for all mics in node
        for u in np.arange(np.int(L / 2 + 1)):
            Zblock[k, u] = np.matmul(np.conjugate(Wext[k][:, u]).T, Yblock[k][:, u])
            Zblock_S[k, u] = np.matmul(np.conjugate(Wext[k][:, u]).T, Sblock[k][:, u])
            Zblock_N[k, u] = np.matmul(np.conjugate(Wext[k][:, u]).T, Nblock[k][:, u])

    # Compute internal optimal filters at each node
    for k in np.arange(nb_nodes):
        Zk = np.concatenate((Zblock[0:k, :], Zblock[k + 1:, :]))  # Remove data of node k
        Zk_S = np.concatenate((Zblock_S[0:k, :], Zblock_S[k + 1:, :]))  # Remove data of node k
        Zk_N = np.concatenate((Zblock_N[0:k, :], Zblock_N[k + 1:, :]))  # Remove data of node k
        In.append(np.concatenate((Yblock[k], Zk)))  # inputs of node k
        In_S.append(np.concatenate((Sblock[k], Zk_S)))  # inputs of node k
        In_N.append(np.concatenate((Nblock[k], Zk_N)))  # inputs of node k
        # Stack masks corresponding to signals in In: local signals + signals coming from 'previous' and 'further' nodes
        Mblockk = np.vstack((np.array(Mblock)[k],  # nb_mics_nodes[k] masks for as many local signals ...
                             np.array(M)[:k],  # ... stacked with one mask per node before current node ...
                             np.array(M)[k + 1:]))  # ... and one mask per node after current node
        In_s.append(Mblockk * In[k])
        In_n.append((1 - Mblockk) * In[k])

        for u in np.arange(np.int(L / 2 + 1)):
            # Autocorrelation matrices
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[k][u] = lambda_cor_ * Rxx[k][u] \
                        + (1 - lambda_cor_) * np.outer(In_s[k][:, u], np.conjugate(In_s[k][:, u]).T)
            Rnn[k][u] = lambda_cor_ * Rnn[k][u] \
                        + (1 - lambda_cor_) * np.outer(In_n[k][:, u], np.conjugate(In_n[k][:, u]).T)

            # Intern filters
            try:
                Wint[k][:, u] = intern_filter(Rxx[k][u], Rnn[k][u], mu=mu, type=f_type, rank=rank)
            except np.linalg.linalg.LinAlgError:  # At the beginning, Rnn is not positive definite
                pass  # Do not update Wint then

            # take small step towards Wext_target
            lambda_ext_ = np.minimum(lambda_ext, 1 - 1 / (i_frame + 1))
            Wext[k][:, u] = lambda_ext_ * Wext[k][:, u] + (1 - lambda_ext_) * Wext_target[k][:, u]

    # DANSE update of external filters
    # only update when there are sufficient fresh samples for BOTH Ryy and Rnn
    if i_samples >= min_nb_samples:
        # reset counters
        i_samples = 0
        nb_updates += 1

        # perform updates
        if simultaneous == 0:  # Sequential node updating
            Wext_target[update_token] = (1 - alpha) * Wext_target[update_token] \
                                       + alpha * Wint[update_token][0:nb_mics_node[update_token], :]
            update_token = np.remainder(update_token, nb_nodes) + 1
        elif simultaneous == 1:  # Simultaneous node updating
            for k in np.arange(nb_nodes):
                Wext_target[k] = (1 - alpha) * Wext_target[k] + alpha * Wint[k][0:nb_mics_node[k], :]
        # print('New target(s) for external filters after ' + str(i_iter / fs) + ' seconds')

    # Compute node-specific output at all nodes
    for k in np.arange(nb_nodes):
        for v in np.arange(np.int(L / 2 + 1)):
            Yest[v] = np.matmul(np.conjugate(Wint[k][:, v]).T, In[k][:, v])
            Yest_S[v] = np.matmul(np.conjugate(Wint[k][:, v]).T, In_S[k][:, v])
            Yest_N[v] = np.matmul(np.conjugate(Wint[k][:, v]).T, In_N[k][:, v])
        y_block_est = np.real(
            np.fft.ifft(np.concatenate((Yest, np.conj(np.flipud(Yest[1:np.int(L / 2)])))), axis=0))
        s_block_est = np.real(
            np.fft.ifft(np.concatenate((Yest_S, np.conj(np.flipud(Yest_S[1:np.int(L / 2)])))), axis=0))
        n_block_est = np.real(
            np.fft.ifft(np.concatenate((Yest_N, np.conj(np.flipud(Yest_N[1:np.int(L / 2)])))), axis=0))
        Y_out[k][i_iter:np.int(i_iter + L)] += np.reshape(np.sqrt(Han[:, 0]), (len(Han[:, 0]), 1)) * y_block_est
        S_out[k][i_iter:np.int(i_iter + L)] += np.reshape(np.sqrt(Han[:, 0]), (len(Han[:, 0]), 1)) * s_block_est
        N_out[k][i_iter:np.int(i_iter + L)] += np.reshape(np.sqrt(Han[:, 0]), (len(Han[:, 0]), 1)) * n_block_est

    return i_samples, \
           Rxx, Rnn, Wint, Wext, Wext_target, \
           Yest, Yest_S, Yest_N, Y_out, S_out, N_out, \
           nb_updates, update_token, Zblock


#%% ##################################### SUBFUNCTIONS FOR SHARING Z ###################################################
def compute_z_at_iter(L, Yblock, Wext):
    """
    :param L:           Window/frame length
    :param Yblock:      STFT frames of mixture signal at all nodes
    :param Wext:        Filter used to compute compressed signal that will be sent to other nodes
    :return Zcomp:      Compressed STFT frames at all nodes
    """

    nb_nodes = len(Yblock)
    Zblock = np.zeros((np.int(nb_nodes), np.int(np.int(L / 2 + 1))), dtype=complex)

    for k in range(nb_nodes):
        for v in range(np.int(L / 2) + 1):
            Zblock[k, v] = np.matmul(np.conjugate(Wext[k][:, v]).T, Yblock[k][:, v])


def filter_with_mask_and_z(i_iter, L, alpha, mu, f_type, rank,
                           i_frame, i_samples, min_nb_samples,
                           Han, lambda_cor, lambda_ext,
                           Yblock, Sblock, Nblock, Zblock,
                           M,
                           Rxx, Rnn, Wint, Wext, Wext_target,
                           Yest, Yest_S, Yest_N, Y_out, S_out, N_out,
                           nb_updates, simultaneous, update_token):

    nb_nodes = len(Yblock)
    nb_mics_node, _ = get_node_params(np.swapaxes(np.array(Wext), 1, 2), nb_nodes)

    # Stack masks in nodes; Compute compressed signals
    Mblock = list([] for i in np.arange(0, nb_nodes))  # Block of mask values at frame i_iter
    Zblock_S = np.zeros((np.int(nb_nodes), np.int(np.int(L / 2 + 1))), dtype=complex)
    Zblock_N = np.zeros((np.int(nb_nodes), np.int(np.int(L / 2 + 1))), dtype=complex)
    In = []
    In_S, In_N = [], []
    In_s, In_n = [], []

    for k in range(nb_nodes):
        Mblock[k] = np.tile(M[k], [nb_mics_node[k], 1])  # Same mask for all mics in node
        for u in np.arange(np.int(L / 2 + 1)):
            Zblock_S[k, u] = np.matmul(np.conjugate(Wext[k][:, u]).T, Sblock[k][:, u])
            Zblock_N[k, u] = np.matmul(np.conjugate(Wext[k][:, u]).T, Nblock[k][:, u])

    # Compute internal optimal filters at each node
    for k in np.arange(nb_nodes):
        Zk = np.concatenate((Zblock[0:k, :], Zblock[k + 1:, :]))  # Remove data of node k
        Zk_S = np.concatenate((Zblock_S[0:k, :], Zblock_S[k + 1:, :]))  # Remove data of node k
        Zk_N = np.concatenate((Zblock_N[0:k, :], Zblock_N[k + 1:, :]))  # Remove data of node k
        In.append(np.concatenate((Yblock[k], Zk)))  # inputs of node k
        In_S.append(np.concatenate((Sblock[k], Zk_S)))  # inputs of node k
        In_N.append(np.concatenate((Nblock[k], Zk_N)))  # inputs of node k
        # Stack masks corresponding to signals in In: local signals + signals coming from 'previous' and 'further' nodes
        Mblockk = np.vstack((np.array(Mblock)[k],       # nb_mics_nodes[k] masks for as many local signals ...
                             np.array(M)[:k],           # ... stacked with one mask per node before current node ...
                             np.array(M)[k + 1:]))      # ... and one mask per node after current node
        In_s.append(Mblockk * In[k])
        In_n.append((1 - Mblockk) * In[k])

        for u in np.arange(np.int(L / 2 + 1)):
            # Autocorrelation matrices
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[k][u] = lambda_cor_ * Rxx[k][u] \
                        + (1 - lambda_cor_) * np.outer(In_s[k][:, u], np.conjugate(In_s[k][:, u]).T)
            Rnn[k][u] = lambda_cor_ * Rnn[k][u] \
                        + (1 - lambda_cor_) * np.outer(In_n[k][:, u], np.conjugate(In_n[k][:, u]).T)

            # Intern filters
            try:
                Wint[k][:, u] = intern_filter(Rxx[k][u], Rnn[k][u], mu=mu, type=f_type, rank=rank)
            except np.linalg.linalg.LinAlgError:  # At the beginning, Rnn is not positive definite
                pass  # Do not update Wint then

            # take small step towards Wext_target
            lambda_ext_ = np.minimum(lambda_ext, 1 - 1 / (i_frame + 1))
            Wext[k][:, u] = lambda_ext_ * Wext[k][:, u] + (1 - lambda_ext_) * Wext_target[k][:, u]

        # DANSE update of external filters
        # only update when there are sufficient fresh samples for BOTH Ryy and Rnn
        if i_samples >= min_nb_samples:
            # reset counters
            i_samples = 0
            nb_updates += 1

            # perform updates
            if simultaneous == 0:  # Sequential node updating
                Wext_target[update_token] = (1 - alpha) * Wext_target[update_token] \
                                            + alpha * Wint[update_token][0:nb_mics_node[update_token], :]
                update_token = np.remainder(update_token, nb_nodes) + 1
            elif simultaneous == 1:  # Simultaneous node updating
                for k in np.arange(nb_nodes):
                    Wext_target[k] = (1 - alpha) * Wext_target[k] + alpha * Wint[k][0:nb_mics_node[k], :]
            # print('New target(s) for external filters after ' + str(i_iter / fs) + ' seconds')

        # Compute node-specific output at all nodes
        for k in np.arange(nb_nodes):
            for v in np.arange(np.int(L / 2 + 1)):
                Yest[v] = np.matmul(np.conjugate(Wint[k][:, v]).T, In[k][:, v])
                Yest_S[v] = np.matmul(np.conjugate(Wint[k][:, v]).T, In_S[k][:, v])
                Yest_N[v] = np.matmul(np.conjugate(Wint[k][:, v]).T, In_N[k][:, v])
            y_block_est = np.real(
                np.fft.ifft(np.concatenate((Yest, np.conj(np.flipud(Yest[1:np.int(L / 2)])))), axis=0))
            s_block_est = np.real(
                np.fft.ifft(np.concatenate((Yest_S, np.conj(np.flipud(Yest_S[1:np.int(L / 2)])))), axis=0))
            n_block_est = np.real(
                np.fft.ifft(np.concatenate((Yest_N, np.conj(np.flipud(Yest_N[1:np.int(L / 2)])))), axis=0))
            Y_out[k][i_iter:np.int(i_iter + L)] += np.reshape(np.sqrt(Han[:, 0]), (len(Han[:, 0]), 1)) * y_block_est
            S_out[k][i_iter:np.int(i_iter + L)] += np.reshape(np.sqrt(Han[:, 0]), (len(Han[:, 0]), 1)) * s_block_est
            N_out[k][i_iter:np.int(i_iter + L)] += np.reshape(np.sqrt(Han[:, 0]), (len(Han[:, 0]), 1)) * n_block_est

    return i_samples, \
           Rxx, Rnn, Wint, Wext, Wext_target, \
           Yest, Yest_S, Yest_N, Y_out, S_out, N_out, \
           nb_updates, update_token
