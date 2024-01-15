"""
Main functions, as used by Bertrand. Second one is similar, but with VAD replaced by mask
"""

from code_utils.se_utils.internal_formulas import intern_filter
import numpy as np


def danse_vad(Y, onoff, fs, L=None, simultaneous=1, alpha=0.7, mu=1, f_type='r1-mwf', f_rank='Full'):
    """
    Classical WOLA_DANSE as described by [1] using VAD.
    :param Y:
    :param onoff:
    :param fs:
    :param L:
    :param simultaneous:
    :param alpha:
    :param mu:
    :param f_type:
    :param f_rank:
    :return:
    """
    if L is None:
        L = np.int(512 * fs / 16000)  # block length of 512 if sampling frequency is 16000

    # %% hardcoded parameters
    nbsamples_per_sec = fs / (L / 2)  # number of DFT blocks per second

    # Determines how fast older samples are forgotten:
    # forgettting factor for estimation of correlation matrices (should be 0<lambda<=1)
    # (default: samples from 2 seconds in the past are weighted with 0.5)
    lambda_cor = np.exp(np.log(0.5) / (2 * nbsamples_per_sec))
    # smoothing of external filter updates (set to zero for non-smooth updates, see code for more details)
    lambda_ext = np.exp(np.log(0.5) / (0.2 * nbsamples_per_sec))

    # Determines update rate of DANSE
    # required number of new samples for both Ryy and Rnn before a new DANSE update can be performed
    # (default: number of blocks in 3 seconds)
    min_nb_samples = 3 * nbsamples_per_sec

    saveperiod = 20  # every time this amount of seconds passed in the simulated signal, the results are saved
    plotresults = 0  # set to 1 if you want to plot results during simulation
    shownode = 0  # the node for which results are shown during simulation
    fname = 'results'  # name for save file

    # %% Initialisation
    nbnodes = len(Y)  # number of nodes
    lengthsignal = np.size(Y[0], 0)  # length of the audio signal
    Ryysamples = np.zeros((nbnodes, ))  # counts the number of samples used to estimate Ryy since last DANSE update
    Ryysamples_ = np.zeros((nbnodes, ))
    Rnnsamples = np.zeros((nbnodes, ))  # counts the number of samples used to estimate Rnn since last DANSE update
    Rnnsamples_ = np.zeros((nbnodes, ))
    nbmicsnode, dimnode, Wint, Wext, Wext_target, Y_out = [], [], [], [], [], []
    Ryy, Rnn, Rxx, Rnninv = list([] for i in np.arange(0, nbnodes)), list([] for i in np.arange(0, nbnodes)), list(
        [] for i in np.arange(0, nbnodes)), list([] for i in np.arange(0, nbnodes))

    for k in np.arange(0, nbnodes):
        nbmicsnode.append(np.size(Y[k], 1))  # Number of mics/sensors in node k
        # dimension of the local estimation problem (for DANSE_1: =M_k+number of neighbors of node k)
        dimnode.append(nbmicsnode[k] + nbnodes - 1)
        # filter internally applied to estimate signal
        Wint.append(np.zeros((dimnode[k], np.int(L / 2 + 1)), 'complex'))
        Wint[k][0, :] = 1
        # actual filter applied for broadcast signals (changes smoothly towards Wext_target{k},
        # based on exponential weighting with lambda_ext)
        Wext.append(np.ones((nbmicsnode[k], np.int(L / 2 + 1)), 'complex'))
        # target filter applied for broadcast signals (at each DANSE-update of node k,
        # Wext_target{k} is changed to Wint{k})
        Wext_target.append(np.ones((nbmicsnode[k], np.int(L / 2 + 1)), 'complex'))
        for u in np.arange(0, L / 2 + 1):
            Ryy[k].append(np.eye(dimnode[k]))  # Signal covariance matrix
            Rnn[k].append(np.eye(dimnode[k]))  # Noise covariance matrix
            Rxx[k].append(np.zeros((dimnode[k], dimnode[k]), 'complex'))        # Speech covariance matrix
            Rnninv[k].append(np.zeros((dimnode[k], dimnode[k]), 'complex'))     # Noise covariance inverse matrix

        Y_out.append(np.zeros((lengthsignal, 1)))

    updatetoken = 0     # The node that can perform the next update (only for sequential updating)
    Han = np.hanning(L).reshape((np.int(L), 1)) * np.ones((1, np.max(nbmicsnode)))
    startupdating = np.zeros((nbnodes, ))   # flag to determine when the internal filters can start updating
    speech_active = np.zeros((nbnodes, ))

    Yest = np.zeros((np.int(L / 2 + 1), 1), 'complex')  # signal estimate of current block
    count = 0           # counts number of blocks that have been processed
    count2 = 0          # counts number of signal samples that have been processed

    # %% WOLA_DANSE algorithm
    for iter in np.arange(0, np.int(lengthsignal - L), np.int(L / 2)):
        Yblock = []
        Zblock = np.zeros((np.int(nbnodes), np.int(np.int(L / 2 + 1))), dtype=complex)
        In = []

        count += 1
        count2 += L / 2

        # create z-signals
        # REMARK: Note that the data exchange between nodes is doubled here
        # due to redundancy caused by the 50% overlap between frames.
        # However, in practice, one can transmit (finalized) samples of the
        # time-domain signal 'estimation{k}' (see below), rather than the
        # FFT blocks themselves, which avoids this redundancy in the data
        # exchange.
        for k in np.arange(nbnodes):
            Yblock.append(np.fft.fft(np.sqrt(Han[:, 0:nbmicsnode[k]]) * Y[k][iter:np.int(iter + L), :], axis=0).T)
            Yblock[k] = Yblock[k][:, 0:np.int(L / 2 + 1)]
            for u in np.arange(np.int(L / 2 + 1)):
                Zblock[k, u] = np.matmul(np.conjugate(Wext[k][:, u]).T, Yblock[k][:, u])

        # Determine if current block contains speech or not
        for k in np.arange(nbnodes):
            if onoff[k][count - 1] == 1:  # Speech is active
                speech_active[k] = 1
                Ryysamples[k] += 1
                Ryysamples_[k] += 1
            else:
                speech_active[k] = 0
                Rnnsamples[k] += 1
                Rnnsamples_[k] += 1

            # Check when the noise reduction filters can be updated (needs minimum
            # number of samples for both Ryy and Rnn)
            # Only update internal filters if sufficient samples have been collected
            if startupdating[k] == 0 and (Ryysamples[k] > np.max(nbmicsnode) + nbnodes - 1
                                          and Rnnsamples[k] > np.max(nbmicsnode) + nbnodes - 1):
                print('Filters started updating at node ' + str(k) + ' after ' + str(iter / fs) + ' seconds')
                startupdating[k] = 1
                Ryysamples[k] = 0
                Rnnsamples[k] = 0

        # Compute internal optimal filters at each node
        for k in np.arange(nbnodes):
            Zk = np.concatenate((Zblock[0:k, :], Zblock[k + 1:, :]))    # Remove data of node k
            In.append(np.concatenate((Yblock[k], Zk)))                  # inputs of node k
            for u in np.arange(np.int(L / 2 + 1)):
                if speech_active[k] == 1:
                    lambda_cor_y = np.minimum(lambda_cor, 1 - 1/(Ryysamples_[k]+1))
                    Ryy[k][u] = lambda_cor_y * Ryy[k][u] \
                                + (1 - lambda_cor_y) * np.outer(In[k][:, u], np.conjugate(In[k][:, u]).T)
                else:
                    lambda_cor_n = np.minimum(lambda_cor, 1 - 1/(Rnnsamples_[k]+1))
                    Rnn[k][u] = lambda_cor_n * Rnn[k][u] \
                                + (1 - lambda_cor_n) * np.outer(In[k][:, u], np.conjugate(In[k][:, u]).T)
                    # if Rnninv_initialized == 1:
                    #     Rnninv[k][u] = np.linalg.inv(Rnn[k][u])
                if startupdating[k] == 1:  # Do not update Wint in the beginning
                    Wint[k][:, u], _ = intern_filter(Ryy[k][u], Rnn[k][u], mu=mu, type=f_type, rank=f_rank)

                    # take small step towards Wext_target
                    lambda_ext_ = np.minimum(lambda_ext, 1 - 1 / Ryysamples_[k])
                    Wext[k][:, u] = lambda_ext_ * Wext[k][:, u] + (1 - lambda_ext_) * Wext_target[k][:, u]

        # DANSE update of external filters
        # Update when there are sufficient fresh samples for BOTH Ryy and Rnn at first node
        if Ryysamples[0] >= min_nb_samples and Rnnsamples[0] >= min_nb_samples:
            # perform updates
            if simultaneous == 0:  # Sequential node updating
                Wext_target[updatetoken] = (1 - alpha) * Wext_target[updatetoken] \
                                           + alpha * Wint[updatetoken][0:nbmicsnode[updatetoken], :]
                updatetoken = np.remainder(updatetoken, nbnodes) + 1
            elif simultaneous == 1:  # Simultaneous node updating
                for k in np.arange(nbnodes):
                    Wext_target[k] = (1 - alpha) * Wext_target[k] + alpha * Wint[k][0:nbmicsnode[k], :]
            print('New target(s) for external filters after ' + str(iter / fs) + ' seconds')

        # Reset counters at both nodes
        for k in np.arange(nbnodes):
            if Ryysamples[k] >= min_nb_samples and Rnnsamples[k] >= min_nb_samples:
                if startupdating[k] == 0:
                    print('min_nb_samples at node ' + str(k) + ' is smaller than max(nbmicsnode)+nbnodes-1, '
                                                               'no updates will be performed')
                # reset counters
                Ryysamples[k] = 0
                Rnnsamples[k] = 0

        # Compute node-specific output at all nodes
        for k in np.arange(nbnodes):
            for v in np.arange(np.int(L / 2 + 1)):
                Yest[v] = np.matmul(np.conjugate(Wint[k][:, v]).T, In[k][:, v])
                y_block_est = np.real(np.fft.ifft(np.concatenate((Yest,
                                                                  np.conj(np.flipud(Yest[1:np.int(L / 2)])))), axis=0))
            Y_out[k][iter:np.int(iter + L)] += np.reshape(np.sqrt(Han[:, 0]), (len(Han[:, 0]), 1)) * y_block_est

    return Y_out


def danse_mask(Y, M, fs, L=None, hop_length=None, simultaneous=1, alpha=0.7, mu=1, f_type='r1-mwf', rank='Full'):
    """
    WOLA-DANSE [1] where VAD is replaced by a T-F mask
    :param Y:
    :param M:
    :param fs:
    :param L:
    :param hop_length:
    :param simultaneous:
    :param alpha:
    :param mu:
    :param f_type:
    :param rank:
    :return:
    """
    if L is None:
        L = np.int(512 * fs / 16000)  # block length of 512 if sampling frequency is 16000
    if hop_length is None:
        hop_length = np.int(L / 2)

    # %% hardcoded parameters
    nbsamples_per_sec = fs / (L / 2)  # number of DFT blocks per second

    # Determines how fast older samples are forgotten:
    # forgettting factor for estimation of correlation matrices (should be 0<lambda<=1)
    # (default: samples from 2 seconds in the past are weighted with 0.5)
    lambda_cor = np.exp(np.log(0.5) / (2 * nbsamples_per_sec))
    # lambda_cor = np.exp(np.log(0.001) / (2 * nbsamples_per_sec))
    # smoothing of external filter updates (set to zero for non-smooth updates, see code for more details)
    lambda_ext = np.exp(np.log(0.5) / (0.2 * nbsamples_per_sec))
    # lambda_ext = np.exp(np.log(0.01) / (0.2 * nbsamples_per_sec))

    # Determines update rate of DANSE
    # required number of new samples for both Ryy and Rnn before a new DANSE update can be performed
    # (default: number of blocks in 3 seconds)
    min_nb_samples = 3 * nbsamples_per_sec

    saveperiod = 20  # every time this amount of seconds passed in the simulated signal, the results are saved
    plotresults = 0  # set to 1 if you want to plot results during simulation
    shownode = 0  # the node for which results are shown during simulation
    fname = 'results'  # name for save file

    # %% Initialisation
    nbnodes = len(Y)  # number of nodes
    lengthsignal = np.size(Y[0], 0)  # length of the audio signal
    i_samples = 0   # counts the number of samples used to estimate Ryy and Rnn since last DANSE update
    i_iter = 0      # counts number of iterations since start of algorithm
    # Rnnsamples = 0  # counts the number of samples used to estimate Rnn since last DANSE update
    nbmicsnode, dimnode, Wint, Wext, Wext_target, Y_out = [], [], [], [], [], []
    Rnn, Rxx, Rnninv = list([] for i in np.arange(0, nbnodes)), \
                       list([] for i in np.arange(0, nbnodes)), \
                       list([] for i in np.arange(0, nbnodes))

    for k in np.arange(0, nbnodes):
        nbmicsnode.append(np.size(Y[k], 1))  # Number of mics/sensors in node k
        # dimension of the local estimation problem (for DANSE_1: =M_k+number of neighbors of node k)
        dimnode.append(nbmicsnode[k] + nbnodes - 1)
        # filter internally applied to estimate signal
        Wint.append(np.zeros((dimnode[k], np.int(L / 2 + 1)), 'complex'))
        # Wint[k][0, :] = 1
        # actual filter applied for broadcast signals (changes smoothly towards Wext_target{k},
        # based on exponential weighting with lambda_ext)
        Wext.append(np.ones((nbmicsnode[k], np.int(L / 2 + 1)), 'complex'))
        # target filter applied for broadcast signals (at each DANSE-update of node k,
        # Wext_target{k} is changed to Wint{k})
        Wext_target.append(np.ones((nbmicsnode[k], np.int(L / 2 + 1)), 'complex'))
        for u in np.arange(0, L / 2 + 1):
            Rnn[k].append(np.zeros((dimnode[k], dimnode[k]), 'complex'))  # Noise covariance matrix
            Rxx[k].append(np.zeros((dimnode[k], dimnode[k]), 'complex'))  # Speech covariance matrix
            Rnninv[k].append(np.zeros((dimnode[k], dimnode[k]), 'complex'))  # Noise covariance inverse matrix

        Y_out.append(np.zeros((lengthsignal, 1)))

    updatetoken = 0  # The node that can perform the next update (only for sequential updating)
    Han = np.hanning(L).reshape((np.int(L), 1)) * np.ones((1, np.max(nbmicsnode)))

    teller = 0  # counts percentage of signal that has been processed already
    nbupdates = 0  # counts number of DANSE updates that have been performed
    Yest = np.zeros((np.int(L / 2 + 1), 1), 'complex')  # signal estimate of current block

    # %% WOLA_DANSE algorithm
    for iter in np.arange(0, np.int(lengthsignal - L), hop_length):
        i_frame = np.int(iter / hop_length)
        Yblock = []
        Mblock = list([] for i in np.arange(0, nbnodes))  # Block of mask values at frame iter
        Zblock = np.zeros((np.int(nbnodes), np.int(np.int(L / 2 + 1))), dtype=complex)
        In = []
        In_s, In_n = [], []

        i_samples += 1
        i_iter += 1

        # create z-signals
        # REMARK: Note that the data exchange between nodes is doubled here
        # due to redundancy caused by the 50% overlap between frames.
        # However, in practice, one can transmit (finalized) samples of the
        # time-domain signal 'estimation{k}' (see below), rather than the
        # FFT blocks themselves, which avoids this redundancy in the data
        # exchange.
        for k in np.arange(nbnodes):
            Yblock.append(np.fft.fft(np.sqrt(Han[:, 0:nbmicsnode[k]]) * Y[k][iter:np.int(iter + L), :], axis=0).T)
            Yblock[k] = Yblock[k][:, 0:np.int(L / 2 + 1)]
            Mblock[k] = np.tile(M[k][:, i_frame], [nbmicsnode[k], 1])  # Select mask frame
            for u in np.arange(np.int(L / 2 + 1)):
                Zblock[k, u] = np.matmul(np.conjugate(Wext[k][:, u]).T, Yblock[k][:, u])

        # Compute internal optimal filters at each node
        for k in np.arange(nbnodes):
            Zk = np.concatenate((Zblock[0:k, :], Zblock[k + 1:, :]))  # Remove data of node k
            In.append(np.concatenate((Yblock[k], Zk)))  # inputs of node k
            Mblockk = np.vstack((np.array(Mblock)[k],
                                 np.array(M)[:k][:, :, i_frame],
                                 np.array(M)[k + 1:][:, :, i_frame]))
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
                    Wint[k][:, u], _ = intern_filter(Rxx[k][u], Rnn[k][u], mu=mu, type=f_type, rank=rank)
                except np.linalg.linalg.LinAlgError:    # At the beginning, Rnn is not positive definite
                    pass                                # Do not update Wint then

                # take small step towards Wext_target
                lambda_ext_ = np.minimum(lambda_ext, 1 - 1 / (i_frame + 1))
                Wext[k][:, u] = lambda_ext_ * Wext[k][:, u] + (1 - lambda_ext_) * Wext_target[k][:, u]

        # DANSE update of external filters
        # only update when there are sufficient fresh samples for BOTH Ryy and Rnn
        if i_samples >= min_nb_samples:
            # reset counters
            i_samples = 0
            nbupdates += 1

            # perform updates
            if simultaneous == 0:  # Sequential node updating
                Wext_target[updatetoken] = (1 - alpha) * Wext_target[updatetoken] \
                                           + alpha * Wint[updatetoken][0:nbmicsnode[updatetoken], :]
                updatetoken = np.remainder(updatetoken, nbnodes) + 1
            elif simultaneous == 1:  # Simultaneous node updating
                for k in np.arange(nbnodes):
                    Wext_target[k] = (1 - alpha) * Wext_target[k] + alpha * Wint[k][0:nbmicsnode[k], :]
            print('New target(s) for external filters after ' + str(iter / fs) + ' seconds')

        # Compute node-specific output at all nodes
        for k in np.arange(nbnodes):
            for v in np.arange(np.int(L / 2 + 1)):
                Yest[v] = np.matmul(np.conjugate(Wint[k][:, v]).T, In[k][:, v])
            blockest = np.real(np.fft.ifft(np.concatenate((Yest, np.conj(np.flipud(Yest[1:np.int(L / 2)])))), axis=0))
            Y_out[k][iter:np.int(iter + L)] += np.reshape(np.sqrt(Han[:, 0]), (len(Han[:, 0]), 1)) * blockest

        if 100 * (iter + 1) / len(Y[0][:, 0]) > teller:
            print(str(teller) + '% processed')
            teller += 1

    return Y_out
