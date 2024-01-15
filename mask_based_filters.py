import numpy as np
import librosa as lb
from code_utils.se_utils.internal_formulas import intern_filter
from code_utils.se_utils.internal_formulas import spatial_correlation_matrix
from code_utils.mask_utils import wiener_mask
import ipdb


def masking(y, s, n, m, win_len=512, win_hop=256):
    y_stft = lb.core.stft(y, n_fft=win_len, hop_length=win_hop, center=True)
    s_stft = lb.core.stft(s, n_fft=win_len, hop_length=win_hop, center=True)
    n_stft = lb.core.stft(n, n_fft=win_len, hop_length=win_hop, center=True)

    m = np.pad(m, ((0, 0), (1, 1)), 'reflect')
    y_m = m*y_stft
    s_m = m*s_stft
    n_m = m*n_stft

    y_f = lb.core.istft(y_m, hop_length=win_hop, win_length=win_len, center=True, length=len(y))
    s_f = lb.core.istft(s_m, hop_length=win_hop, win_length=win_len, center=True, length=len(s))
    n_f = lb.core.istft(n_m, hop_length=win_hop, win_length=win_len, center=True, length=len(n))

    return y_f, s_f, n_f


def mwf(y, s, n, m, fs=16000, win_len=512, win_hop=256, mu=1):
    """
    Multichannel Wiener filter. Inputs are signal arrays, with one column per channel
    :param y:               Array of mixture signals
    :param s:
    :param n:
    :param m:               Masks list. One per signal
    :param win_len:
    :param win_hop:
    :param mu:              SD constant in SDW-MWF
    :return:
    """
    # Filter parameters same as DANSE
    nbsamples_per_sec = fs / (win_len / 2)
    lambda_cor = np.exp(np.log(0.5) / (2 * nbsamples_per_sec))
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    Rxx = np.zeros((n_freq, n_ch, n_ch), 'complex')
    Rnn = np.zeros((n_freq, n_ch, n_ch), 'complex')
    w = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_out = np.zeros(y.shape)
    s_out = np.zeros(y.shape)
    n_out = np.zeros(y.shape)

    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(y[:, i_ch], n_fft=win_len, hop_length=win_hop, center=False)
        s_stft[:, :, i_ch] = lb.core.stft(s[:, i_ch], n_fft=win_len, hop_length=win_hop, center=False)
        n_stft[:, :, i_ch] = lb.core.stft(n[:, i_ch], n_fft=win_len, hop_length=win_hop, center=False)

        # Input estimation
        s_stft_hat[:, :, i_ch] = m[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - m[i_ch]) * y_stft[:, :, i_ch]

    # Compute Rx, Rn with mask
    for i_frame in np.arange(n_frames):
        for i_freq in np.arange(n_freq):
            lambda_cor_ = np.minimum(lambda_cor, 1 - 1 / (i_frame + 1))
            Rxx[i_freq, :, :] = spatial_correlation_matrix(Rxx[i_freq, :, :], s_stft_hat[i_freq, i_frame, :],
                                                           lambda_cor=lambda_cor_, M=None)
            Rnn[i_freq, :, :] = spatial_correlation_matrix(Rnn[i_freq, :, :], n_stft_hat[i_freq, i_frame, :],
                                                           lambda_cor=lambda_cor_, M=None)

            try:
                w[i_freq, i_frame, :], _ = intern_filter(Rxx[i_freq, :, :], Rnn[i_freq, :, :],
                                                         mu=mu, type='gevd', rank=1)
            except np.linalg.linalg.LinAlgError:
                pass
            y_filt[i_freq, i_frame, :] = np.matmul(np.conjugate(w[i_freq, i_frame, :]), y_stft[i_freq, i_frame, :])
            s_filt[i_freq, i_frame, :] = np.matmul(np.conjugate(w[i_freq, i_frame, :]), s_stft[i_freq, i_frame, :])
            n_filt[i_freq, i_frame, :] = np.matmul(np.conjugate(w[i_freq, i_frame, :]), n_stft[i_freq, i_frame, :])

    for i_ch in range(n_ch):
        y_out[:, i_ch] = lb.core.istft(np.pad(y_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                       hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        s_out[:, i_ch] = lb.core.istft(np.pad(s_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                       hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        n_out[:, i_ch] = lb.core.istft(np.pad(n_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                       hop_length=win_hop, win_length=win_len, center=True, length=len(y))

    return y_out, s_out, n_out, m


def batch_mwf(y, s, n, m, win_len=512, win_hop=256, mu=1, filter_type='basic', recompute_mask=False):
    """
    Batch Multichannel Wiener filter, i.e. the covariance matrices are computed from the whole signal.
    Inputs are signal arrays, with one column per channel
    :param y:               Array of mixture signals
    :param s:
    :param n:
    :param m:               Masks list. One per signal
    :param win_len:
    :param win_hop:
    :param filter_type:     Filter computation (r1-MWF, GEVD, basic ?)
    :param mu:              SD constant in SDW-MWF
    :param recompute_mask:  Whether to recompute mask as wiener(S, N). If false, input is kep [False]
    :return:
    """
    # Input data parameters
    n_freq = int(win_len / 2 + 1)
    n_frames = int(1 + np.floor((len(y) - win_len) / win_hop))
    n_ch = y.shape[1]
    # Initialize variables
    y_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_stft_hat = np.zeros((n_freq, n_frames, n_ch), 'complex')
    r_ss = np.zeros((n_freq, n_ch, n_ch), 'complex')
    r_nn = np.zeros((n_freq, n_ch, n_ch), 'complex')
    w = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    s_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    n_filt = np.zeros((n_freq, n_frames, n_ch), 'complex')
    y_out = np.zeros(y.shape)
    s_out = np.zeros(y.shape)
    n_out = np.zeros(y.shape)

    for i_ch in range(n_ch):
        y_stft[:, :, i_ch] = lb.core.stft(y[:, i_ch], n_fft=win_len, hop_length=win_hop, center=False)
        s_stft[:, :, i_ch] = lb.core.stft(s[:, i_ch], n_fft=win_len, hop_length=win_hop, center=False)
        n_stft[:, :, i_ch] = lb.core.stft(n[:, i_ch], n_fft=win_len, hop_length=win_hop, center=False)

        # Input estimation
        if recompute_mask:
            m[i_ch] = wiener_mask(abs(s_stft[:, :, i_ch]), abs(n_stft[:, :, i_ch]), power=1)
        s_stft_hat[:, :, i_ch] = m[i_ch] * y_stft[:, :, i_ch]
        n_stft_hat[:, :, i_ch] = (1 - m[i_ch]) * y_stft[:, :, i_ch]

    # Compute Rx, Rn with mask
    for f in range(n_freq):
        phi_s_f = [[] for it in range(n_frames)]  # Covariance matrix at every frame
        phi_n_f = [[] for it in range(n_frames)]  # Covariance matrix at every frame
        for t in range(n_frames):
            phi_s_f[t] = np.outer(s_stft_hat[f, t, :], np.conjugate(s_stft_hat[f, t, :]).T)
            phi_n_f[t] = np.outer(n_stft_hat[f, t, :], np.conjugate(n_stft_hat[f, t, :]).T)
        r_ss[f] = np.mean(np.array(phi_s_f), axis=0)
        r_nn[f] = np.mean(np.array(phi_n_f), axis=0)
        w[f], _ = intern_filter(r_ss[f, :, :], r_nn[f, :, :], mu=mu, type=filter_type, rank=1)
        for i_frame in range(n_frames):
            y_filt[f, i_frame, :] = np.matmul(np.conjugate(w[f, i_frame, :]), y_stft[f, i_frame, :])
            s_filt[f, i_frame, :] = np.matmul(np.conjugate(w[f, i_frame, :]), s_stft[f, i_frame, :])
            n_filt[f, i_frame, :] = np.matmul(np.conjugate(w[f, i_frame, :]), n_stft[f, i_frame, :])

    for i_ch in range(n_ch):
        y_out[:, i_ch] = lb.core.istft(np.pad(y_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                       hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        s_out[:, i_ch] = lb.core.istft(np.pad(s_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                       hop_length=win_hop, win_length=win_len, center=True, length=len(y))
        n_out[:, i_ch] = lb.core.istft(np.pad(n_filt[:, :, i_ch], ((0, 0), (1, 1)), 'reflect'),
                                       hop_length=win_hop, win_length=win_len, center=True, length=len(y))

    return y_out, s_out, n_out, m
