import numpy as np
np.random.seed(10)
import pyroomacoustics as pra
import soundfile as sf
import sys
from code_utils.math_utils import floor_to_multiple, db2lin
from code_utils.sigproc_utils import vad_oracle_batch
from code_utils.metrics import snr, seg_snr
from code_utils.db_utils import *
import math
import glob
import time

import json

with open('config_rir_bigroom.json', 'r') as f:
    config_rir = json.load(f)

snr_min= config_rir["miscellaneous properties"]['snr_min']
snr_max= config_rir["miscellaneous properties"]['snr_max']
beta_range= config_rir["miscellaneous properties"]['beta_range']
early_reverb_len= config_rir["miscellaneous properties"]['early_reverb_len']
n_mics= config_rir["miscellaneous properties"]['n_mics']

len_min= config_rir["Room geometry"]['len_min']
len_max= config_rir["Room geometry"]['len_max']
wid_min= config_rir["Room geometry"]['wid_min']
wid_max= config_rir["Room geometry"]['wid_max']
hei_min= config_rir["Room geometry"]['hei_min']
hei_max=config_rir["Room geometry"]['hei_max']

beta_min= config_rir["Room acoustics"]['beta_min']
beta_max= config_rir["Room acoustics"]['beta_max']

d= config_rir["Sensor positions"]['d']
d_wal=config_rir["Sensor positions"]['d_wal']
d_mic= config_rir["Sensor positions"]['d_mic']
z_cst= config_rir["Sensor positions"]['z_cst']
d_rnd_mics= config_rir["Sensor positions"]['d_rnd_mics']

d_sou= config_rir["Source properties"]["d_sou"]
d_sou_wal= config_rir["Source properties"]["d_sou_wal"]

dur_min_test= config_rir["Source signal properties"]["dur_min_test"]
dur_min_train= config_rir["Source signal properties"]["dur_min_train"]
dur_max= config_rir["Source signal properties"]["dur_max"]
max_order= config_rir["Source signal properties"]["max_order"]


def next_pow_2(x):
    return 2^x



def noise_from_signal(x):

    x = np.asarray(x)
    n_x = x.shape[-1]
    n_fft = next_pow_2(n_x)
    X = np.fft.rfft(x, next_pow_2(n_fft))
    # Randomize phase.
    noise_mag = np.abs(X) * np.exp(2 * np.pi * 1j * np.random.random(X.shape[-1]))
    noise = np.real(np.fft.irfft(noise_mag, n_fft))
    out = noise[:n_x]
    
    return out


def stack_talkers(tlk_list, dur_min, speaker, nb_tlk=5):
    """Stacks talkers from tlk_list until dur_min is reached and number of
                    
    speakers exceeds nb_tlk
    Arguments:
        - tlk_list       list of flac/wav files to pick talkers in
        - speaker    ID of a speaker that should not be picked (e;G. if it is a target speaker whose spectral properties should not be seen)
        - dur_min       Minimal duration *in seconds* of the signal
    """
    i_tlk = 0
    tlk_tot = np.array([])
    str_files = str()
    fs = 16000
    while len(tlk_tot) < int(dur_min * fs) or i_tlk < nb_tlk:  # At least 5 talkers' speech shape
        rnd_tmp = np.random.randint(0, len(tlk_list))  # random talker we are going to pick
        spk_tmp = re.split('/', tlk_list[rnd_tmp])[-1].split('-')[0]
        
        if spk_tmp != speaker:  # Don't take same speaker for SSN
            #print(i_tlk)
            tlk_tmp, fs = sf.read(tlk_list[rnd_tmp])
            tlk_tot = np.hstack((tlk_tot, tlk_tmp))
            i_tlk += 1
            str_files = str_files + os.path.basename(tlk_list[rnd_tmp])[:-5] + '\n'
    return tlk_tot, fs, str_files

def get_noise(in_fpath, duration):
    #in_fpath = '/home/ajinkyak/Music/speech_shaped_noise/input_filelist.txt'
    tlk_list = []
    for fpath in open(in_fpath, 'r').readlines():
        tlk_list.append(fpath.strip())
    dur_min = 20
    speaker_id = 1
    tlk_tot, fs, str_files = stack_talkers(tlk_list, dur_min, speaker_id)
    n = noise_from_signal(tlk_tot)
    #print(n.shape, duration, fs)
    n = n[:int(duration*fs)]
    return n, fs

import glob, os

rtest_files_file = open('./filelists/robovox_testset.txt', 'r')
list_robovox_test = rtest_files_file.readlines()
np.random.seed(12345)
np.random.shuffle(list_robovox_test)

rtrain_files_file = open('./filelists/robovox_trainset.txt', 'r')
list_robovox_train = rtrain_files_file.readlines()
np.random.seed(12345)
np.random.shuffle(list_robovox_train)

ftrain_files_file = open('./filelists/freesound_noiseset.txt', 'r')
list_freesound = ftrain_files_file.readlines()
np.random.seed(12345)
np.random.shuffle(list_freesound)

ctrain_files_file = open('./filelists/chime_noiseset.txt', 'r')
list_chime = ctrain_files_file.readlines()
np.random.seed(12345)
np.random.shuffle(list_chime)


ssn_list = './filelists/ssn_stack_list.txt'

# selecting noise segments as per noise type, robovox, freesound, chime, ssn
def get_noise_segment_(noise_type, case, duration):
    
    if noise_type == 'robovox' and case == 'train':
        n, fs, n_file, n_file_start = read_random_part(list_robovox_train, duration)
    elif noise_type == 'robovox' and case == 'test':
        n, fs, n_file, n_file_start = read_random_part(list_robovox_test, duration)
    elif noise_type == 'freesound':
        n, fs, n_file, n_file_start = read_random_part(list_freesound, duration)
    elif noise_type == 'chime':
        n, fs, n_file, n_file_start = read_random_part(list_chime, duration)
    elif noise_type == 'ssn':
        n, fs = get_noise(ssn_list, duration)
        n_file = 'ssn'
    return n, n_file, fs

var_max = db2lin(-20)

def rir_id_exists(path_out_data, rir_id):
    path = os.path.join(path_out_data+'/dry_noise/', str(rir_id) + '.wav')
    #print(path)
    return os.path.exists(path)


def pad_to_maxlen(x, y):
    diff = len(x) - len(y)
    if diff > 0 :
        y = np.concatenate([y, np.zeros(diff)])
    elif diff < 0 :
        x = np.concatenate([x, np.zeros(abs(diff))])

    return x, y

def get_room_configuration():
    # Geometric properties
    length = len_min + (len_max - len_min)*np.random.rand()
    width = wid_min + (wid_max - wid_min)*np.random.rand()
    height = hei_min + (hei_max - hei_min)*np.random.rand()
    vol = length * width * height
    sur = 2 * (length * width) + 2 * (length * height) + 2 * (width * height)

    # Acoustic properties
    beta = beta_min + (beta_max - beta_min) * np.random.rand()
    alpha = 1 - np.exp((0.017 * beta - 0.1611)*vol/(beta * sur))

    return length, width, height, alpha, beta

def get_room_configuration_beta(beta):
    # Geometric properties
    length = len_min + (len_max - len_min)*np.random.rand()
    width = wid_min + (wid_max - wid_min)*np.random.rand()
    height = hei_min + (hei_max - hei_min)*np.random.rand()
    vol = length * width * height
    sur = 2 * (length * width) + 2 * (length * height) + 2 * (width * height)

    # Acoustic properties
    #beta = beta_min + (beta_max - beta_min) * np.random.rand()
    alpha = 1 - np.exp((0.017 * beta - 0.1611)*vol/(beta * sur))

    return length, width, height, alpha, beta

def get_room_configuration_multiple_beta(beta1, beta2, beta3):
    # Geometric properties
    length = len_min + (len_max - len_min)*np.random.rand()
    width = wid_min + (wid_max - wid_min)*np.random.rand()
    height = hei_min + (hei_max - hei_min)*np.random.rand()
    vol = length * width * height
    sur = 2 * (length * width) + 2 * (length * height) + 2 * (width * height)

    # Acoustic properties
    #beta = beta_min + (beta_max - beta_min) * np.random.rand()
    alpha1 = 1 - np.exp((0.017 * beta1 - 0.1611)*vol/(beta1 * sur))
    alpha2 = 1 - np.exp((0.017 * beta2 - 0.1611)*vol/(beta2 * sur))
    alpha3 = 1 - np.exp((0.017 * beta3 - 0.1611)*vol/(beta3 * sur))

    return length, width, height, alpha1, alpha2, alpha3


def get_array_positions(length, width):
    """

    :param length:
    :param width:
    :param phi:
    :return:        n1 - first node centre position
                    n2 - Second node centre position
                    o  - Array centre position
    """
    phi = np.pi*np.random.rand()                # random orientation in the room (in rad). Symmetric so up to pi only

    # Position of the microphone in the referential of the node
    x_mics_local = [d_mic * np.cos(phi),            # x-positions
                    d_mic * np.sin(phi),            # First mic is mic
                    -d_mic * np.cos(phi),           # in extension of array;
                    -d_mic * np.sin(phi)]
    x_mic_max = np.max(x_mics_local)

    y_mics_local = [d_mic * np.sin(phi),            # y-positions
                    -d_mic * np.cos(phi),
                    -d_mic * np.sin(phi),
                    d_mic * np.cos(phi)]
    y_mic_max = np.max(y_mics_local)

    # Position of the array *centre*
    o_x_min = d_wal + d/2*abs(np.cos(phi)) + x_mic_max          # Smallest allowed x-position (distance to x=0 axis)
    o_x_max = length - o_x_min
    o_x = o_x_min + (o_x_max - o_x_min) * np.random.rand()      # Random x-position in allowed area

    o_y_min = d_wal + d/2 * np.sin(phi) + y_mic_max
    o_y_max = width - o_y_min
    o_y = o_y_min + (o_y_max - o_y_min) * np.random.rand()      # Random x-position in allowed area

    # Nodes centre positions
    n1_x = o_x + d/2 * np.cos(phi)
    n1_y = o_y + d/2 * np.sin(phi)
    n2_x = o_x - d/2 * np.cos(phi)
    n2_y = o_y - d/2 * np.sin(phi)

    return [[n1_x, n1_y, z_cst], [n2_x, n2_y, z_cst]], [o_x, o_y, z_cst], phi


def get_noise_type_list(path_robovox, case):
    noise_type_list = {}
   
    if case == 'train':
        for line in open(os.path.join(path_robovox, 'training.csv'), 'r'):
            noise_type_list[line.strip().split(',')[3]] = line.strip().split(',')[1]
    else:
        for line in open(os.path.join(path_robovox, 'test.csv'), 'r'):
            noise_type_list[line.strip().split(',')[3]] = line.strip().split(',')[1]

    return noise_type_list


def get_array_positions_phi(length, width, phi):
    """

    :param length:
    :param width:
    :param phi:
    :return:        n1 - first node centre position
                    n2 - Second node centre position
                    o  - Array centre position
    """
    #phi = np.pi*np.random.rand()                # random orientation in the room (in rad). Symmetric so up to pi only

    # Position of the microphone in the referential of the node
    x_mics_local = [d_mic * np.cos(phi),            # x-positions
                    d_mic * np.sin(phi),            # First mic is mic
                    -d_mic * np.cos(phi),           # in extension of array;
                    -d_mic * np.sin(phi)]
    x_mic_max = np.max(x_mics_local)

    y_mics_local = [d_mic * np.sin(phi),            # y-positions
                    -d_mic * np.cos(phi),
                    -d_mic * np.sin(phi),
                    d_mic * np.cos(phi)]
    y_mic_max = np.max(y_mics_local)

    # Position of the array *centre*
    o_x_min = d_wal + d/2*abs(np.cos(phi)) + x_mic_max          # Smallest allowed x-position (distance to x=0 axis)
    o_x_max = length - o_x_min
    o_x = o_x_min + (o_x_max - o_x_min) * np.random.rand()      # Random x-position in allowed area

    o_y_min = d_wal + d/2 * np.sin(phi) + y_mic_max
    o_y_max = width - o_y_min
    o_y = o_y_min + (o_y_max - o_y_min) * np.random.rand()      # Random x-position in allowed area

    # Nodes centre positions
    n1_x = o_x + d/2 * np.cos(phi)
    n1_y = o_y + d/2 * np.sin(phi)
    n2_x = o_x - d/2 * np.cos(phi)
    n2_y = o_y - d/2 * np.sin(phi)

    return [[n1_x, n1_y, z_cst], [n2_x, n2_y, z_cst]], [o_x, o_y, z_cst], phi


def get_random_mics_positions(length, width):
    """
    Return the (x, y, z) coordinates of two microphones randomly placed in the room, at least distant of d_wal from the
    walls and d_rnd_mic from each other
    :param length:
    :param width:
    :return:
    """
    m1_x = d_wal + (length - 2 * d_wal) * np.random.rand()
    m1_y = d_wal + (width - 2 * d_wal) * np.random.rand()

    m2_x = d_wal + (length - 2 * d_wal) * np.random.rand()
    m2_y = d_wal + (width - 2 * d_wal) * np.random.rand()

    while np.sqrt((m1_x - m2_x)**2 + (m1_y - m2_y)**2) < d_rnd_mics:
        m2_x = d_wal + (length - 2 * d_wal) * np.random.rand()
        m2_y = d_wal + (width - 2 * d_wal) * np.random.rand()

    return [m1_x, m1_y, z_cst], [m2_x, m2_y, z_cst]

def get_theta(mic_xyz, source_position):
    
    dy = mic_xyz[1] - source_position[1]
    dx = mic_xyz[0] - source_position[0]

    phi = np.rad2deg(np.arctan(dy/dx))

    return phi

def get_source_positions_sfn(length, width, nodes_center, d_to_nodes=d_sou):
    """

    :param length:
    :param width:
    :param nodes_center:        Nodes central position (avoid source too close to nodes)
    :param d_to_nodes:          Distance to nodes (avoid source too close to nodes)
    :return:
        - Sources positions (x, y, z)
        - a counter: if equal to 100, no configuration was found and new input arguments should be given
    """
    ss = [[], []]
    ss_angle = 180
    
    min_angle_tolerence, max_angle_tolerence = -1.0,1.0
    cnt_alpha = 0
    for i in range(2):
        if cnt_alpha < 1000:     # Check (for i=1) that previous source is OK
            cnt_alpha = 0       # Reset to 0 after first source is found
            p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
            p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
            
            if i == 1:
                while (np.sqrt((nodes_center[0][0] - p_x) ** 2 + (nodes_center[0][1] - p_y) ** 2) < d_to_nodes
                      or np.sqrt((nodes_center[1][0] - p_x) ** 2 + (nodes_center[1][1] - p_y) ** 2) < d_to_nodes) \
                      and cnt_alpha < 1000:
                    p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
                    p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
                    cnt_alpha += 1
                ss[i] = [p_x, p_y, z_cst]
            elif i == 0:
                angle = get_theta([nodes_center[0][0], nodes_center[0][1]], [p_x, p_y])
                while (np.sqrt((nodes_center[0][0] - p_x) ** 2 + (nodes_center[0][1] - p_y) ** 2) < d_to_nodes
                      or np.sqrt((nodes_center[1][0] - p_x) ** 2 + (nodes_center[1][1] - p_y) ** 2) < d_to_nodes) \
                      and cnt_alpha < 1000 or (min_angle_tolerence > angle or angle > max_angle_tolerence):
                    p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
                    p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
                    angle = get_theta([nodes_center[0][0], nodes_center[0][1]], [p_x, p_y])
                    
                    cnt_alpha += 1
                ss[i] = [p_x, p_y, z_cst]
                ss_angle = angle
                #print(ss_angle, 'facing angle')
        else:
            return ss, cnt_alpha, ss_angle
        
        
        
    return ss, cnt_alpha, angle



def get_source_positions(length, width, nodes_center, d_to_nodes=d_sou):
    """

    :param length:
    :param width:
    :param nodes_center:        Nodes central position (avoid source too close to nodes)
    :param d_to_nodes:          Distance to nodes (avoid source too close to nodes)
    :return:
        - Sources positions (x, y, z)
        - a counter: if equal to 100, no configuration was found and new input arguments should be given
    """
    ss = [[], []]

    cnt_alpha = 0
    for i in range(2):
        if cnt_alpha < 100:     # Check (for i=1) that previous source is OK
            cnt_alpha = 0       # Reset to 0 after first source is found
            p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
            p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
            while (np.sqrt((nodes_center[0][0] - p_x) ** 2 + (nodes_center[0][1] - p_y) ** 2) < d_to_nodes
                  or np.sqrt((nodes_center[1][0] - p_x) ** 2 + (nodes_center[1][1] - p_y) ** 2) < d_to_nodes) \
                  and cnt_alpha < 100:
                p_x = d_sou_wal + (length - 2 * d_sou_wal) * np.random.random()
                p_y = d_sou_wal + (width - 2 * d_sou_wal) * np.random.random()
                cnt_alpha += 1
            ss[i] = [p_x, p_y, z_cst]
        else:
            return ss, cnt_alpha
    return ss, cnt_alpha


def get_target_segments(target_file, min_duration, max_duration):
    """
    Return source signals (one noise, one target)
    :param target_file:     name of a .wav/.flac file
    :return:                If target_file is long enough, the reshaped signal; if too short, None
    """
    signal, fs = sf.read(target_file)
    signal = signal[:, np.newaxis]
    sig_duration = len(signal) / fs
    #print(sig_duration)

    if sig_duration < min_duration:
        ssignal = -1
        vsignal = -1
    else:
        # If signal too long, reshape it into several segments
        if sig_duration > max_duration:
            signal = np.reshape(signal[:floor_to_multiple(sig_duration * fs, max_duration * fs)],
                                (max_duration * fs, np.int(np.floor(sig_duration / max_duration))),
                                order='F')
        # Add one second of silence at the beginning
        nb_seg = signal.shape[1]
        ssignal = np.zeros((signal.shape[0] + fs, nb_seg))
        vsignal = np.zeros(ssignal.shape)
        for i_seg in np.arange(nb_seg):
            # VAD
            vad_signal = vad_oracle_batch(signal[:, i_seg], thr=0.001)
            # Normalize the segment
            signal[:, i_seg] *= np.sqrt(var_max / np.var(signal[vad_signal == 1, i_seg]))
            ssignal[:, i_seg] = np.concatenate((np.zeros(fs), signal[:, i_seg]))
            vsignal[:, i_seg] = np.concatenate((np.zeros(fs), vad_signal))

    return ssignal, vsignal, fs


def get_noise_segment(noise_type, robovox_list, duration):
    #print(n_type)#TODO
    #print('get_noise_segment', len(robovox_list))

    n, fs, n_file, n_file_start = read_random_part(robovox_list, duration)
    noise_vad = None
   
    return n, n_file, n_file_start, noise_vad, fs


def pad_to_length(signal, length):
    """
    Pad with 0 a signal (1-D) to desired length
    :param signal:
    :param length:
    :return:
    """
    samples_to_pad = np.max((int(length - len(signal)), 0))
    padded_signal = np.pad(signal, (0, samples_to_pad), 'constant', constant_values=0)
    return padded_signal
