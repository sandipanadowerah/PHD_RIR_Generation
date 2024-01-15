#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(10)
import pyroomacoustics as pra
import soundfile as sf
import sys
from code_utils.math_utils import cart2pol, pol2cart, floor_to_multiple, db2lin
from code_utils.sigproc_utils import vad_oracle_batch
from code_utils.metrics import snr, seg_snr, reverb_ratios
from code_utils.db_utils import *
import math
import glob
import time
import random
import pickle
from utils import *
import random
import librosa
import sys
import json


# In[2]:


class RIR_Generate():
    

    def __init__(self, param, task_id):
        # init
        
        self.noise_type = param["noise_type"]
        self.output_path = param["output_path"]
        self.case = param["case"]
        self.rt = param["rt"]
        self.snr = param["snr"]
        self.rir_start = param["rir_start"]
        self.rir_end = param["rir_end"]
        self.input_file = param["input_file"]
        self.step_size = param["step_size"]
        self.task_id = int(task_id)
        self.i = 0
                               
        dur_min, target_files_list = self.initilize_parameters()
        
        self.dur_min = dur_min
        self.target_files_list = target_files_list
        
    def get_filelist(self, d, rir_diff):
        random.shuffle(d)
        data = []
        while self.rir_end>1:
            for i in d:
                data.append(i)
                #print(len(data),len(data) == rir_diff)
                if len(data) == rir_diff:
                    return data
        
    def initilize_parameters(self, ):
        
        target_files_file = open(self.input_file, 'r')
        target_files_list = target_files_file.readlines()
        target_files_list = self.get_filelist(target_files_list, self.rir_end - self.rir_start)
        target_files_list = [target_files_list[i:i+self.step_size] for i in range(0,len(target_files_list), self.step_size)][self.task_id]
        
        if self.case == 'train':
            dur_min = dur_min_train
        else:
            dur_min = dur_min_test

        os.makedirs(self.output_path, exist_ok=True) # create directories for saving speech files
        #os.makedirs(os.path.join(self.output_path , 'dry_noise/'), exist_ok=True)
        #os.makedirs(os.path.join(self.output_path , 'dry_target/'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path , 'Noisy/'), exist_ok=True)
        #os.makedirs(os.path.join(self.output_path , 'Early_reverb_target/'), exist_ok=True)
        #os.makedirs(os.path.join(self.output_path , 'Noise/'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path , 'Target/'), exist_ok=True)
        #os.makedirs(os.path.join(self.output_path , 'RIR/'), exist_ok=True)

        return dur_min, target_files_list
    
    
    def generate_rir(self, rir_id, target_file):
    
        beta, snr = self.get_rt_snr()
        
        target_segment, target_vad, dur_sig, fs = self.get_target(target_file)
        
        noise_segment, noise_file = self.get_noise(dur_sig, fs, target_segment, snr, target_vad)
        
        clean_reverbed_signals, noisy_reverbed_signals, rirs, infos = self.get_room_impulse_response(rir_id, beta, fs, target_segment, noise_segment, target_file, noise_file, snr, dur_sig, target_vad)
        
        outfname = target_file.split('/')[-3]+'-'+target_file.split('/')[-2]+'-'+target_file.split('/')[-1]
        self.write_rir_generated_signals(outfname, target_segment, noise_segment, fs, clean_reverbed_signals, noisy_reverbed_signals, rirs, infos)
        
        print('rir id', rir_id, 'generated for ', os.path.basename(target_file), 'clean speech')   
    
        return 1
    
    def run_rir_generation(self, ):

        rir_id = self.rir_start + (self.task_id*self.step_size)
        for _ in range(len(self.target_files_list)):
            return_id = -1
            trails = 0
            while return_id == -1 :
                target_file = self.target_files_list[self.i].strip()
                
                print()
		
                outfname = target_file.split('/')[-3]+'-'+target_file.split('/')[-2]+'-'+target_file.split('/')[-1]
		
                if os.path.exists(os.path.join(self.output_path , 'Noisy/', os.path.basename(target_file)+'.wav')) == False:
                # generating rir configs with wav files saved in path_out_data

                    return_id = self.generate_rir(rir_id, target_file)

                    trails += 1
                else:
                    print('exists already, skipping ', rir_id, 'processing .....')
                    return_id = 1

                if return_id == 2:
                    return_id = self.generate_rir(rir_id, target_file)


            rir_id = rir_id + 1 #print(rir_id, 'processed .....')
            self.i = self.i + 1
        
    def get_rt_snr(self,):
        # if rt is none rt will be selected randomly
        if self.rt == 'none':
            beta = random.choice(beta_range)
        else:
            beta = self.rt # else we will take input rt value

        # if snr is none rt will be selected randomly      
        if self.snr == 'none':
            snr_rnd = snr_min + (snr_max - snr_min) * np.random.rand()
        else:
            snr_rnd = self.snr # else we will take input rt value

        return beta, snr_rnd
    
    def get_target(self, target_file):
    
        if os.path.exists(target_file) == False:
            print('issue with target filepath', target_file)

        target_signal, target_vads, fs = get_target_segments(target_file, int(self.dur_min), int(dur_max)) # extracting target segment

        if type(target_signal) == int:
            print('target signal length is less than minimum duration')# *** check all the return -1
            self.i = self.i + 1
            target_file = self.target_files_list[self.i].strip()
            
            return self.get_target(target_file)

        target_segment =  np.ascontiguousarray(target_signal[:, 0]) 
        target_vad = np.ascontiguousarray(target_vads[:, 0])
        dur_sig = len(target_segment)/fs
        len_to_pad = int(dur_max * fs)

        return target_segment, target_vad, dur_sig, fs
    
    def get_noise(self, dur_sig, fs, target_segment, snr_rnd, target_vad):
        # randomly selecting the noise segment get_noise_segment(noise_type, case, duration)
        noise_segment, noise_file, fs_noise = get_noise_segment_(self.noise_type,self.case,dur_sig) 


        # if noise segment is multichannel
        if len(noise_segment.shape) > 1:
            noise_segment = noise_segment[:,0]

        if fs_noise != fs:
            noise_segment = librosa.resample(noise_segment, fs_noise, fs)


        noise_segment = noise_segment[:target_segment.shape[0],]     # increasing snr in noise segment
        noise_segment = increase_to_snr(target_segment, noise_segment, snr_rnd,
                                                    weight=False, vad_tar=target_vad, vad_noi=None, fs=fs)

        return noise_segment, noise_file


    
    def write_rir_generated_signals(self, rir_id, target_segment, noise_segment, fs, clean_reverbed_signals, noisy_reverbed_signals, rirs, infos):
        # saving dry signals 

        #sf.write(os.path.join(self.output_path,'dry_target/', str(rir_id) + '.wav'), target_segment, fs)
        #sf.write(os.path.join(self.output_path,'dry_noise/' , str(rir_id) + '.wav'), noise_segment, fs)


        for i_mic in range(n_mics):

            # Mixed signal
            target_convolved = clean_reverbed_signals[0][i_mic]
            noise_convolved = clean_reverbed_signals[1][i_mic]

            noisy_convolved = noisy_reverbed_signals[i_mic]

            target_convolved = np.convolve(target_segment, rirs[i_mic][0])

            first_peak = rirs[i_mic][0].argmax() 

            early_reverb_target_convolved = np.convolve(target_segment, rirs[i_mic][0][:first_peak + early_reverb_len])

            target_convolved, noise_convolved = pad_to_maxlen(target_convolved, noise_convolved)

            # saving reverberated signals
            sf.write(os.path.join(self.output_path,'Noisy/', str(rir_id).replace('.wav', '') + '.wav'),
                     noisy_convolved, fs)
            sf.write(os.path.join(self.output_path , 'Target/' , str(rir_id).replace('.wav', '') + '.wav'),
                    target_convolved, fs)
    
    
    def get_room_impulse_response(self, rir_id, beta, fs, target_segment, noise_segment, target_file, noise_file, snr_rnd, dur_sig, target_vad):
        # %% drawing room configurations with 100 trails
        delta_snr_is_enough = False
        n_trials = 0
        while not delta_snr_is_enough:
            #print(n_trials)#print('drawing new configuration because of SNR')
            n_trials += 1
            draw_new_config = 1
            if n_trials == 4:

                noise_segment, noise_file = self.get_noise(dur_sig, fs, target_segment, snr_rnd, target_vad)

                return self.get_room_impulse_response(rir_id, beta, fs, target_segment, noise_segment, target_file, noise_file, snr_rnd, dur_sig, target_vad)
              

            while draw_new_config:
                #print("xtracting room parameter including rt60 parameter room_alpha")
                room_length, room_width, room_height, room_alpha, room_beta = get_room_configuration_beta(beta)
                nodes_positions, array_centre_position, phi = get_array_positions(room_length, room_width)

                sources_positions, cnt_rnd_angle = get_source_positions(room_length, room_width, nodes_positions)

                if cnt_rnd_angle < 100:
                    draw_new_config = 0

            #print('room config ...')
            room = pra.ShoeBox([room_length, room_width, room_height],
                                   fs=fs,
                                   max_order=max_order,
                                   absorption=room_alpha)

            # %% Add sources and microphones
            room.add_source(sources_positions[0], signal=target_segment)
            room.add_source(sources_positions[1], signal=noise_segment)  # Only one noise so far

            node1 = pra.circular_2D_array(center=[nodes_positions[0][0], nodes_positions[0][1]], M=n_mics,
                                              phi0=np.pi / 2 * np.random.rand(), radius=d_mic)

            node1 = np.vstack((node1, np.tile(0.0, node1.shape[1])))

            room.add_microphone_array(pra.MicrophoneArray(node1, room.fs))



            room.image_source_model()
            room.compute_rir()
            rirs = room.rir
            clean_reverbed_signals = room.simulate(return_premix=True)
            noisy_reverbed_signals = room.mic_array.signals

            tmp_snrs = [[] for k in range(n_mics)]

            for i_mic in range(n_mics):
                target_convolved = clean_reverbed_signals[0][i_mic]
                noise_convolved = clean_reverbed_signals[1][i_mic]
                # SNRs at mic level
                vad_signal = vad_oracle_batch(target_convolved, thr=0.001)
                _, snr_fw, _ = fw_snr(target_convolved, noise_convolved, fs, vad_tar=vad_signal, vad_noi=None)
                tmp_snrs[i_mic] = snr_fw

            # Retain config if delta_snr is high enough and each SNR is within bounds
		# controling snr at reciver between snr_min and snr_max
            delta_snr_is_enough =  snr_rnd-5.0 < np.mean(tmp_snrs) and np.mean(tmp_snrs) < snr_rnd+5.0

            print(np.mean(tmp_snrs), snr_rnd, delta_snr_is_enough, n_trials)


        # %% Save everything
        mics = room.mic_array.R
        sous = np.vstack((room.sources[0].position, room.sources[1].position))
        infos = {'RIR_id': rir_id,
                     'speech_filename': os.path.basename(target_file),
                     'noise_filename': noise_file,
                     'room_len': room_length, 'room_wid': room_width, 'room_hei': room_height,
                     'room_absorb': room_alpha,
                     'beta': beta,
                     'phi': phi,
                     'mics_xyz': mics, 'sous_xyz': sous, 'snr_dry':snr_rnd, 'fw_snrs': tmp_snrs,
                     'rirs': rirs
                    }

        return clean_reverbed_signals, noisy_reverbed_signals, rirs, infos


# In[ ]:


if __name__ == "__main__":
    config_filepath = sys.argv[1] # argument 1 config file
    print(config_filepath)
    task_id = sys.argv[2]
    if os.path.exists(config_filepath) == False:
        print('Please check config filepath', config_filepath)
    else:
        with open(config_filepath, 'r') as f:
            params = json.load(f)


    rir_generator = RIR_Generate(params, task_id)

    rir_generator.run_rir_generation() 


# In[ ]:




