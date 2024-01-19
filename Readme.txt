To cite, refer to "https://arxiv.org/abs/2210.08834"
"How to Leverage DNN-based speech enhancement for multi-channel speaker verification?" 


Step1. Install the necessary packages to run the RIR generation script, this script will do the room impulse response simulation and generate rir information and reverberated speech signals. 

Installation packages:

1. pip install librosa
2. pip install pyroomacoustics
3. pip install soundfile
4. pip install numpy

Step 2. First refer to config_readme.txt file and then change the config_rir.json

Step 3. Open config.json, edit the following global parameters,

{
        "noise_type": "robovox", # noise type: robovox, chime3, ssh
	"input_file":"/home/sdowerah/robovox/code/LIA_rir_generation_v2/filelists/Faboile_trainset.txt", # filepath to clean speech corpus
        "output_path": "./robovox_train", # output directory path
        "case": "train", # train or test
        "rt": "none", # if you choose "none" then rt will be selected randomly else you can specify values 0.2, 0.4, 0.6 .. for RT60 200msec, 400msec, 600msec
        "snr": "none",  # if you choose "none" then snr will be selected randomly else you can specify values 
	"rir_start":1 # starting rir id
} 


Step 4. You can also access the RIR  log files in RIR directory using following code snippet

import numpy as np
rir = np.load(rir_filepath, allow_pickle=True).all()

The structure of rir dict is as given below,

{'RIR_id': rir_id,
                     'speech_filename': os.path.basename(target_file),
                     'noise_filename': noise_file,
                     'room_len': room_length, 'room_wid': room_width, 'room_hei': room_height,
                     'room_absorb': room_alpha,
                     'beta': beta,
                     'phi': phi,
                     'mics_xyz': mics, 'sous_xyz': sous, 'snr_dry':snr_rnd, 'fw_snrs': tmp_snrs,
                     'rirs': rirs
                    }
