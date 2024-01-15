{
   "miscellaneous properties": { 
        "snr_min": 0, # Minimum SNR
        "snr_max": 20, # Maximum SNR
        "beta_range": [0.2,0.3,0.4,0.5,0.6], # range of RT60 parameter in secs
        "early_reverb_len": 800, # early reverberation length in millisec
        "n_mics": 1 # number of mics
    },

    "Room geometry": { 
        "len_min": 3, # minimum length of room
        "len_max": 8, # maximum length of room
        "wid_min": 3, # minimum width of room 
        "wid_max": 5, # maximum width of room
        "hei_min": 2, # minimum heigth of room 
	"hei_max":3   # maximum heigth of room 
    },

    "Sensor positions": {
        "d": 1.5,  # microphone array length
        "d_wal":1, # minimum distance to the walls
        "d_mic": 0.1, # distance between mics and node centre
        "z_cst": 1.5, # height of microphone array
        "d_rnd_mics": 1 # random distance between two mics
    },

    "Source properties": {
        "d_sou": 1.5, # Minimum distance of sources to microphones
        "d_sou_wal": 0.25 # Minimum distance of source to wall
    },

    "Source signal properties": {
        "dur_min_test": 1, # minimum speech segment duration for test set
        "dur_min_train": 5, # minimum speech segment duration for train set
        "dur_max": 10, # maximum speech segment duration
        "max_order": 20 # maximum order
    }

}



