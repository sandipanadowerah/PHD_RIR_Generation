#!/bin/bash

#<ADD path to conda envoirnment>

export PATH=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/sdowerah/anaconda/bin:$PATH

#<activate your conda envoirnment>
source activate robovox

python rir_generate.py config_p_10.json $1
python rir_generate.py config_p_15.json $1
python rir_generate.py config_p_20.json $1
