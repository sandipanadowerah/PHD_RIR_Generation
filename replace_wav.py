import os
import sys

dirpath = sys.argv[1]

for fname in os.listdir(dirpath):
    fpath = os.path.join(dirpath, fname)
    ofpath = os.path.join(dirpath, fname.replace('.wav.wav', '.wav'))
    os.rename(fpath, ofpath)
    
