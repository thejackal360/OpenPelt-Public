#!/usr/bin/env python3

import math
from scipy.io.wavfile import write
import numpy as np

sampleRate = 44100 # Hz
duration = 1.0 # s
frequency = 440.00 # Hz
t = np.linspace(0.00, 1.00, sampleRate)
amplitude = np.iinfo(np.int16).max
data = amplitude * np.sin(2. * np.pi * frequency * t)
write("output/wave.wav", sampleRate, data.astype(np.int16))
