#!/usr/bin/env python3

from jinja2 import Template
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import wave

### Constants ###
SAMPLE_RATE = 44100 # Hz

### TEC model params class ###
class tec_pms:
    def __init__(self, l_tec, \
                       k_tec, \
                       a_tec, \
                       m_tec, \
                       c_tec, \
                       alpha, \
                       l_metal, \
                       k_metal, \
                       a_metal, \
                       m_metal, \
                       c_metal, \
                       rs):
        self.l_tec = l_tec
        self.k_tec = k_tec
        self.a_tec = a_tec
        self.m_tec = m_tec
        self.c_tec = c_tec
        self.alpha = alpha
        self.l_metal = l_metal
        self.k_metal = k_metal
        self.a_metal = a_metal
        self.m_metal = m_metal
        self.c_metal = c_metal
        self.rs = rs

### Generate wav file ###
# data -> a numpy array of data
def gen_wav(data):
    write("output/wave.wav", SAMPLE_RATE, data.astype(np.int16))

### Generate params file ###
def gen_pms(pms_obj):
    with open("params.mod.template", "r") as p:
        t = Template(p.read())
        with open("output/params.mod", "w") as m:
            m.write(t.render(l_tec   = pms_obj.l_tec, \
                             k_tec   = pms_obj.k_tec, \
                             a_tec   = pms_obj.a_tec, \
                             m_tec   = pms_obj.m_tec, \
                             c_tec   = pms_obj.c_tec, \
                             alpha   = pms_obj.alpha, \
                             l_metal = pms_obj.l_metal, \
                             k_metal = pms_obj.k_metal, \
                             a_metal = pms_obj.a_metal, \
                             m_metal = pms_obj.m_metal, \
                             c_metal = pms_obj.c_metal, \
                             rs      = pms_obj.rs))

### Call ngspice ###
def call_ngspice():
    os.system("./spice-audio-tools/wavtospice.py output/wave.wav output/inputvalues")
    os.system("ngspice -b circuit.cir")
    os.system("./spice-audio-tools/spicetowav.py output/output output/output.wav")

### Plot values ###
# TODO: Need real voltages
def plot_values():
    plt.plot(np.fromstring(wave.open("output.wav").readframes(-1), "Int16"))

### Clean directory ###
# TODO

if __name__ == "__main__":
    import math
    tcp = tec_pms(0.20, 0.50, 0.20, 0.10, 0.90, 0.10, 0.1, 0.2, 0.65, 0.1, 0.3, 40.00)
    t = np.linspace(0.00, 5.00, 1000.00)
    y = np.sin(2.00 * math.pi * 100.00 * t)
    gen_wav(y)
    gen_pms(tcp)
    call_ngspice()
    plot_values()
