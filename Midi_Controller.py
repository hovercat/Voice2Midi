''' MIDI Controller converts sound to Midi-Signals
It should act as a MIDI device.

It is a class that should be created using the device identifier.
Initializaion:
    - Threshold is set depending on background noise

Special Init: (optional)
    - All input devices can be requested and checked for sound.
    - The one that is different should be used. (so dynamic sound)


After calling start():
    - Voice is converted to Midi, if input is over a threshold and tone is clear.

Midi is optional though.
There should be a method (or the object itself as iterable) that returns MIDI-Frequencies 
or otherwise a clear indication for no sound.

'''

from functools import cached_property
#import librosa
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import scale
from mido import Message
import mido
import numpy as np
from numpy import fft
import pandas as pd
import time
import scipy.signal as sig
import sounddevice as sd

logging.basicConfig(level=logging.INFO)
PLOT = False
T_ANALYZE = 0.06 # s
T_ANALYZE_BUFFER_LENGTH = 100/1000 # s
T_THRESHOLD = 2 #  s
T_VOLUME_BUFFER_LENGTH = 0.10 # s
T_BUFFER_LENGTH = 3 #  s
T_LOG_AUDIO_ONLINE = 5 # s
T_PLOT_RATE = 1/2 # 12 fps

SAMPLING_RATE = 48000 # Hz
NYQUIST_RATE = SAMPLING_RATE / 2
THRESHOLD_SCALING = 4

MIDI = None # böse

if PLOT:
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.ion()
    plt.show()

def _get_from_list(lst, i, default=0):
    if 0 <= i < len(lst):
        return lst[i]
    else:
        return default

class Signal(np.ndarray):
    def __new__(self, data = None):
        self = np.zeros((0)).view(Signal) if data is None else data
        return self

    @cached_property
    def loudness(self):
        return np.average(np.abs(self))

#    @property
#    def loudness_db(self):
#        pass

    @cached_property
    def fft_spectrum(self):
        return np.fft.rfft(self)


    @cached_property
    def fft_magnitude(self):
        return np.abs(self.fft_spectrum)
    

    @cached_property
    def freqs(self):
        #return self.fft_spectrum_abs
        return np.fft.rfftfreq(len(self), d=1/SAMPLING_RATE)


    @cached_property
    def spectral_centroid(self):
        #  TODO caching
        return np.sum(self.fft_magnitude * self.freqs) / np.sum(self.fft_magnitude)


    def highpass(self, hertz: int):
        b, a = sig.butter(2, hertz / (SAMPLING_RATE / 2) , 'highpass')
        filtered_self = sig.filtfilt(b, a, self, method='gust')
        return filtered_self.view(Signal)
        

    def lowpass(self, hertz: int):
        b, a = sig.butter(2, hertz / (SAMPLING_RATE / 2) , 'lowpass')
        filtered_self = sig.filtfilt(b, a, self, method='gust')
        return filtered_self.view(Signal)

   
    def welch(self):
        f, welch_spec = sig.welch(self, SAMPLING_RATE, scaling='spectrum', nperseg=2048)
        return f, welch_spec

    def remove_noise(self, noise):
        return (self - noise[0:len(self)]).view(Signal)           
        


    def get_harmonic_product_spectrum(self):
        fft_mag = self.fft_magnitude
        #fft_mag /= fft_mag.max()
        fft_mag[self.freqs < 50] = 0

        fft_2 = sig.resample_poly(fft_mag, down=2, up=1)
        fft_3 = sig.resample_poly(fft_mag, down=3, up=1)
        fft_4 = sig.resample_poly(fft_mag, down=4, up=1)
        

        harmonics = np.zeros(fft_3.shape)
        avg = np.average(fft_mag[0:1000])
        # TODO more pythonic pls
        for i in range(0, len(harmonics)):
            #for k in range(0, 3):
                # harmonics[i] *= _get_from_list(fft_mag, i*(2**(k+1)), default=1)
            #harmonics[i] = _get_from_list(fft_mag, i, 0) *  _get_from_list(fft_mag, 2*i, 0) * _get_from_list(fft_mag, 3*i, 0) #* _get_from_list(fft_mag, 4*i, 0)
            #harmonics[i] = fft_mag[i] * fft_2[i] * fft_3[i]
            if fft_mag[i] < 3*avg:
                continue

            harmonics[i] = fft_mag[i]
            #if fft_2[i] > fft_mag[i]:
            #    continue
            harmonics[i] *= fft_2[i]
            #if fft_3[i] > fft_2[i]:
            #    continue
            harmonics[i] *= fft_3[i]


            #plt.cla()
            #plt.plot(harmonics)

       # harmonics = harmonics / harmonics.max()
        #harmonics[self.freqs < 50] = 0
        harmonics[harmonics < 0] = 0
        return harmonics, self.freqs[:len(harmonics)]

        

    def filter(self, filter_type = 'gauss'):
        if filter_type == 'gauss':
            return self # TODO GAUSS
        else:
            return self

class Midi_Controller(object):
    midi_out = None
    input_device: int = None
    input_stream: sd = None

    _threshold: float = None
    _signal_buffer: Signal = None
    _s_buffer_raw: Signal = None
    _s_current_raw: Signal = None


    def __init__(self, input_device: int, midi_id):
        self.input_device = input_device
        self.midi_id = midi_id
        self._s_buffer_raw = Signal()
        self._s_current_raw = Signal()

    def note_off(self, note: int):
        if note is not None:
            self.midi_out.send(Message('note_off', note=note))
            self.midi_out.send(Message('pitchwheel', pitch=0))
    
    notes = []
    pitch = 2000
    def note_on(self, hertz: int, volume: int = 64, pitch = 0):
        note_midi = int(MIDI.iloc[(MIDI['hz1']-hertz).abs().argsort()[:1]]['i'])
        self.notes.append(note_midi)

        if len(set(self.notes)) != 1:
            self.notes = []
        
        if len(self.notes) < 2:     # todo ist grad auf 1        
            return False

        if self.current_note != note_midi:
            self.note_off(self.current_note)
            self.current_note = note_midi
            self.midi_out.send(Message('note_on', note=note_midi, velocity=volume))
            return True

    def pitch(self, hertz: int): 
        return
        # this must work better somehow
        nearest_note_ids = MIDI.iloc[(MIDI['hz1']-hertz).abs().argsort()[:2]]

        note_hz = nearest_note_ids.iloc[0]['hz1']
        next_hz = nearest_note_ids.iloc[1]['hz1']
        # hacky TODO

        hertz_diff = note_hz - hertz
        pitch = hertz_diff / (note_hz - next_hz)
        pitch /= 4

        self.midi_out.send(Message('pitchwheel', pitch=int(pitch * 8192)))
        


    current_note = None
    #  how to set period of analysis? maybe like every 10th of a second?
    def start(self): #  TODO  __start__ ???
        t_start = time.time()
        t_loop_n_analyze = t_start
        t_loop_n_timer = t_start
        n = 0
        n_clb_2 = self.n_clb
        
        with sd.InputStream(device=self.input_device, blocksize=480, channels=1, samplerate=SAMPLING_RATE, callback=self.audio_callback) as in_stream, mido.open_output(self.midi_id) as midi_out:
            self.midi_out = midi_out # hacky TODO
            while True:
                t_loop_n = time.time()

                if self.n_clb == n_clb_2:
                    continue

                n_clb_2 = self.n_clb

                # checking n
                if t_loop_n - t_loop_n_timer > 1:
                    logging.info("Full Executions in 1 second:\tn: {}\tn_clb: {}".format(n, self.n_clb))
                    logging.info("Latency: {}".format(in_stream.latency))
                    n = 0
                    t_loop_n_timer = time.time()

                # setting sound threshold in beginning of program
                if t_loop_n - t_start > T_THRESHOLD:
                    if self.threshold is None:
                        self.threshold_buffer = self.signal_buffer.view(Signal)
                        self.set_threshold(self.signal_buffer)
                        self.noise = self.threshold_buffer.lowpass(2000).highpass(30)
                else:
                    continue

                # dont analyze if signal is too weak
                s_volume_check = self.signal_buffer[-int(SAMPLING_RATE * T_VOLUME_BUFFER_LENGTH):]
                if s_volume_check.loudness < self.threshold:
                    if self.current_note is not None:
                        self.note_off(self.current_note)
                        self.current_note = None
                    continue
                
                s_analyze = self.signal_buffer[-int(SAMPLING_RATE * T_ANALYZE_BUFFER_LENGTH):]
                #s_analyze = self.signal_buffer.lowpass(4000).highpass(60)
                #s_analyze = s_analyze.remove_noise(self.noise)


                # fq_welch, fq_spec = s_analyze.welch()
                # fq_spec = fq_spec / fq_spec.max()
                # fq_w = fq_welch[np.argmax(fq_spec)]
                # logging.debug("WELCH FQ:\t {}Hz".format(fq_welch[np.argmax(fq_spec)]))
                
                # fft_magnitude = s_analyze.fft_magnitude
                # fft_magnitude = fft_magnitude / fft_magnitude.max()
                # fq_m = s_analyze.freqs[np.argmax(fft_magnitude)]
                # logging.debug("MAGN FQ:\t {}Hz".format(s_analyze.freqs[np.argmax(fft_magnitude)]))


                hps, hps_freqs = s_analyze.get_harmonic_product_spectrum()
                hps = hps / hps.max()
                fq_hps = hps_freqs[np.argmax(hps)]
                logging.debug("HPS FQ: {} Hz".format(fq_hps))

                
                #fq = fq_hps if min(fq_w, fq_m) < 500 else min(fq_m,fq_w)

                # fq = 0
                # if fq_spec.var() < min(fft_magnitude.var(), hps.var()):
                #     fq = fq_w
                # elif fft_magnitude.var() < min(fq_spec.var(), hps.var()):
                #     fq = fq_m
                # else:
                #     fq = fq_hps

                fq = fq_hps
                #fq = fq_hps
                sent = self.note_on(fq)#, int(127 * ((s_analyze.loudness - self.threshold)/(1-self.threshold))))
                self.pitch(fq)


                if PLOT and time.time() - self.t_plot_n1 > T_PLOT_RATE:
                    self.t_plot_n1 = time.time()
                    self.plot(fq_spec, fq_welch, "welch", hold=True, clear=True)
                    self.plot(fft_magnitude, s_analyze.freqs, "magn", hold=True, clear=False)
                    self.plot(hps, hps_freqs, "hps", hold=False, clear=False)


                if sent:
                    logging.info('Time passe since signal came in: {}ms'.format((time.time() - t_loop_n)*1000))



                #self.note_on(min(fq_w, fq_m, fq_hps))
                n += 1


    t_plot_n1 = time.time()    
    def plot(self, y, x, label = '', hold=False, clear=False):
        if clear:
            plt.cla()
            plt.ioff()
            plt.xlim(0, 2500)
           # plt.ylim(0, 1)

        plt.plot(x, y, label=label)

        if not hold:
            plt.ion()
            plt.draw()
            plt.pause(0.001)
            plt.legend()
            self.t_plot_n1 = time.time()
 
    @property
    def threshold(self):
        return self._threshold
        
    #@property.setter
    def set_threshold(self, signal: Signal, reset = False):
        self._threshold = signal.loudness * THRESHOLD_SCALING
        logging.info("_________________________")
        logging.info("Threshold set.")
        logging.info("Surrounding volume: {} units".format(signal.loudness))
        logging.info("Threshold volume: {} units".format(self.threshold))
        logging.info("_________________________")

    @property
    def signal_buffer(self):
        if self._signal_buffer is None or len(self._signal_buffer) == 0:
            self._signal_buffer = Signal()

        return self._signal_buffer

    @signal_buffer.setter
    def signal_buffer(self, signal):
        self._signal_buffer = signal


    ''' audio_callback
    called when audio comes in.
    may be subject to being overwritten in inherited classes
    '''
    _t_audio = _t_audio_n1 = time.time()
    n_clb = 0
    def audio_callback(self, signal, frames, time_passed, status):
        self._t_audio = time.time()
        if self._t_audio - self._t_audio_n1 > T_LOG_AUDIO_ONLINE:
            logging.info("Audio callback online.")
            self._t_audio_n1 = self._t_audio

        self._s_current_raw = signal.transpose()[0].view(Signal) 
        self._s_buffer_raw = np.concatenate((self._s_buffer_raw, self._s_current_raw))
        self._s_buffer_raw = self._s_buffer_raw[-T_BUFFER_LENGTH*SAMPLING_RATE:].view(Signal)
        
        self.signal_buffer = self._s_buffer_raw

        self.n_clb += 1
        #b, a = sig.butter(2, 0.1, btype='highpass')
        #self.signal_buffer = sig.filtfilt(b, a, self.signal_buffer).view(Signal)
        
        



if __name__ == "__main__":
    
    logging.info("Midi Controller MAIN")
    logging.info("Audio Interfaces: ")
    logging.info(sd.query_devices())
    logging.info("MIDI Interfaces: ")
    logging.info(mido.get_output_names())

    # haaaacky
    MIDI = pd.read_csv('midi.csv')    

    #mc = Midi_Controller(20, 'mcv 1')
    sd.default.latency = ['low', 'low']
    mc = Midi_Controller(29, 'LoopBe Internal MIDI 1')
    mc.start()

