#!/usr/bin/env python

import argparse
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wf
import sounddevice as sd
import bottleneck as bn
import paho.mqtt.publish as publish


plt.close('all')

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


try:
    columns, _ = shutil.get_terminal_size()
except AttributeError:
    columns = 80

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser.add_argument(
    '-b', '--block-duration', type=float, metavar='DURATION', default=50,
    help='block size (default %(default)s milliseconds)')
parser.add_argument(
    '-c', '--columns', type=int, default=columns,
    help='width of spectrogram')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-g', '--gain', type=float, default=10,
    help='initial gain factor (default %(default)s)')
parser.add_argument(
    '-r', '--range', type=float, nargs=2,
    metavar=('LOW', 'HIGH'), default=[100, 2000],
    help='frequency range (default %(default)s Hz)')
args = parser.parse_args(remaining)
low, high = args.range
if high <= low:
    parser.error('HIGH must be greater than LOW')


plt.close('all')


def dbfft(x, fs, win=None, ref=1):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
        win: vector containing window samples (same length as x).
             If not provided, then rectangular window is used by default.
        ref: reference value used for dBFS scale. 32768 for int16 and 1 for float

    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """

    N = len(x)  # Length of input sequence

    if win is None:
        win = np.ones(1, N)
    if len(x) != len(win):
            raise ValueError('Signal and window must be of the same length')
    x = x * win
    rms = np.sqrt(np.mean(x ** 2))

    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)

    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / np.sum(win)
    # Convert to dBFS
    s_dbfs = 20 * np.log10(s_mag/ref)


    return freq, s_dbfs

def main():
    fs = 43200
    duration = 60 # seconds
    #Set input device settings
    inputDict = sd.query_devices('USB')
    print(inputDict['name'])
    sd.default.device = inputDict['name']
    sd.default.samplerate = inputDict['default_samplerate']
    sd.default.channels = inputDict['max_input_channels']
    print(sd.default.device, sd.default.samplerate)
    myrecording = sd.rec(duration * fs, samplerate=sd.default.samplerate, channels=1, dtype='float64')

    print("Recording Audio")
    while sd.wait():
        counter = 0
        print("waiting for " + counter + "seconds")
        counter += 1
    wf.write('test.wav', fs, myrecording)
    print("Audio recording complete, Play Audio")

    fs, signal = wf.read('test.wav')

    # Take slice
    N = myrecording.size
    win = np.hanning(N)
    freq, s_dbfs = dbfft(signal[0:N], fs, win)

    # Scale from dBFS to dB
    K = 120
    sensitivity = 51
    print(len(s_dbfs))

    s_dbmean = bn.move_mean(s_dbfs, window=50, min_count=1)
    s_db = s_dbmean + sensitivity + K
    # get running mean

    maxdb = 70
    sampletimescale = 1 / (sd.default.samplerate/2.2)
    count = len([i for i in s_db if i > maxdb])
    print((count*sampletimescale))
    secondsAboveMaxDB = count*sampletimescale
    publish.single('audioStorage', payload='seconds_above_70db(spl) seconds=%d' % secondsAboveMaxDB, hostname='localhost')

    #plt.plot(freq, s_db)
#    plt.xlim((0, fs / 2))
    #plt.ylim((0, 125))
    #plt.grid(True)
    #plt.xlabel('Frequency [Hz]')
    #plt.ylabel('Amplitude [dB]')
    #plt.show()


if __name__ == "__main__":
    try:
        main()
        sd.wait()
        print("Processing Audio Complete")
    except KeyboardInterrupt:
        parser.exit('Interrupted by user')