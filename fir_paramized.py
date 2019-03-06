# fir_paramized.py
#
# test a basic fir filter with inputs and coefficients
# scaled to various precisions

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

plt.style.use('ggplot')

N = 16384
fs = 48e3                 #48 kHz sample rate
t = np.arange(N) * (1/fs) #time steps
fsig = 1e3                #1 kHz test tone

acc_type_dict = { 64 : np.int64, 32 : np.int32, 16 : np.int16 }
accumulator_bits = 64 # total accumulator bits
accum_type = acc_type_dict[accumulator_bits]
sig_precision = 15 # fractional signal bits (total bits - 1)
coeff_precision = 23 # fractional coefficient bits
assert sig_precision + coeff_precision <= accumulator_bits - 1, "too many damn bits"
output_precision = 23 # fractional output bits
assert output_precision <= accumulator_bits - 1

#input sine
sig = np.sin(2*np.pi*fsig*t)

sig = sig * 2**sig_precision
# The python integer type used for the signal, coefficients, and output
# is the same as the accumulator, but they are all scaled
# according to their precision, so it accurately simulates a smaller integer size.
sig = sig.astype(accum_type)

# view quantized signal spectrum
sigq = sig.astype(np.float64) / 2**sig_precision
f, pxx_den = scipy.signal.periodogram(sigq, fs, 'flattop', scaling='spectrum')
Pxx = 10*np.log10(pxx_den)
plt.plot(f, Pxx, alpha=0.5)

# generate lowpass filter
taps = 91
cutoff = 10000 #Hz
filt = scipy.signal.remez(taps, [0, cutoff, cutoff + (fs/15), fs/2], [1, 0], fs=fs)
#filt = scipy.signal.firwin(taps, cutoff, fs=fs) # compare to window design method
filt = filt * 2**coeff_precision
filt = filt.astype(accum_type)

# view filter spectrum with quantized coefficients
filtq = filt.astype(np.float64) / 2**coeff_precision
filt_spec = np.fft.rfft(filtq, N)
filt_mag = np.absolute(filt_spec)
filt_db = 10*np.log10(filt_mag**2)
plt.plot(f, filt_db, alpha=0.5)

# filter the signal
out = np.convolve(sig, filt, 'valid') # valid elminates the time domain transition regions

# saturate before normalizing
out = np.minimum(out, 2**(sig_precision + coeff_precision) - 1)
out = np.maximum(out, -2**(sig_precision + coeff_precision))

# normalize (this could be done in one step, but I go to 1.XX first)
norm = accumulator_bits - (sig_precision + coeff_precision) - 1
out = out * 2**norm # normalize precision to 1.XX
out = out / 2**(accumulator_bits - 1 - output_precision) # scale to output precision
out = out.astype(accum_type) # previous step results in conversion to float,
                             # need to force back to int to get the true quantized result (truncated)
out = out.astype(np.float64) # convert quantized output to floats
scale = output_precision
out = out / 2**scale # scale floats to +/- 1.0

f, pxx_den = scipy.signal.periodogram(out, fs, 'flattop', scaling='spectrum')
Pxx = 10*np.log10(pxx_den)

plt.plot(f, Pxx)

plt.show()
