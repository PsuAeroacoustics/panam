import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.io import savemat

from flight_acoustics import psd

font = {'family': 'serif',
        'weight': 'normal',
        'size': 10}

plt.style.use('ggplot')
matplotlib.rc('font', **font)
matplotlib.rcParams.update({'mathtext.fontset': 'dejavuserif'})
matplotlib.rcParams.update({'figure.autolayout': True})

xlim = (0, 5000)
ylim = (35, 100)

amb, fsamb = sf.read('calibration.wav')
lnom = 70

famb, psdamb, lamb = psd(amb[:, 0], fsamb)

cal = lnom - lamb

famb, psdamb, lamb = psd(amb[:, 0], fsamb, cal)
print('Calibrated level is {:0.1f} with {:0.1f} expected.'.format(lamb, lnom))
print('Calibration factor is {:0.1f} dB.'.format(cal))

files = glob.glob('wav_analysis/*.wav')
figl, axl = plt.subplots()

rs = [0.125, 0.25, 0.5, 1, 2, 4]
# Mach 0.20
data1, fs = sf.read(files[0])
data2, _ = sf.read(files[6])
data3, _ = sf.read(files[12])
data4, _ = sf.read(files[3])
data5, _ = sf.read(files[9])  # 1R
data6, _ = sf.read(files[15])  # 4R

f1, p1, l1 = psd(data1[:, 0], fs, cal)
f2, p2, l2 = psd(data2[:, 0], fs, cal)
f3, p3, l3 = psd(data3[:, 0], fs, cal)
f4, p4, l4 = psd(data4[:, 0], fs, cal)
f5, p5, l5 = psd(data5[:, 0], fs, cal)
f6, p6, l6 = psd(data4[:, 0], fs, cal)

M20 = np.array([l1, l4, l2, l5, l3, l6])
axl.plot(rs, M20, 'ko-', label='Mach 0.20')

fig, ax = plt.subplots()
ax.plot(f1, p1, label='0.125R, Mach 0.20')
ax.plot(f4, p4, label='0.250R, Mach 0.20')
ax.plot(f2, p2, label='0.500R, Mach 0.20')
ax.plot(f3, p3, label='2.000R, Mach 0.20')

savemat('wav_analysis/M20.mat',
        {'f125': f1, 'f250': f4, 'f500': f2, 'f2000': f3, 'p125': p1, 'p250': p4, 'p500': p2, 'p2000': p3})

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('Frequency, Hz')
ax.set_ylabel('SPL, dB')
ax.legend()

fig.savefig('wav_analysis/M20.pdf')

# Mach 0.25
data1, fs = sf.read(files[1])
data2, _ = sf.read(files[7])
data3, _ = sf.read(files[13])
data4, _ = sf.read(files[4])
data5, _ = sf.read(files[10])  # 1R
data6, _ = sf.read(files[16])  # 4R

f1, p1, l1 = psd(data1[:, 0], fs, cal)
f2, p2, l2 = psd(data2[:, 0], fs, cal)
f3, p3, l3 = psd(data3[:, 0], fs, cal)
f4, p4, l4 = psd(data4[:, 0], fs, cal)
f5, p5, l5 = psd(data5[:, 0], fs, cal)
f6, p6, l6 = psd(data4[:, 0], fs, cal)

M25 = np.array([l1, l4, l2, l5, l3, l6])
axl.plot(rs, [l1, l4, l2, l5, l3, l6], 'ro-', label='Mach 0.25')

fig, ax = plt.subplots()
ax.plot(f1, p1, label='0.125R, Mach 0.25')
ax.plot(f4, p4, label='0.250R, Mach 0.25')
ax.plot(f2, p2, label='0.500R, Mach 0.25')
ax.plot(f3, p3, label='2.000R, Mach 0.25')
savemat('wav_analysis/M25.mat',
        {'f125': f1, 'f250': f4, 'f500': f2, 'f2000': f3, 'p125': p1, 'p250': p4, 'p500': p2, 'p2000': p3})

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('Frequency, Hz')
ax.set_ylabel('SPL, dB')
ax.legend()
fig.savefig('wav_analysis/M25.pdf')

# Mach 0.30
data1, fs = sf.read(files[2])
data2, _ = sf.read(files[8])
data3, _ = sf.read(files[14])
data4, _ = sf.read(files[5])
data5, _ = sf.read(files[11])  # 1R
data6, _ = sf.read(files[17])  # 4R

f1, p1, l1 = psd(data1[:, 0], fs, cal)
f2, p2, l2 = psd(data2[:, 0], fs, cal)
f3, p3, l3 = psd(data3[:, 0], fs, cal)
f4, p4, l4 = psd(data4[:, 0], fs, cal)
f5, p5, l5 = psd(data5[:, 0], fs, cal)
f6, p6, l6 = psd(data4[:, 0], fs, cal)

M30 = np.array([l1, l4, l2, l5, l3, l6])
axl.plot(rs, [l1, l4, l2, l5, l3, l6], 'bo-', label='Mach 0.30')
axl.legend()
axl.set_xlabel('Rotor Separation in Radii')
axl.set_ylabel('SPL, dB')

fig, ax = plt.subplots()
ax.plot(f1, p1, label='0.125R, Mach 0.30')
ax.plot(f4, p4, label='0.250R, Mach 0.30')
ax.plot(f2, p2, label='0.500R, Mach 0.30')
ax.plot(f3, p3, label='2.000R, Mach 0.30')
savemat('wav_analysis/M30.mat',
        {'f125': f1, 'f250': f4, 'f500': f2, 'f2000': f3, 'p125': p1, 'p250': p4, 'p500': p2, 'p2000': p3})

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('Frequency, Hz')
ax.set_ylabel('SPL, dB')
ax.legend()
fig.savefig('wav_analysis/M30.pdf')

savemat('wav_analysis/trends.mat', {'radii': rs, 'M20': M20, 'M25': M25, 'M30': M30})
figl.savefig('wav_analysis/trends.pdf')
plt.show(block=True)
