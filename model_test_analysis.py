import acoustics
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.io import wavfile

from flight_acoustics import psd

r_ref = 16.7
fs = 51200
rpm = 1400
rpm_ref = 700
frequency_scale = rpm_ref / rpm
scalename = "scaled"
scaled_fs = int(fs * frequency_scale)

geartob = np.array([1000, 1250, 1600, 2000]) * 2 * frequency_scale
geargain = [-4.0, -20.0, -14.0, -4.0]
gearfilter = False

if gearfilter is True:
    filtername = "gear_filter_"
else:
    filtername = ""

cases = {
    "r1332": ("rellis/", "rellis_16p7_1400_01_n3_1332-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "r1337": ("rellis/", "rellis_16p7_1400_01_n3_1337-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "r1326": ("rellis/", "rellis_16p7_1400_13_08_1326-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "r1324": ("rellis/", "rellis_16p7_1400_14_09_1324-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "r1321": ("rellis/", "rellis_16p7_1400_17_12_1321-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "r1320": ("rellis/", "rellis_16p7_1400_18_12_1320-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "r1330": ("rellis/", "rellis_16p7_1400_21_14_1330-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "r1328": ("rellis/", "rellis_16p7_1400_21_15_1328-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "q1300": ("rellis/", "rellis_quad_hover_1_1300-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "q1303": ("rellis/", "rellis_quad_hover_2_1303-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "q1305": ("rellis/", "rellis_quad_hover_3_1305-Raw.txt", "rellis/", "rellis_calib_2_1346-Raw.txt", 'rellis/',
              'rellis_16p7_amb_1316-Raw.txt', 16.7),
    "d7L1": ("distance_7ft", "outside_07_1400_20_1_2126-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
             'distance_7ft/', 'outside_7_amb_1_2128-Raw.txt', 7.0),
    "d7L2": ("distance_7ft", "outside_07_1400_20_2_2127-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
             'distance_7ft/', 'outside_7_amb_1_2128-Raw.txt', 7.0),
    "d12L2": ("distance_12ft", "outside_12_1400_20_2_L_2138-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
              'distance_12ft/', 'outside_12_amb_2_L_2143-Raw.txt', 12.0),
    "d12L3": ("distance_12ft", "outside_12_1400_20_3_L_2146-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
              'distance_12ft/', 'outside_12_amb_2_L_2143-Raw.txt', 12.0),
    "d12L4": ("distance_12ft", "outside_12_1400_20_4_L_2147-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
              'distance_12ft/', 'outside_12_amb_2_L_2143-Raw.txt', 12.0),
    "d12C00L1": ("distance_12ft", "outside_12_1400_00_1_L_2149-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
                 'distance_12ft/', 'outside_12_amb_2_L_2143-Raw.txt', 12.0),
    "d12C00L2": ("distance_12ft", "outside_12_1400_00_2_L_2150-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
                 'distance_12ft/', 'outside_12_amb_2_L_2143-Raw.txt', 12.0),
    "d14L1": ("distance_14ft", "outside_14_1400_20_1_L_2157-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
              'distance_14ft/', 'outside_14_amb_1_L_2154-Raw.txt', 14.0),
    "d14L2": ("distance_14ft", "outside_14_1400_20_2_L_2158-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
              'distance_14ft/', 'outside_14_amb_1_L_2154-Raw.txt', 14.0),
    "d14L3": ("distance_14ft", "outside_14_1400_20_3_L_2202-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
              'distance_14ft/', 'outside_14_amb_1_L_2154-Raw.txt', 14.0),
    "d16p7L1": ("distance_16p7", "outside_16p7_1400_20_1_2108-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
                'distance_16p7/', 'outside_16p7_amb_1_2104-Raw.txt', 16.7),
    "d16p7L2": ("distance_16p7", "outside_16p7_1400_20_2_2114-Raw.txt", 'distance_14ft/', 'outside_calib_2122-Raw.txt',
                'distance_16p7/', 'outside_16p7_amb_1_2104-Raw.txt', 16.7)
}

font = {'family': 'serif',
        'weight': 'normal',
        'size': 10}

plt.style.use('ggplot')
matplotlib.rc('font', **font)
matplotlib.rcParams.update({'mathtext.fontset': 'dejavuserif'})
matplotlib.rcParams.update({'figure.autolayout': True})


def load_signal(filename, fs, k=1.0):
    data = np.genfromtxt(filename, skip_header=5)
    p = data[:, 2]
    return acoustics.Signal(p * k, fs)


def third_octave_spectrum(sig):
    tob, pob = sig.third_octaves()
    fc = tob.center
    lo = []
    for p in pob:
        lo.append(p.leq())
    return fc, np.array(lo)


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def tobfilter(signal, tobs, gains):
    to = signal.third_octaves(zero_phase=True)
    full_tob = to[0].center
    full_gains = np.ones_like(full_tob)
    for (gi, tob) in enumerate(tobs):
        idx = find_nearest_index(full_tob, tob)
        full_gains[idx] = 10.0 ** (gains[gi] / 20.0)
    signal = 0.0 * to[1][0]
    for (si, tosignal) in enumerate(to[1]):
        signal = signal + full_gains[si] * tosignal
    return signal


def run_case(case):
    (folder, file, calfolder, calfile, ambfolder, ambfile, r) = cases[case]
    geometric_scaling = r / r_ref
    # Get calibration factor
    c = load_signal(calfolder + "/" + calfile, fs)
    kcal = 1.0 / np.sqrt(np.mean(c ** 2.0))

    # Calibrate signals
    s = geometric_scaling * load_signal(folder + "/" + file, fs, k=kcal)
    a = geometric_scaling * load_signal(folder + "/" + ambfile, fs, k=kcal)

    a1 = a[fs:2 * fs - 1]
    a1 = a1.highpass(70)
    s1 = s[fs:2 * fs - 1]
    s1 = s1.highpass(70)

    # Generate PSD
    fa1, pa1, la1 = psd(a1, fs)
    fs1, ps1, ls1 = psd(s1, fs)

    # Resolve to full-scale
    s_s = geometric_scaling * load_signal(folder + "/" + file, scaled_fs, k=kcal)
    a_s = geometric_scaling * load_signal(folder + "/" + ambfile, scaled_fs, k=kcal)

    # High pass
    s_s = s_s.highpass(30 * 2 * frequency_scale, zero_phase=True)
    a_s = a_s.highpass(30 * 2 * frequency_scale, zero_phase=True)

    # Apply gear filter
    if gearfilter:
        s_s = tobfilter(s_s, geartob, geargain)

    # 1/3 oct
    tob, lob = third_octave_spectrum(s_s)
    toba, loba = third_octave_spectrum(a_s)

    # A-weighted 1/3 oct
    tobw, lobw = third_octave_spectrum(s_s.weigh('A'))

    return s1, s_s, fs1, ps1, tob, lob, a1, a_s, fa1, pa1, toba, loba, tobw, lobw, folder


def plot_cases(cases):
    for case in cases:
        s1, s_s, fs1, ps1, tob, lob, a1, a_s, fa1, pa1, toba, loba, tobw, lobw, folder = run_case(case)

        fig, ax = plt.subplots()
        ax.plot(fs1, ps1, label='Signal:   {:0.1f} dBA, {:0.1f} dB'.format(s1.weigh('A').leq(), s1.weigh('Z').leq()))
        ax.plot(fa1, pa1, label='Ambient:  {:0.1f} dBA, {:0.1f} dB'.format(a1.weigh('A').leq(), a1.weigh('Z').leq()))
        ax.set_xlim((0, 5000))
        ax.set_ylim((20, 85))
        ax.set_xlabel('Frequency, Hz')
        ax.set_ylabel('Power Spectral Density, dB')
        ax.legend(loc='best')
        fig.savefig(folder + "/" + case + "_" + filtername + 'geo_PSD.png')
        plt.close()

        fig, ax = plt.subplots()
        ax.semilogx(tob, lob,
                    label='Signal:  {:0.1f} dBA, {:0.1f} dB'.format(s_s.weigh('A').leq(), s_s.weigh('Z').leq()))
        ax.semilogx(toba, loba,
                    label='Ambient: {:0.1f} dBA, {:0.1f} dB'.format(a_s.weigh('A').leq(), a_s.weigh('Z').leq()))
        ax.set_xlabel('Frequency, Hz')
        ax.set_ylabel('SPL, dB')
        ax.set_ylim((35, 90))
        ax.set_xlim([10, 10e3])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
        ax.legend(loc='best')
        fig.savefig(folder + "/" + case + "_" + filtername + scalename + '_TOB.png')
        plt.close()

        # Write out scaled wavefile
        kfactor = 0.3 * 32767
        wavsignal = (kfactor * s_s).astype(np.int16)
        wavfile.write(folder + "/" + case + "_" + filtername + scalename + '.wav', int(s_s.fs), wavsignal)

        # Write out MATLAB data
        savemat(folder + "/" + case + "_" + filtername + scalename + '.mat',
                {'signal': s_s, 'ambient': a_s, 'third_octave_band_frequencies': tob, 'third_octave_band_levels': lob,
                 'third_octave_band_ambient': loba, 'SPL_A': s_s.weigh('A').leq(), 'SPL_Z': s_s.leq()})


def compare_cases(caselist):
    fig, ax = plt.subplots()
    for case in caselist:
        s1, s_s, fs1, ps1, tob, lob, a1, a_s, fa1, pa1, toba, loba, tobw, lobw, folder = run_case(case)
        ax.semilogx(tob, lob, label=case + ': {:0.1f} dBA'.format(s_s.weigh('A').leq()))
    ax.semilogx(toba, loba, label="Ambient")
    ax.set_xlabel('Frequency, Hz')
    ax.set_ylabel('SPL, dB')
    ax.set_ylim((35, 90))
    ax.set_xlim([10, 10e3])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.legend(loc='best')
    fig.savefig(folder + "/" + "compare" + "_" + filtername + scalename + '_TOB.png')


if __name__ == "__main__":
    #caselist = ['r1326', 'r1324', 'r1321', 'r1320', 'r1330', 'r1328']
    #compare_cases(caselist)
    plot_cases(['r1330'])
