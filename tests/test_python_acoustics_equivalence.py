from pathlib import Path

import numpy as np
import pytest

import panam_acoustics
from panam_acoustics import filters as pa_filters

FIXTURE_PATH = Path(__file__).parent / "data" / "python_acoustics_reference.npz"


@pytest.mark.parametrize("zero_phase", [False, True])
@pytest.mark.parametrize("cutoff", [50.0, 200.0])
def test_highpass_equivalence(zero_phase, cutoff):
    data = np.load(FIXTURE_PATH)
    fs = float(data["fs"])
    signal = data["signal_hp"]
    order = int(data["order"])

    key = f"hp_cut{cutoff}_zp{int(zero_phase)}"
    ref = data[key]
    test = pa_filters.highpass(signal, float(cutoff), fs, order=order, zero_phase=zero_phase)

    if zero_phase:
        trim = int(0.05 * fs)
        ref = ref[trim:-trim]
        test = test[trim:-trim]
        assert np.allclose(test, ref, rtol=5e-2, atol=5e-4)
    else:
        assert np.allclose(test, ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("zero_phase", [False, True])
@pytest.mark.parametrize("cutoff", [50.0, 200.0])
def test_lowpass_equivalence(zero_phase, cutoff):
    data = np.load(FIXTURE_PATH)
    fs = float(data["fs"])
    signal = data["signal_lp"]
    order = int(data["order"])

    key = f"lp_cut{cutoff}_zp{int(zero_phase)}"
    ref = data[key]
    test = pa_filters.lowpass(signal, float(cutoff), fs, order=order, zero_phase=zero_phase)

    if zero_phase:
        trim = int(0.05 * fs)
        ref = ref[trim:-trim]
        test = test[trim:-trim]
        assert np.allclose(test, ref, rtol=5e-2, atol=5e-4)
    else:
        assert np.allclose(test, ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "temperature, pressure, rh",
    [
        (293.15, 101.325, 20.0),
        (278.15, 90.0, 60.0),
        (310.15, 105.0, 10.0),
    ],
)
def test_atmosphere_attenuation_equivalence(temperature, pressure, rh):
    data = np.load(FIXTURE_PATH)
    freqs = data["freqs"]
    key = f"att_T{temperature}_P{pressure}_RH{rh}"
    ref = data[key]

    test = panam_acoustics.Atmosphere(
        temperature=temperature,
        pressure=pressure,
        relative_humidity=rh,
    ).attenuation_coefficient(freqs)

    assert np.allclose(test, ref, rtol=1e-10, atol=1e-12)
