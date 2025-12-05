import numpy as np
import pytest
from flight_acoustics import ega


def test_ega_basic_pure_tone():
    """Test EGA with simple pure tone configuration."""
    hs = 10.0  # ft
    hr = 5.0   # ft
    d2 = 100.0 # ft
    f = 500.0  # Hz
    a = 1116.0 # ft/s (speed of sound)
    flores = 200.0
    
    atten, phase = ega(hs, hr, d2, f, a, flores, pt=True, cturb=0.0)
    
    # Basic sanity checks
    assert np.isfinite(atten)
    assert np.isfinite(phase)
    assert isinstance(atten, (float, np.ndarray))
    assert isinstance(phase, (float, np.ndarray))


def test_ega_broadband():
    """Test EGA in broadband mode."""
    hs = 10.0
    hr = 5.0
    d2 = 100.0
    f = 1000.0
    a = 1116.0
    flores = 200.0
    
    atten, phase = ega(hs, hr, d2, f, a, flores, pt=False, cturb=0.0)
    
    assert np.isfinite(atten)
    assert np.isnan(phase)  # Phase should be NaN for broadband


def test_ega_array_frequencies():
    """Test EGA with array of frequencies."""
    hs = 10.0
    hr = 5.0
    d2 = 100.0
    f = np.array([100.0, 500.0, 1000.0, 2000.0])
    a = 1116.0
    flores = 200.0
    
    atten, phase = ega(hs, hr, d2, f, a, flores, pt=True, cturb=0.0)
    
    assert atten.shape == f.shape
    assert phase.shape == f.shape
    assert np.all(np.isfinite(atten))
    assert np.all(np.isfinite(phase))


def test_ega_with_turbulence():
    """Test EGA with turbulence parameter."""
    hs = 10.0
    hr = 5.0
    d2 = 100.0
    f = 500.0
    a = 1116.0
    flores = 200.0
    cturb = 16e-4  # rad·s·√m (converted to ft units if needed)
    
    atten_turb, phase_turb = ega(hs, hr, d2, f, a, flores, pt=False, cturb=cturb)
    atten_no_turb, phase_no_turb = ega(hs, hr, d2, f, a, flores, pt=False, cturb=0.0)
    
    assert np.isfinite(atten_turb)
    assert np.isfinite(atten_no_turb)
    # Turbulence should affect attenuation
    assert atten_turb != atten_no_turb


def test_ega_ground_effect():
    """Test that ground impedance affects results."""
    hs = 10.0
    hr = 5.0
    d2 = 100.0
    f = 500.0
    a = 1116.0
    
    # Different ground types (different flow resistance)
    atten_soft, _ = ega(hs, hr, d2, f, a, flores=50.0, pt=True)
    atten_hard, _ = ega(hs, hr, d2, f, a, flores=500.0, pt=True)
    
    assert np.isfinite(atten_soft)
    assert np.isfinite(atten_hard)
    # Different ground should give different attenuation
    assert atten_soft != atten_hard


def test_ega_geometry_effect():
    """Test that geometry affects phase delay."""
    hr = 5.0
    d2 = 100.0
    f = 500.0
    a = 1116.0
    flores = 200.0
    
    # Different source heights
    atten_low, phase_low = ega(hs=5.0, hr=hr, d2=d2, f=f, a=a, flores=flores, pt=True)
    atten_high, phase_high = ega(hs=20.0, hr=hr, d2=d2, f=f, a=a, flores=flores, pt=True)
    
    assert np.isfinite(atten_low) and np.isfinite(atten_high)
    assert np.isfinite(phase_low) and np.isfinite(phase_high)
    # Different geometry should give different results
    assert (atten_low != atten_high) or (phase_low != phase_high)
