"""Use a noninteractive backend for automated plotting tests."""
import os

os.environ['MPLBACKEND'] = 'Agg'
