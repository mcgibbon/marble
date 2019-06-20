#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `marble` package."""

import unittest
import marble
import numpy as np
import sympl as sp


test_era5_filename = '/home/twine/data/era5/era5-interp-2016.nc'


def get_test_state(pc_value=0.):
    n_features = marble.components.marble.name_feature_counts
    state = {
        'time': sp.timedelta(0),
        'liquid_water_static_energy_components': sp.DataArray(
            np.ones([n_features['sl']]) * pc_value,
            dims=('sl_latent',), attrs={'units': ''}),
        'total_water_mixing_ratio_components': sp.DataArray(
            np.ones([n_features['rt']]) * pc_value,
            dims=('rt_latent',), attrs={'units': ''}),
        'cloud_water_mixing_ratio_components': sp.DataArray(
            np.ones([n_features['rcld']]) * pc_value,
            dims=('rcld_latent',), attrs={'units': ''}),
        'rain_water_mixing_ratio_components': sp.DataArray(
            np.ones([n_features['rrain']]) * pc_value,
            dims=('rrain_latent',), attrs={'units': ''}),
        'cloud_fraction_components': sp.DataArray(
            np.ones([n_features['cld']]) * pc_value,
            dims=('cld_latent',), attrs={'units': ''}),
        'liquid_water_static_energy_components_horizontal_advective_tendency': sp.DataArray(
            np.ones([n_features['sl']]) * pc_value,
            dims=('sl_latent',), attrs={'units': ''}),
        'total_water_mixing_ratio_components_horizontal_advective_tendency': sp.DataArray(
            np.ones([n_features['sl']]) * pc_value,
            dims=('rt_latent',), attrs={'units': ''}),
        'vertical_wind_components': sp.DataArray(
            np.ones([n_features['w']]) * pc_value,
            dims=('w_latent',), attrs={'units': ''}),
    }
    return state


class TestPrincipalComponentConversions(unittest.TestCase):
    """Tests for `marble` package."""

    def test_convert_input_zero_latent_to_height_and_back(self):
        state = get_test_state(pc_value=0.)
        converter = marble.InputPrincipalComponentsToHeight()
        inverse_converter = marble.InputHeightToPrincipalComponents()
        intermediate = converter(state)
        intermediate['time'] = state['time']
        result = inverse_converter(intermediate)
        for name in result.keys():
            self.assertIn(name, state)
            self.assertEqual(result[name].shape, state[name].shape, name)
            self.assertTrue(np.allclose(result[name].values, state[name].values), name)

    def test_convert_input_nonzero_latent_to_height_and_back(self):
        state = get_test_state(pc_value=0.6)
        converter = marble.InputPrincipalComponentsToHeight()
        inverse_converter = marble.InputHeightToPrincipalComponents()
        intermediate = converter(state)
        intermediate['time'] = state['time']
        result = inverse_converter(intermediate)
        for name in result.keys():
            self.assertIn(name, state)
            self.assertEqual(result[name].shape, state[name].shape, name)
            self.assertTrue(np.allclose(result[name].values, state[name].values), name)

    def test_convert_diagnostic_zero_latent_to_height(self):
        """
        This only tests that the conversion runs without errors, it does not
        check anything about the output value.
        """
        state = get_test_state(pc_value=0.)
        converter = marble.DiagnosticPrincipalComponentsToHeight()
        result = converter(state)



if __name__ == '__main__':
    unittest.main()
