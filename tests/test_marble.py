#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `marble` package."""

import unittest
import marble
import numpy as np


test_era5_filename = '/home/twine/data/era5/era5-interp-2016.nc'


class TestPrincipalComponentConversions(unittest.TestCase):
    """Tests for `marble` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_create_height_initial_state(self):
        result = marble.get_era5_state(test_era5_filename, latent=False)
        self.assertNotIn('vertical_wind', result)

    def test_create_latent_initial_state(self):
        result = marble.get_era5_state(test_era5_filename, latent=True)
        self.assertNotIn('vertical_wind_components', result)

    def test_convert_input_latent_to_height_and_back(self):
        state = marble.get_era5_state(test_era5_filename, latent=True)
        state.update(marble.get_era5_forcing(test_era5_filename, latent=True, i_timestep=0))
        converter = marble.InputPrincipalComponentsToHeight()
        inverse_converter = marble.InputHeightToPrincipalComponents()
        intermediate = converter(state)
        intermediate['time'] = state['time']
        result = inverse_converter(intermediate)
        for name in result.keys():
            self.assertIn(name, state)
            self.assertEqual(result[name].shape, state[name].shape, name)
            self.assertTrue(np.allclose(result[name].values, state[name].values), name)


if __name__ == '__main__':
    unittest.main()
