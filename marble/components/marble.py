# -*- coding: utf-8 -*-
"""Main module."""
from sympl import TendencyComponent
import ast
import os
import xarray as xr
import numpy as np

data_path = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    ),
    'data'
)
weight_ds = xr.open_dataset(os.path.join(data_path, 'weights-nep-mb19.nc'))
pc_ds = xr.open_dataset(os.path.join(data_path, 'era5-pc-mb19.nc'))


state_name_list = ast.literal_eval(weight_ds.state_name_list)
# advective terms are not NN inputs but are included in this list due to a bug
# must add state terms
pbl_input_name_list = state_name_list + ast.literal_eval(weight_ds.pbl_input_name_list)
diagnostic_name_list = ast.literal_eval(weight_ds.pbl_diagnostics_name_list)
name_feature_counts = ast.literal_eval(weight_ds.name_feature_counts)
decomposition_names = ast.literal_eval(weight_ds.decomposition_name_mapping)


def zero_sl_rad_clr_mean(weight_ds):
    """Need to set mean for sl_rad_clr to zero because its mean is not actually
    subtracted during neural network training.
    """
    if 'sl_rad_clr' in diagnostic_name_list:
        i_sl_rad_clr = diagnostic_name_list.index('sl_rad_clr')
        i_start = 0
        for i_name in range(0, i_sl_rad_clr):
            i_start += name_feature_counts.get(diagnostic_name_list[i_name], 1)
        i_end = i_start + name_feature_counts['sl_rad_clr']
        weight_ds['diagnostic_mean'][i_start:i_end] = 0.

zero_sl_rad_clr_mean(weight_ds)


def concatenate_pbl_input(state):
    concat_list = []
    for name in pbl_input_name_list:
        array = state[name]
        if len(array.shape) == 2:
            concat_list.append(array)
        else:
            concat_list.append(array[:, None])
    return np.concatenate(concat_list, axis=1)


def normalize_pbl_input(pbl_input_array):
    pbl_input_array -= weight_ds['pbl_input_mean'].values[None, :]
    pbl_input_array /= weight_ds['pbl_input_scale'].values[None, :]


def denormalize_diagnostic_output(diagnostic_array):
    diagnostic_array *= weight_ds['diagnostic_scale'].values[None, :]
    diagnostic_array += weight_ds['diagnostic_mean'].values[None, :]


def denormalize_state(state_array, add_mean=True):
    state_array *= weight_ds['state_scale'].values[None, :]
    if add_mean:
        state_array += weight_ds['state_mean'].values[None, :]


def get_network_outputs(pbl_input_array):
    X = pbl_input_array
    X = np.dot(X, weight_ds['pbl_encoder_W'].values) + weight_ds['pbl_encoder_b'].values
    X[X < 0.] = 0.
    X = np.dot(X, weight_ds['pbl_hidden_W'].values) + weight_ds['pbl_hidden_b'].values
    X[X < 0.] = 0.
    tend_pbl = np.dot(X, weight_ds['pbl_tend_decoder_W'].values) + weight_ds['pbl_tend_decoder_b'].values
    diag = np.dot(X, weight_ds['pbl_diag_decoder_W'].values) + weight_ds['pbl_diag_decoder_b'].values
    return tend_pbl, diag


def get_diagnostic_dict_from_array(diagnostic_array):
    """Splits up a [*, n_latent] array of diagnostics into individual quantity
    arrays."""
    out_dict = {}
    i_latent = 0
    for name in diagnostic_name_list:
        n_latent = name_feature_counts.get(name, 1)
        out_dict[name] = diagnostic_array[:, i_latent:i_latent+n_latent]
        i_latent += n_latent
    for name, array in out_dict.items():
        if out_dict[name].shape[1] == 1:
            out_dict[name] = array[:, 0]  # remove dummy dimension
    return out_dict


def get_state_dict_from_array(state_array):
    """Splits up a [*, n_latent] state array into individual quantity arrays."""
    out_dict = {}
    i_latent = 0
    for name in state_name_list:
        n_latent = name_feature_counts[name]
        out_dict[name] = state_array[:, i_latent:i_latent+n_latent]
        i_latent += n_latent
    return out_dict


class LatentMarble(TendencyComponent):
    """
    MARBLE component which works in latent space (inputs and outputs
    denormalized principal components) without converting to or from the
    real height coordinate.
    """

    input_properties = {
        'liquid_water_static_energy_components': {
            'dims': ['*', 'sl_latent'],
            'units': '',
            'alias': 'sl',
        },
        'total_water_mixing_ratio_components': {
            'dims': ['*', 'rt_latent'],
            'units': '',
            'alias': 'rt',
        },
        'vertical_wind_components': {
            'dims': ['*', 'w_latent'],
            'units': '',
            'alias': 'w'
        },
        'liquid_water_static_energy_at_3km': {
            'dims': ['*'],
            'units': 'J/kg',
            'alias': 'sl_domain_top',
        },
        'total_water_mixing_ratio_at_3km': {
            'dims': ['*'],
            'units': 'kg/kg',
            'alias': 'rt_domain_top',
        },
        'surface_latent_heat_flux': {
            'dims': ['*'],
            'units': 'W/m^2',
            'alias': 'lhf',
        },
        'surface_sensible_heat_flux': {
            'dims': ['*'],
            'units': 'W/m^2',
            'alias': 'shf',
        },
        'surface_temperature': {
            'dims': ['*'],
            'units': 'degK',
            'alias': 'sst',
        },
        'mid_cloud_fraction': {
            'dims': ['*'],
            'units': '',
            'alias': 'cldmid',
        },
        'high_cloud_fraction': {
            'dims': ['*'],
            'units': '',
            'alias': 'cldhigh',
        },
        'downwelling_shortwave_radiation_at_top_of_atmosphere': {
            'dims': ['*'],
            'units': 'W/m^2',
            'alias': 'swdn_toa',
        },
        'downwelling_shortwave_radiation_at_3km': {
            'dims': ['*'],
            'units': 'W/m^2',
            'alias': 'swdn_tod',
        },
        'surface_air_pressure': {
            'dims': ['*'],
            'units': 'Pa',
            'alias': 'p_surface',
        },
        'rain_water_mixing_ratio_at_3km': {
            'dims': ['*'],
            'units': 'kg/kg',
            'alias': 'rrain_domain_top',
        }
    }

    tendency_properties = {
        'liquid_water_static_energy_components': {
            'dims': ['*', 'sl_latent'],
            'units': 'hr^-1',
            'alias': 'sl',
        },
        'total_water_mixing_ratio_components': {
            'dims': ['*', 'rt_latent'],
            'units': 'hr^-1',
            'alias': 'rt',
        },
    }

    diagnostic_properties = {
        'cloud_water_mixing_ratio_components': {
            'dims': ['*', 'rcld_latent'],
            'units': '',
            'alias': 'rcld',
        },
        'rain_water_mixing_ratio_components': {
            'dims': ['*', 'rrain_latent'],
            'units': '',
            'alias': 'rrain',
        },
        'cloud_fraction_components': {
            'dims': ['*', 'cld_latent'],
            'units': '',
            'alias': 'cld',
        },
        'clear_sky_radiative_heating_rate_components': {
            'dims': ['*', 'sl_latent'],
            'units': 'hr^-1',
            'alias': 'sl_rad_clr',
        },
        'low_cloud_fraction': {
            'dims': ['*'],
            'units': '',
            'alias': 'cldlow',
        },
        'surface_precipitation_rate': {
            'dims': ['*'],
            'units': 'mm/hr',
            'alias': 'precip',
        },
        'column_cloud_water': {
            'dims': ['*'],
            'units': 'kg/m^2',
            'alias': 'ccw',
        },
        'height': {
            'dims': ['z_star'],
            'units': 'm',
            'alias': 'z',
        }
    }

    def array_call(self, state):
        state['lhf'] *= 3600.  # NN expects J/m^2 integrated over an hour
        state['shf'] *= 3600.
        pbl_input_array = concatenate_pbl_input(state)
        normalize_pbl_input(pbl_input_array)
        tendency_array, diagnostic_array = get_network_outputs(pbl_input_array)
        denormalize_diagnostic_output(diagnostic_array)
        diagnostic_dict = get_diagnostic_dict_from_array(diagnostic_array)
        denormalize_state(tendency_array, add_mean=False)
        tendency_dict = get_state_dict_from_array(tendency_array)
        tendency_dict['sl'] += diagnostic_dict['sl_rad_clr']
        diagnostic_dict['z'] = np.linspace(0., 3000., 20)
        return tendency_dict, diagnostic_dict

