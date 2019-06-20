import numpy as np
from sympl import DiagnosticComponent
from marble.components.marble import pc_ds, name_feature_counts


class InputHeightToPrincipalComponents(DiagnosticComponent):

    input_properties = {
        'liquid_water_static_energy': {
            'dims': ['*', 'z_star'],
            'units': 'J/kg',
            'alias': 'sl',
        },
        'total_water_mixing_ratio': {
            'dims': ['*', 'z_star'],
            'units': 'kg/kg',
            'alias': 'rt',
        },
        'vertical_wind': {
            'dims': ['*', 'z_star'],
            'units': 'm/s',
            'alias': 'w'
        },
    }

    diagnostic_properties = {
        'liquid_water_static_energy_components': {
            'dims': ['*', 'sl_latent'],
            'units': '',
            'alias': 'sl_latent',
        },
        'total_water_mixing_ratio_components': {
            'dims': ['*', 'rt_latent'],
            'units': '',
            'alias': 'rt_latent',
        },
        'vertical_wind_components': {
            'dims': ['*', 'w_latent'],
            'units': '',
            'alias': 'w_latent'
        },
    }

    def array_call(self, state):
        diagnostic_dict = {}
        for name in 'w', 'sl', 'rt':
            diagnostic_dict[f'{name}_latent'] = convert_height_to_principal_components(state[name], name, subtract_mean=True)
        return diagnostic_dict


def convert_height_to_principal_components(array, basis_name, subtract_mean=True):
    if subtract_mean:
        array = array - pc_ds[f'{basis_name}_mean'].values
    return np.dot(
        array,
        pc_ds[f'{basis_name}_principal_components'].values[:name_feature_counts[basis_name], :].T
    )


def convert_principal_components_to_height(array, basis_name, add_mean=True):
    result = np.dot(
        array,
        pc_ds[f'{basis_name}_principal_components'].values[:name_feature_counts[basis_name], :]
    )
    if add_mean:
        result += pc_ds[f'{basis_name}_mean'].values
    return result


class InputPrincipalComponentsToHeight(DiagnosticComponent):

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
    }

    diagnostic_properties = {
        'liquid_water_static_energy': {
            'dims': ['*', 'z_star'],
            'units': 'J/kg',
            'alias': 'sl',
        },
        'total_water_mixing_ratio': {
            'dims': ['*', 'z_star'],
            'units': 'kg/kg',
            'alias': 'rt',
        },
        'vertical_wind': {
            'dims': ['*', 'z_star'],
            'units': 'm/s',
            'alias': 'w'
        },
    }

    def array_call(self, state):
        diagnostic_dict = {}
        for name in 'sl', 'rt', 'w':
            diagnostic_dict[name] = convert_principal_components_to_height(
                state[name], basis_name=name, add_mean=True
            )
        return diagnostic_dict


class DiagnosticPrincipalComponentsToHeight(DiagnosticComponent):

    input_properties = {
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
    }

    diagnostic_properties = {
        'cloud_water_mixing_ratio': {
            'dims': ['*', 'z_star'],
            'units': '',
            'alias': 'rcld',
        },
        'rain_water_mixing_ratio': {
            'dims': ['*', 'z_star'],
            'units': '',
            'alias': 'rrain',
        },
        'cloud_fraction': {
            'dims': ['*', 'z_star'],
            'units': '',
            'alias': 'cld',
        },
        'clear_sky_radiative_heating_rate': {
            'dims': ['*', 'z_star'],
            'units': 'degK hr^-1',
            'alias': 'sl_rad_clr',
        },
    }

    def array_call(self, state):
        diagnostic_dict = {}
        for name in 'rcld', 'rrain', 'cld':
            diagnostic_dict[name] = convert_principal_components_to_height(
                state[name], basis_name=name, add_mean=True,
            )
        diagnostic_dict['sl_rad_clr'] = convert_principal_components_to_height(
            state['sl_rad_clr'], basis_name='sl', add_mean=False
        )
        return diagnostic_dict
