import numpy as np
from sympl import DiagnosticComponent
from marble.components.marble import pc_ds, name_feature_counts
from marble.docstrings import document_properties


@document_properties
class InputHeightToPrincipalComponents(DiagnosticComponent):
    """
    Converts MARBLE's vertically-resolved inputs from height coordinates to
    principal components.
    """

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
    """
    Converts a numpy array from height coordinates on a 20-point equidistant grid
    from 0 to 3km (inclusive) into principal components required by MARBLE.

    Args:
        array: numpy array whose final dimension is of size 20
        basis_name: short alias name of the quantity whose principal components
            to use. For example, 'rt', 'sl', 'cld', 'rcld', 'rrain', or 'w'.
        subtract_mean: whether to subtract the mean vertical profile of the
            basis quantity from the numpy array before converting into principal
            components. Generally this is True if you are converting the basis
            quantity itself, and False if you are converting a difference to
            apply to the basis quantity (such as a tendency).

    Returns:
        return_array: numpy array whose final dimension length is equal to the
            number of principal components used for hte basis quantity.
    """
    if subtract_mean:
        array = array - pc_ds[f'{basis_name}_mean'].values
    return np.dot(
        array,
        pc_ds[f'{basis_name}_principal_components'].values[:name_feature_counts[basis_name], :].T
    )


def convert_principal_components_to_height(array, basis_name, add_mean=True):
    """
    Converts a numpy array from principal components as used by MARBLE to
    height coordinates on a 20-point equidistant grid from 0 to 3km (inclusive).

    Args:
        array: numpy array whose final dimension is principal component number
        basis_name: short alias name of the quantity whose principal components
            are used. For example, 'rt', 'sl', 'cld', 'rcld', 'rrain', or 'w'.
        add_mean: whether to add in the mean vertical profile of the
            basis quantity from the numpy array after converting to height
            coordinates. Generally this is True if you are converting the basis
            quantity itself, and False if you are converting a difference
            applied to the basis quantity (such as a tendency).

    Returns:
        return_array: numpy array whose final dimension length is 20.
    """
    result = np.dot(
        array,
        pc_ds[f'{basis_name}_principal_components'].values[:name_feature_counts[basis_name], :]
    )
    if add_mean:
        result += pc_ds[f'{basis_name}_mean'].values
    return result


@document_properties
class InputPrincipalComponentsToHeight(DiagnosticComponent):
    """
    Converts MARBLE's vertically-resolved inputs from principal components to
    height coordinates.
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


@document_properties
class DiagnosticPrincipalComponentsToHeight(DiagnosticComponent):
    """
    Converts MARBLE's vertically-resolved diagnostic outputs from principal
    components to height coordinates.
    """

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
