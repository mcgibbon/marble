from sympl import TendencyComponent


class LatentHorizontalAdvectiveForcing(TendencyComponent):
    """
    MARBLE component which applies advective forcings in latent space
    (inputs and outputs denormalized principal components) without converting
    to or from the real height coordinate.

    Works by applying an advective tendency already loaded and specified in the
    model state.
    """

    input_properties = {
        'liquid_water_static_energy_components_horizontal_advective_tendency': {
            'dims': ['*', 'sl_latent'],
            'units': 'hr^-1',
            'alias': 'sl_adv',
        },
        'total_water_mixing_ratio_components_horizontal_advective_tendency': {
            'dims': ['*', 'rt_latent'],
            'units': 'hr^-1',
            'alias': 'rt_adv',
        },
    }

    tendency_properties = {
        'liquid_water_static_energy_components': {
            'dims': ['*', 'sl_latent'],
            'units': 'hr^-1',
            'alias': 'sl_adv',
        },
        'total_water_mixing_ratio_components': {
            'dims': ['*', 'rt_latent'],
            'units': 'hr^-1',
            'alias': 'rt_adv',
        },
    }

    diagnostic_properties = {
    }

    def array_call(self, state):
        state.pop('time')
        return state, {}
