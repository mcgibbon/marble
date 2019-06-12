"""
In this example, we take an initial thermodynamic state and project it onto
principal components and back, then compare this with the true vertical
profiles.
"""
import marble as mb
import sympl as sp
import matplotlib.pyplot as plt

mb.register_alias_dict(
    {
        'sl': 'liquid_water_static_energy',
        'rt': 'total_water_mixing_ratio',
        'z': 'height',
    }
)

components_to_height = mb.InputPrincipalComponentsToHeight()
height_to_components = mb.InputHeightToPrincipalComponents()

test_era5_filename = '/home/twine/data/era5/era5-interp-2016.nc'

state = mb.get_era5_state(test_era5_filename, latent=False)
state.update(mb.get_era5_forcing(test_era5_filename, i_timestep=0, latent=False))
component_state = height_to_components(state)
component_state.update(state)  # carry over things like height profile and model time
reconstructed_state = components_to_height(component_state)

state = mb.AliasDict(state)
reconstructed_state = mb.AliasDict(reconstructed_state)

Cpd = sp.get_constant('heat_capacity_of_dry_air_at_constant_pressure', units='J/kg/degK')

fig, ax = plt.subplots(1, 2)
ax[0].plot(state['sl'] / Cpd, state['z'], label='truth')
ax[0].plot(reconstructed_state['sl'] / Cpd, state['z'], label='reconstructed')
ax[0].legend(loc='best')
ax[0].set_title('Liquid Water Static Energy (K)')
ax[1].plot(state['rt'], state['z'], label='truth')
ax[1].plot(reconstructed_state['rt'], state['z'], label='reconstructed')
ax[1].set_title('Total Water Mixing Ratio (kg/kg)')
plt.tight_layout()
plt.show()
