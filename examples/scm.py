import marble as mb
import sympl as sp
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np

mb.register_alias_dict(
    {
        'sl': 'liquid_water_static_energy',
        'sl_latent': 'liquid_water_static_energy_components',
        'rt': 'total_water_mixing_ratio',
        'rt_latent': 'total_water_mixing_ratio_components',
        'z': 'height',
        'sl_adv_latent': 'liquid_water_static_energy_components_horizontal_advective_tendency',
        'rt_adv_latent': 'total_water_mixing_ratio_components_horizontal_advective_tendency',
    }
)

test_era5_filename = '/home/twine/data/era5/era5-interp-2016.nc'

state = mb.get_era5_state(test_era5_filename, latent=True)
timestep = timedelta(hours=1)
marble_tendency_component = mb.LatentMarble()
components_to_height = mb.InputPrincipalComponentsToHeight()
advection = mb.LatentHorizontalAdvectiveForcing()
stepper = sp.AdamsBashforth([marble_tendency_component, advection], order=1)

data_lists = {}
reference_data_lists = {}

i_hour = 0
while state['time'] < timedelta(days=3):
    i_hour = int(state['time'].total_seconds() / 3600.)
    state.update(mb.get_era5_forcing(test_era5_filename, latent=True, i_timestep=i_hour))
    z_dict = components_to_height(state)
    reference_z_dict = mb.get_era5_state(test_era5_filename, latent=False, i_timestep=i_hour)
    for name in ('liquid_water_static_energy', 'total_water_mixing_ratio'):
        if isinstance(z_dict[name], sp.DataArray):
            data_lists[name] = data_lists.get(name, [])
            data_lists[name].append(z_dict[name].values[None, :])
            reference_data_lists[name] = reference_data_lists.get(name, [])
            reference_data_lists[name].append(reference_z_dict[name].values[None, :])
    diagnostics, next_state = stepper(state, timestep=timestep)
    state.update(next_state)
    state['time'] += timestep

timeseries_dict = {}
reference_dict = {}
for name, list in data_lists.items():
    timeseries_dict[name] = np.concatenate(list, axis=0)
    reference_dict[name] = np.concatenate(reference_data_lists[name], axis=0)

state.update(components_to_height(state))

state = mb.AliasDict(state)
timeseries_dict = mb.AliasDict(timeseries_dict)
reference_dict = mb.AliasDict(reference_dict)

plt.figure()
plt.pcolormesh(timeseries_dict['sl'].T)
plt.figure()
plt.pcolormesh(reference_dict['sl'].T)
plt.show()

Cpd = sp.get_constant('heat_capacity_of_dry_air_at_constant_pressure', units='J/kg/degK')

reference = mb.get_era5_state(test_era5_filename, latent=False, i_timestep=i_hour)
reference = mb.AliasDict(reference)

fig, ax = plt.subplots(1, 2)
ax[0].plot(state['sl'].values / Cpd, state['z'].values, label='MARBLE')
ax[0].plot(reference['sl'].values / Cpd, state['z'].values, label='ERA5')
ax[1].plot(state['rt'].values, state['z'].values)
ax[1].plot(reference['rt'].values, state['z'].values)
ax[0].legend(loc='best')
# plt.show()

plt.show()
