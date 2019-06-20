import marble as mb
import initialization as init
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
        'cld': 'cloud_fraction',
    }
)

# Initialize components
timestep = timedelta(hours=1)
marble_tendency_component = mb.LatentMarble()
inputs_to_height = mb.InputPrincipalComponentsToHeight()
diagnostics_to_height = mb.DiagnosticPrincipalComponentsToHeight()
advection = mb.LatentHorizontalAdvectiveForcing()
# First order Adams Bashforth is the same as Forward Euler
stepper = sp.AdamsBashforth([marble_tendency_component, advection], order=1)
model_monitor = mb.ColumnStore()
reference_monitor = mb.ColumnStore()

state = init.get_era5_state(init.column_filename, latent=True)

while state['time'] < timedelta(days=30):
    i_hour = int(state['time'].total_seconds() / 3600.)
    state.update(init.get_era5_forcing(init.column_filename, latent=True, i_timestep=i_hour))
    diagnostics, next_state = stepper(state, timestep=timestep)
    state.update(diagnostics)
    # Convert to height coordinates and store in monitors for later analysis
    z_dict = inputs_to_height(state)
    z_dict.update(diagnostics_to_height(state))
    model_monitor.store(z_dict)
    reference_z_dict = init.get_era5_state(init.column_filename, latent=False, i_timestep=i_hour)
    reference_z_dict.update(init.get_era5_diagnostics(init.column_filename, i_timestep=i_hour))
    reference_monitor.store(reference_z_dict)
    # Increment state and timestep
    state.update(next_state)
    state['time'] += timestep


# Integration is over, here analysis code starts

def get_min_max(array1, array2):
    """
    Returns the minimum and maximum values from a pair of numpy arrays.
    """
    vmin = min(np.min(array1), np.min(array2))
    vmax = max(np.max(array1), np.max(array2))
    return vmin, vmax


Cpd = sp.get_constant('heat_capacity_of_dry_air_at_constant_pressure', units='J/kg/degK')

time = np.arange(0, model_monitor['sl'].shape[0]) / 24.
z = state['height'].values / 1000.
T, Z = np.broadcast_arrays(time[:, None], z[None, :])

marble_sl = model_monitor['sl'] / Cpd
reference_sl = reference_monitor['sl'] / Cpd
vmin, vmax = get_min_max(marble_sl, reference_sl)


fig, ax = plt.subplots(3, 2, figsize=(8, 6))

im = ax[0, 0].pcolormesh(T, Z, marble_sl, vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax[0, 0])
ax[0, 0].set_title('MARBLE $s_l$ (K)')

im = ax[0, 1].pcolormesh(T, Z, reference_sl, vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax[0, 1])
ax[0, 1].set_title('ERA5 $s_l$ (K)')

vmin, vmax = get_min_max(model_monitor['rt'], reference_monitor['rt'])
vmin *= 1e3
vmax *= 1e3
im = ax[1, 0].pcolormesh(T, Z, 1e3 * model_monitor['rt'], vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax[1, 0])
ax[1, 0].set_title('MARBLE $r_t$ (g/kg)')

im = ax[1, 1].pcolormesh(T, Z, 1e3 * reference_monitor['rt'], vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax[1, 1])
ax[1, 1].set_title('ERA5 $r_t$ (g/kg)')

im = ax[2, 0].pcolormesh(T, Z, model_monitor['cld'], vmin=0., vmax=1.)
plt.colorbar(im, ax=ax[2, 0])
ax[2, 0].set_title('MARBLE cloud fraction')
im = ax[2, 1].pcolormesh(T, Z, reference_monitor['cld'], vmin=0., vmax=1.)
plt.colorbar(im, ax=ax[2, 1])
ax[2, 1].set_title('ERA5 cloud fraction')

ax[-1, 0].set_xlabel('Time elapsed (days)')
ax[-1, 1].set_xlabel('Time elapsed (days)')

plt.tight_layout()
plt.show()
