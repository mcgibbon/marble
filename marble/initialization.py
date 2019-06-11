import xarray as xr
import sympl
import numpy as np
from .components import InputHeightToPrincipalComponents, convert_height_to_principal_components

height_to_pc = InputHeightToPrincipalComponents()

def convert_dataarray_to_sympl(dict_of_dataarray):
    for name, array in dict_of_dataarray.items():
        if isinstance(array, xr.DataArray):
            dict_of_dataarray[name] = sympl.DataArray(array)


def get_era5_state(latent_filename, latent=True, i_timestep=0):
    state = {}
    ds = xr.open_dataset(latent_filename)
    state['total_water_mixing_ratio'] = ds['rt'][0, 0, i_timestep, :]
    state['total_water_mixing_ratio'].attrs['units'] = 'kg/kg'
    state['liquid_water_static_energy'] = ds['sl'][0, 0, i_timestep, :]
    state['liquid_water_static_energy'].attrs['units'] = 'J/kg'
    state['height'] = sympl.DataArray(
        np.linspace(0, 3000., 20),
        dims=['z_star'],
        attrs={'units': 'm'},
    )
    state['time'] = sympl.timedelta(0)
    if latent:
        state['vertical_wind'] = ds['w'][0, 0, i_timestep, :]
        state['vertical_wind'].attrs['units'] = 'm/s'
    convert_dataarray_to_sympl(state)
    if latent:
        state = height_to_pc(state)
        state.pop('vertical_wind_components')
    state['time'] = sympl.timedelta(0)
    return state


def get_era5_forcing(latent_filename, i_timestep, latent=True):
    state = {}
    ds = xr.open_dataset(latent_filename)
    state['surface_latent_heat_flux'] = ds['lhf'][0, 0, i_timestep] / 3600.  # divide by one hour to go from J/m^2 to W/m^2
    state['surface_latent_heat_flux'].attrs['units'] = 'W/m^2'
    state['surface_sensible_heat_flux'] = ds['shf'][0, 0, i_timestep] / 3600.
    state['surface_sensible_heat_flux'].attrs['units'] = 'W/m^2'
    state['surface_temperature'] = ds['sst'][0, 0, i_timestep]
    state['surface_temperature'].attrs['units'] = 'degK'
    state['surface_air_pressure'] = ds['p_surface'][0, 0, i_timestep]
    state['surface_air_pressure'].attrs['units'] = 'Pa'
    state['vertical_wind'] = ds['w'][0, 0, i_timestep, :]
    state['vertical_wind'].attrs['units'] = 'm/s'
    state['liquid_water_static_energy_horizontal_advective_tendency'] = ds['sl_adv'][0, 0, i_timestep, :]
    state['total_water_mixing_ratio_horizontal_advective_tendency'] = ds['rt_adv'][0, 0, i_timestep, :]
    state['downwelling_shortwave_radiation_at_3km'] = ds['swdn_tod'][0, 0, i_timestep]
    state['downwelling_shortwave_radiation_at_3km'].attrs['units'] = 'W/m^2'
    state['downwelling_shortwave_radiation_at_top_of_atmosphere'] = ds['swdn_toa'][0, 0, i_timestep]
    state['downwelling_shortwave_radiation_at_top_of_atmosphere'].attrs['units'] = 'W/m^2'
    state['mid_cloud_fraction'] = ds['cldmid'][0, 0, i_timestep]
    state['mid_cloud_fraction'].attrs['units'] = ''
    state['high_cloud_fraction'] = ds['cldhigh'][0, 0, i_timestep]
    state['high_cloud_fraction'].attrs['units'] = ''
    state['total_water_mixing_ratio_at_3km'] = ds['rt'][0, 0, i_timestep, -1]
    state['total_water_mixing_ratio_at_3km'].attrs['units'] = 'kg/kg'
    state['liquid_water_static_energy_at_3km'] = ds['sl'][0, 0, i_timestep, -1]
    state['liquid_water_static_energy_at_3km'].attrs['units'] = 'J/kg'
    state['rain_water_mixing_ratio_at_3km'] = ds['rrain'][0, 0, i_timestep, -1]
    state['rain_water_mixing_ratio_at_3km'].attrs['units'] = 'kg/kg'
    if latent:
        state['total_water_mixing_ratio'] = ds['rt'][0, 0, i_timestep, :]
        state['total_water_mixing_ratio'].attrs['units'] = 'kg/kg'
        state['liquid_water_static_energy'] = ds['sl'][0, 0, i_timestep, :]
        state['liquid_water_static_energy'].attrs['units'] = 'J/kg'
    convert_dataarray_to_sympl(state)
    if latent:
        state['liquid_water_static_energy_components_horizontal_advective_tendency'] = \
            sympl.DataArray(
                convert_height_to_principal_components(
                    state['liquid_water_static_energy_horizontal_advective_tendency'],
                    basis_name='sl', subtract_mean=False
                ), dims=['sl_latent'], attrs={'units': 's^-1'}
            )
        state['total_water_mixing_ratio_components_horizontal_advective_tendency'] = \
            sympl.DataArray(
                convert_height_to_principal_components(
                    state['total_water_mixing_ratio_horizontal_advective_tendency'],
                    basis_name='rt', subtract_mean=False
                ), dims=['rt_latent'], attrs={'units': 's^-1'}
            )
        pc_state = {}
        pc_state.update(state)
        pc_state['height'] = sympl.DataArray(
            np.linspace(0, 3000., 20),
            dims=['z_star'],
            attrs={'units': 'm'},
        )
        pc_state['time'] = sympl.timedelta(0)
        pc_state = height_to_pc(pc_state)
        pc_state.pop('total_water_mixing_ratio_components')
        pc_state.pop('liquid_water_static_energy_components')
        pc_state.pop('height')
        state.update(pc_state)
    return state
