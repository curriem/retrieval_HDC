# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pickle
import astropy.units as unit
import sys
sys.path.append("/gscratch/vsm/mcurr/PACKAGES/")
import high_res_tools as hrt
import math
from scipy.interpolate import splrep, splev
import time
system_metadata = pickle.load(open("../../terrestrial_HRCCS/metadata/high_res_earths_metadata.p", "rb"))



# init params

refl_spec = "../metadata/prxcn_pie_full_5000_20000cm_toa.rad"
tran_spec = "../metadata/prxcn_pie_full_5000_20000cm.trnst"
telescope = "ELT"
star = "prxcn"
env = "pie"
star_env_tag = star + "_" + env

n_orbits = 1000
obs_type = "refl"
band = "full"
molecule = "full"

add_clouds = False

resolution = 100000
contrast = 1e-5

incl = 60
RV_system = 20
RV_bary = 0
dist = 1.3
tellurics = "off"


# system properties
R_star = system_metadata[star_env_tag]["stel_rad"] # solar radii
P_rot_star = 82.6 # [day] from https://arxiv.org/pdf/1608.07834.pdf
P_orb = system_metadata[star_env_tag]["P_s"] # s
R_plan = system_metadata[star_env_tag]["plan_rad"]  # earth radii
P_rot_plan = P_orb # synchronous rotator

a_p = system_metadata[star_env_tag]["a_AU"] # AU
M_s = system_metadata[star_env_tag]["star_mass"]  # solar mass
M_p = 1 # earth mass


if obs_type == "refl":
    
    # observing properties
    length_of_night_hr = 8 * unit.hr
    length_of_cadence_s = 8*60*60 * unit.s
    texp = length_of_cadence_s.value
    
elif obs_type == "tran":

    transit_duration = calc_transit_duration(star_env_tag)

    length_of_cadence_s = transit_duration
    texp = length_of_cadence_s.value

if telescope == "ELT":
    tele_area = 978 # m^2
    tele_diam = 2*np.sqrt(tele_area / np.pi) * unit.m
elif telescope == "TMT":
    tele_diam = 30 * unit.m
elif telescope == "GMT":
    tele_area =  368 # m^2
    tele_diam = 2*np.sqrt(tele_area / np.pi) * unit.m
elif telescope == "ALLELT":
    tele_area = 978 + 707 + 368 # m^2
    tele_diam = 2*np.sqrt(tele_area / np.pi) * unit.m
elif telescope == "VLT":
    tele_diam = 8.2 * unit.m
elif telescope == "ULTIMATE":
    tele_diam = 100 * unit.m
else:
    assert False, "Specify recognized telescope"


if obs_type == "refl":
    
# =============================================================================
#     # realistic phases
#     phases = calculate_optimal_refl_phases("{}_{}".format(star, env),
#                                            band,
#                                            length_of_cadence_s,
#                                            length_of_night_hr,
#                                            quadrature_phase)
# =============================================================================
    phases = np.array([np.pi/2]) * unit.rad    
    phases = phases.to(unit.deg)




elif obs_type == "tran":
    phases = calculate_phases_of_transit(star_env_tag,
                                         band, 
                                         length_of_cadence_s,
                                         dist * unit.pc,
                                         incl * unit.deg)
    
    phases = phases.to(unit.deg)

def calc_transit_duration(star_env):
    orb_period = system_metadata[star_env]["P_s"] * unit.s

    incl_rad = (incl*unit.deg).to(unit.rad)

    r_star = (system_metadata[star_env]["stel_rad"]* unit.solRad).to(unit.km)
    r_plan = (system_metadata[star_env]["plan_rad"] * unit.earthRad).to(unit.km)
    a_p = (system_metadata[star_env]["a_AU"] * unit.AU).to(unit.km)
    b = a_p * np.cos(incl_rad) / r_star

    # alpha is the phase angle difference between ingress and egress
    alpha = 2 * np.arcsin(np.sqrt((r_star + r_plan)**2 - (b*r_star)**2)/ a_p)
    transit_duration = orb_period * alpha / (2 * np.pi)

    return transit_duration


def calculate_phases_of_transit(star, band, cadence, dist, incl):
    
    transit_duration = calc_transit_duration(star)
    
    cadences_per_transit = math.floor(transit_duration.value / cadence.value)
    
    center_of_transit_phase = np.pi
    if cadences_per_transit == 1.:
        transit_phases = np.array([center_of_transit_phase])
    elif cadences_per_transit > 1:
        transit_phases = np.linspace(center_of_transit_phase - alpha.value, center_of_transit_phase + alpha.value, cadences_per_transit)
    else:
        assert False, "Check cadences per transit"
    
    return transit_phases * unit.rad

def calculate_optimal_refl_phases(star, band, cadence, length_of_night_hr, quad):
    orb_period = system_metadata[star]["P_s"] * unit.s

    cadences_per_orbit = orb_period / cadence
    
    rad_per_cadence = (2 * np.pi) / cadences_per_orbit 
    cadences_per_night = length_of_night_hr.to(unit.s) / cadence
    
    
    # receeding quadrature
    rec_quad = np.pi / 2
    rec_starting_rad = rec_quad - (cadences_per_night / 2) * rad_per_cadence
    rec_ending_rad = rec_starting_rad + (cadences_per_night-1) * rad_per_cadence
    
    if cadence == length_of_night_hr.to(unit.s):
        rec_phases = np.array([rec_quad])
    else:
        rec_phases = np.arange(rec_starting_rad, rec_ending_rad+rad_per_cadence.value, rad_per_cadence.value)
    
    
    # approaching quadrature
    app_quad = -np.pi / 2
    app_starting_rad = app_quad - (cadences_per_night / 2) * rad_per_cadence
    app_ending_rad = app_starting_rad + (cadences_per_night-1) * rad_per_cadence
    
    if cadence == length_of_night_hr.to(unit.s):
        app_phases = np.array([app_quad])
    else:
        app_phases = np.arange(app_starting_rad, app_ending_rad+rad_per_cadence.value, rad_per_cadence.value)
    
    
    # concatenate the two
    refl_phases = np.concatenate((rec_phases, app_phases))
    
    return refl_phases * unit.rad






nspec = len(phases)

observation = hrt.pipeline.SimulateObservation(star, env, molecule, band, obs_type, add_clouds=add_clouds, instrument_R=resolution, coronagraph_contrast=contrast)
observation.get_wl_bounds()
observation.load_smart_spectra(path='../metadata/')

# simulate initial observation
observation.run(incl,
                R_star,
                P_rot_star,
                P_orb, R_plan,
                P_rot_plan,
                a_p,
                M_s,
                M_p,
                RV_system,
                RV_bary,
                texp,
                phases,
                dist,
                tele_diam.value)


assert ~np.isnan(np.sum(observation.signal_matrix)), "There is a nan in the simulated data"
rvtot =  np.ones_like(phases.value)*(observation.RV_sys + observation.RV_bary )



    
signal_per_night = observation.signal_matrix
# take care of any values less than zero
signal_less_zero = (signal_per_night < 0)
signal_per_night[signal_less_zero] = 0.

if obs_type == "tran":

    #signal_oot_per_night = observation.signal_oot_matrix
    signal_per_night = observation.signal_matrix
    signal_oot_per_night = observation.signal_oot_matrix
    signal_oot_less_zero = (signal_oot_per_night < 0)
    signal_oot_per_night[signal_oot_less_zero] = 0.
    

elif obs_type == "refl":

    signal_only_star_per_night = observation.signal_only_star
    signal_only_speckle_per_night = observation.signal_only_speckle



    signal_only_star_less_zero = (signal_only_speckle_per_night < 0)
    signal_only_speckle_per_night[signal_only_star_less_zero] = 0.
    
    signal_only_speckle_per_night_no_T = observation.signal_only_speckle_no_T
    


read_noise_per_night = observation.read_noise[observation.data_naninds]
dark_noise_per_night = observation.dark_noise[observation.data_naninds]
sky_noise_per_night = observation.sky_noise[observation.data_naninds]





    
if obs_type == "tran":
    signal = signal_per_night * n_orbits
    noise = sky_noise_per_night**2*n_orbits + dark_noise_per_night**2*n_orbits + read_noise_per_night**2*n_orbits

    signal_oot = signal_oot_per_night * n_orbits
    
    #spec = np.random.poisson(signal+noise)
    spec = np.random.poisson(signal)
    
    spec_oot = signal_oot # + noise
    obs_spec = spec / spec_oot
    obs_spec_noiseless = signal / signal_oot

elif obs_type == "refl" :

    
    sky_detector_background = dark_noise_per_night**2*n_orbits + read_noise_per_night**2*n_orbits + sky_noise_per_night**2*n_orbits
    
    #total_photons_on_detector = signal_per_night * n_orbits + signal_only_speckle_per_night * n_orbits + sky_detector_background
    
    #model_star_tellurics_detector = signal_only_speckle_per_night * n_orbits + sky_detector_background
    
    signal = signal_per_night * n_orbits
    noise = np.sqrt(signal + 2 * signal_only_speckle_per_night * n_orbits + sky_detector_background)
    
    signal_no_T = observation.signal_matrix_no_T * n_orbits
    noise_no_T = np.sqrt(signal_no_T + 2 * signal_only_speckle_per_night_no_T * n_orbits + sky_detector_background)
    

    

    
    
    rand_nums = np.random.randn(noise.shape[0], noise.shape[1], noise.shape[2])
    data = signal + rand_nums * noise
    
    data_no_T = signal_no_T + rand_nums * noise_no_T
    
    np.save("../data/prxcn_pie_data_noiseless.npy", signal_no_T)
    np.save("../data/prxcn_pie_wavelength.npy", observation.instrument_lam)
    
    np.save("../data/prxcn_pie_data.npy", data)
    
    np.save("../data/prxcn_pie_data_no_T.npy",  data_no_T)
    