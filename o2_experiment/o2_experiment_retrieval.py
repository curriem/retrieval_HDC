# -*- coding: utf-8 -*-

# Import standard libraries
import os, sys
import numpy as np
import scipy as sp
import pandas as pd

# Import our retrieval tools
import coronagraph as cg
import smart
import smarter
from smarter.priors import UniformPrior

import astropy.units as unit
from scipy.interpolate import interp1d
import astropy.constants as constants
import astropy.io


case = sys.argv[1]

if case == "fo2_01":
    fo2_forward = 0.01
    fch4_forward = 1e-3
    fh2o_forward = 3e-3
    fo3_forward = 7e-7
    fco2_forward = 0.1
    P0 = 1.013 * 1e5    # Surface pressure
elif case == "fo2_05":
    fo2_forward = 0.05
    fch4_forward = 1e-3
    fh2o_forward = 3e-3
    fo3_forward = 7e-7
    fco2_forward = 0.1
    P0 = 1.013 * 1e5    # Surface pressure
elif case == "fo2_21" or case == "fo2_21_noiseless":
    fo2_forward = 0.21
    fch4_forward = 1e-3
    fh2o_forward = 3e-3
    fo3_forward = 7e-7
    fco2_forward = 0.1
    P0 = 1.013 * 1e5    # Surface pressure
elif case == "fo2_50":
    fo2_forward = 0.50
    fch4_forward = 1e-3
    fh2o_forward = 3e-3
    fo3_forward = 7e-7
    fco2_forward = 0.1
    P0 = 1.013 * 1e5    # Surface pressure
elif case == "o2_10bar":
    fo2_forward = 0.95
    fch4_forward = 1e-12
    fh2o_forward = 3e-3
    fo3_forward = 7e-7
    fco2_forward = 5e-3
    P0 = 9.60200e5    # Surface pressure
else:
    assert False, "Case not found/specified"
    
if case == "fo2_21_noiseless":
    noise_switch = 0.
else:
    noise_switch = 1.



# Tell smart to run in our predetermined "place"
# Where we are going to save SMART input/output
place = os.path.join(os.path.abspath("."), "smart_io_{}".format(case))

# Make sure there is a directory to save smart outputs
if not os.path.exists(place):
    os.mkdir(place)
    print("Created dir: %s" %place)
else:
    print("Dir already exists: %s" %place)

def make_data(spec_fl, lammin, lammax, 
              phases, radstar, dist, r_AU, tele_diam,
              RV_sys, RV_bary, Kp,
              throughput=0.1,seed=None):
    
    ##################################
    ############ STEP 1 ##############
    ##################################
    
    # read smart file to make data
    prxcn_rad = smart.readsmart.Rad(spec_fl)

    # reverse order to ascending lambda
    sflux = prxcn_rad.sflux[::-1]
    pflux = prxcn_rad.pflux[::-1]
    lam = prxcn_rad.lam[::-1]

    # cut to lammin, lammax size
    m = (lam > lammin) & (lam < lammax)
    sflux = sflux[m]
    pflux = pflux[m]
    lam = lam[m]

    # define planet albedo
    Ahr = pflux/sflux

    # calculate exposure time per phase
    texp_per_phase = texp_total / len(phases)


    # Define ELT-like Telescope
    telescope = cg.Telescope(lammin = np.min(lam),    # Wavelength minimum
                             lammax = np.max(lam),    # Wavelength maximum
                             R = 100000,            # Spectral resolution
                             Tput = throughput,        # Throughput
                             D = tele_diam,           # Telescope mirror diameter [m]
                             Tsys = 275.0,       # Telescope mirror temperature [K]
                             C = 1e-5,
                             IWA=0.
                             # DEFINE OTHER PARAMS
                            )


    star = cg.Star(Teff = 2992.,               # Stellar effective temperature [K]
                   Rs = radstar)                   # Stellar radius [solar radii]     


    planet = cg.Planet(name="PCb",
                       d = dist,                 # Distance [pc]
                       a = r_AU,                  # Semi-major axis [AU]
                       alpha = 0. 
                      )

    # Use coronagraph model to create spectrum
    cn = cg.CoronagraphNoise(telescope=telescope,
                             planet=planet,
                             star=star,
                             texp = texp_per_phase, # hr
                            )

    cn.run_count_rates(Ahr, lam, sflux)


    # get tellurics
    # set telluric resolution to be retrieved
    
    tellurics_fits = astropy.io.fits.open("../sky/skycalc_500_2000nm.fits")
    
    tel_lam = tellurics_fits[1].data["LAM"] 
    tel_trans = tellurics_fits[1].data["TRANS"]
    tel_flux = tellurics_fits[1].data["FLUX"]
    
    omega_sky = np.pi*(tel_lam*1e-6/tele_diam*180.*3600./np.pi)**2. # arcsec2
            
    sky_background = tel_flux * omega_sky # units are photons/s/m2/um
    
    cthe_hr = sky_background * np.pi * (tele_diam/2)**2 * throughput # units are photons/s/um 


    # interpolate tellurics onto data wl grid
    f_trans = interp1d(tel_lam, tel_trans, fill_value="extrapolate")
    tellurics = f_trans(cn.lam[:-1])
    #tellurics = np.ones_like(tellurics)

    f_tflux = interp1d(tel_lam, cthe_hr, fill_value="extrapolate")
    cthe = f_tflux(cn.lam)


    wl = cn.lam[:-1]
    cp = cn.cp[:-1] * texp_per_phase
    csp = cn.csp[:-1] * texp_per_phase

    step1_arr = cp
    
    
    ##################################
    ############ STEP 2 ##############
    ##################################
    
    cs_signal = sp.interpolate.splrep(wl, cp, s=0.0)  #spline coeffs for planet, also doing the Rp
    cs_speck = sp.interpolate.splrep(wl, csp, s=0.0)


    signal_arr = []
    speck_arr = []
    for phase in phases:

        RVplanet = RV_sys + RV_bary + Kp * np.sin((phase*unit.deg).to(unit.rad))

        lam_shifted = wl * (1.0 - RVplanet*1e3/constants.c.value)

        signal_shifted = sp.interpolate.splev(lam_shifted, cs_signal, der=0)    
        speck_shifted = sp.interpolate.splev(lam_shifted, cs_speck, der=0)    

        signal_arr.append(signal_shifted)
        speck_arr.append(speck_shifted)

    signal_arr = np.array(signal_arr)
    speck_arr = np.array(speck_arr)


    buffer = 100

    signal_arr = signal_arr[:, buffer:-buffer]
    speck_arr = speck_arr[:, buffer:-buffer]
    wl_arr = wl[buffer:-buffer]

    step2_matrix = signal_arr
    
    ##################################
    ############ STEP 3 ##############
    ##################################
    
    step3_matrix = step2_matrix * tellurics[buffer:-buffer]

    
    ##################################
    ############ STEP 4 ##############
    ##################################
    
    np.random.seed(seed)
    noise = np.random.randn(step3_matrix.shape[0], step3_matrix.shape[1]) * np.sqrt(step2_matrix * tellurics[buffer:-buffer] + speck_arr * tellurics[buffer:-buffer])

    step4_matrix = step2_matrix * tellurics[buffer:-buffer] + noise_switch * noise

    
    ##################################
    ############ STEP 5 ##############
    ##################################
    
    step5_matrix = np.empty_like(step4_matrix)
    for i in range(len(phases)):
        spec = step4_matrix[i]

        largest_vals = np.sort(spec)[::-1][:300]
        med_largest_vals = np.median(largest_vals)
        step5_matrix[i] = spec / med_largest_vals

    
    ##################################
    ############ STEP 6 ##############
    ##################################
    
    avg_spec = np.mean(step5_matrix, axis=0)

    step6_matrix = np.empty_like(step5_matrix)

    for i in range(len(phases)):
        spec = step5_matrix[i]

        coef = np.polyfit(avg_spec, spec,2)
        fit = coef[0]*avg_spec**2 + coef[1]*avg_spec + coef[2]

        spec_new = spec / fit 

        step6_matrix[i] = spec_new

    
    ##################################
    ############ STEP 7 ##############
    ##################################
    
    step7_matrix = np.empty_like(step6_matrix)
    phases_rad = (phases*unit.deg).to(unit.rad).value

    for j in range(len(avg_spec)):
        col = step6_matrix[:, j]


        coef = np.polyfit(phases_rad, col,2)
        fit = coef[0]*phases_rad**2 + coef[1]*phases_rad + coef[2]
        col_new = col / fit 


        step7_matrix[:, j] = col_new
    
    ##################################
    ############ STEP 8 ##############
    ##################################
    
    step8_matrix = np.empty_like(step7_matrix)

    std_matrix = np.std(step7_matrix)

    counter = 0
    for j in range(len(avg_spec)):
        col = step7_matrix[:, j]

        #print(std_matrix, np.std(col))
        if np.std(col) > 3*std_matrix:
            new_col = np.nan*np.ones_like(col)
            counter += 1
        else:
            new_col = col

        step8_matrix[:, j] = new_col
        
        
    data_for_retrieval = np.copy(step8_matrix)
    lam_for_retrieval = np.copy(wl_arr)
    
    return lam_for_retrieval, data_for_retrieval


spec_fl = "/gscratch/vsm/mcurr/PROJECTS/retrieval_HDC/o2_experiment/spectra/smart_io/prxcn_21percent_5000_20000cm_toa.rad"
phases = np.arange(30, 61, 1.)

texp_total = 1000 * 60 * 60 

radstar = 0.1542
r_AU = 0.04060742579931119
dist = 1.3 # pc

lammin = 0.5
lammax = 2.0

RV_sys = -22.
RV_bary = 0.
Kp = 49.07319034574975 # assumes a 60 deg inclination
#Kp = 100
tele_diam = 39.
throughput = 0.1

seed = 55

# get data
lam_for_retrieval, data_for_retrieval  = make_data(spec_fl, lammin, lammax, 
                                                   phases, radstar, dist, r_AU,
                                                   tele_diam, RV_sys, RV_bary, Kp, seed=seed)



# set up a data object for retrieval
data_hrccs = smarter.Data(x = lam_for_retrieval, 
                    y=data_for_retrieval, 
                    yerr=np.nan, 
                    name = "Testing HRCCS", 
                    ylabel = "Planet/Star Flux $(F_p/F_s)$"
                   )




# Create default SMART model for reflected light spectroscopy
resolution = 5e5
delta_nu = 1e4 / (lammin * resolution)

# Set default SMART parameters
smartin = smart.interface.SmartIn(out_format = 1,                  # IMPORTANT: No transit spectrum
                                  source  = 3,                     # IMPORTANT: Solar and Thermal source functions
                                  grav    = 9.81,                  # Surface gravity (Earth's)
                                  r_AU    = r_AU,                # Semi-major axis (Earth's)
                                  radius  = 6371.,                 # Planet radius (Earth's in km)
                                  maxwn   = 1e4 / (0.95*lammin),    # Max wavenumber (with padding)
                                  minwn   = 1e4 / (1.05*lammax),   # Min wavenumber (with padding)
                                  err_tau = 0.35,                  # Optical depth binning tolerance
                                  err_alb = 0.35,                  # Albedo binning tol
                                  err_g   = 0.35,                  # <cos> binning tol
                                  err_pi0 = 0.35,                  # pi0 binning tol
                                  FWHM = delta_nu, 
                                  sample_res = delta_nu
                                 )

# Set default LBLABC parameters (using the same values as smartin)
lblin = smart.interface.LblIn(radius = smartin.radius,
                              grav   = smartin.grav,
                              maxwn  = smartin.maxwn,
                              minwn  = smartin.minwn)


sol_fl = "../data/hazmat_prxcn_xhires.txt"
sol_skip = 0

stellar = smart.interface.StellarSpectrum(sol_fl, sol_skip, iyunit=2, ixunit=1)                                                                                                                                   
     

# Use the default composite Earth surface
surface_albedo = smart.interface.Surface.default_earth_composite1()

# Load the text file into memory 
surface_albedo.load_file()


### Construct an initial vertical atmospheric structure and composition   

P1 = 0.1 * 1e5      # Tropopause pressure
T0 = 250.0          # Isothermal temperature
plvls = 30

# Create a standard pressure grid
Pnew = smart.analysis.pgrid(P0, 1e-7*1e5, plvls = plvls)

# Create TP
Tnew = T0 * np.ones_like(Pnew)

# Set atmosphere parameters with a Python dictionary
d = {}
d["Press"] = Pnew
d["Temp"] = Tnew
d["H2O"] = fh2o_forward * np.ones_like(Pnew)
d["CO2"] = fco2_forward * np.ones_like(Pnew)
d["O2"] = fo2_forward * np.ones_like(Pnew)
d["O3"] = fo3_forward * np.ones_like(Pnew)
d["CH4"] = fch4_forward * np.ones_like(Pnew)


# Convert dictionary to a pandas.DataFrame
df = pd.DataFrame(d)

# Convert DataFrame to AtmDataFrame for use with smart
atm = smart.interface.AtmDataFrame(df)


# Define smart smiulation object
sim = smart.interface.Smart(tag        = case,    # Simulation name for i/o
                            smartin    = smartin,        # SMART parameters
                            lblin      = lblin,          # LBLABC Parameters
                            atmosphere = atm,            # Atmospheric TP and composition
                            surface    = surface_albedo, # Planet surface albedo
                            stellar    = stellar) # Stellar spectrum

# Automatically set SMART and LBLABC exes
sim.set_executables_automatically()


sim.set_run_in_place(place)


sim.smartin.radstar = radstar     # times Solar radius



## Define an Reflectance Spectrum forward model and run it 

theta_names = ["As", "Temp", "H2O", "CO2", "O2", "O3", "CH4", "Vsys", "Kp"]
# Define the forward model
fx_highres = smarter.forward_models.FxReflect_HRCCS("isothermal",               # Type of TP profile
                                      "default",                   # Type of cloud 
                                       theta_names=theta_names,
                                       smart    = sim,            # Default SMART model
                                       pmin     = 1e-2,           # Min pressure [Pa]
                                       use_existing_pgrid = True, # Use the input atm pressure grid
                                       plvls = plvls,             # Number of pressure levels, if fitting for surface pressure
                                       adapt_mu = True,           # Self-consistent Mean Molec. Weight
                                       fill_gas = "N2",           # Use a VMR filling gas
                                       Vbary = np.zeros(len(phases)),
                                       phi = (phases*unit.deg).to(unit.rad))

# These are our paramaters and their initial values  
As = np.log10(0.05)       # Surface albedo (log)
#Pt_cld = np.log10(0.6e5)  # Cloud top pressure (log Pa)
#dP_cld = np.log10(0.1e5)  # Cloud pressure thickness (log Pa)
#tau_cld = np.log10(13.0)  # Cloud optical depth (log)
#f_cld = 0.5               # Cloud fraction (linear)
h2o = np.log10(fh2o_forward)      # H2O VMR (log)
co2 = np.log10(fco2_forward)                # CO2 VMR (log)
o2 = np.log10(fo2_forward)       # O2 VMR (log)
o3 = np.log10(fo3_forward)       # O3 VMR (log)
ch4 = np.log10(fch4_forward)
#Vsys = 0.
#Kp = 49.07319034574975 # km/s

# These are the true values
truths = [As, 
          T0, 
          #Pt_cld, 
          #dP_cld, 
          #tau_cld, 
          #f_cld, 
          h2o, 
          co2, 
          o2, 
          o3,
          ch4,
          RV_sys,
          Kp
         ]

# Define a list of priors
priors = [
    
    # Surface Albedo
    UniformPrior(-2.0, 0.0, theta_name="As"), 
    
    # TP controlling Parameters
    UniformPrior(100.0, 400.0, theta_name="Temp"), 
    
    # Cloud parameters
    #UniformPrior(3.0, 7.0, theta_name = "Pt_cld"), 
    #UniformPrior(2.0, 7.0, theta_name = "dP_cld"), 
    #UniformPrior(-2.0, 2.0, theta_name = "tau_cld"), 
    #UniformPrior(0.0, 1.0, theta_name = "f_cld"), 
    
    # Gas Abundances 
    UniformPrior(-12.0, 0.0, theta_name="H2O"),
    UniformPrior(-12.0, 0.0, theta_name="CO2"),
    UniformPrior(-12.0, 0.0, theta_name="O2"),
    UniformPrior(-12.0, 0.0, theta_name="O3"),
    UniformPrior(-12.0, 0.0, theta_name="CH4"),

    
    # RV params
    UniformPrior(-50., 50., theta_name="Vsys"),
    UniformPrior(0., 100., theta_name="Kp"),
    
]

# Make sure these have the same length and are in the same order!!
assert len(truths) == len(priors)

# Set the names so that we can run fx outside a retrieval 
fx_highres.theta_names = smarter.priors.get_theta_names(priors)


# Run the forward model
xhr, yhr = fx_highres.evaluate(truths)


retrieval = smarter.Retrieval(tag = os.path.join(place, fx_highres.smart.tag),
                              data = data_hrccs,
                              priors = priors,
                              forward = fx_highres,
                              instrument = smarter.instruments.NaiveDownbin(),
                              verbose = True)

retrieval.call_dynesty_hrccs(processes=40)



