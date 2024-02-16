# -*- coding: utf-8 -*-

# Import standard libraries
import os, sys
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import copy
import pandas as pd
import astropy.units as units
import subprocess

# Import our retrieval tools
import coronagraph as cg
import smart
import smarter
from smarter.priors import UniformPrior
import time

import scipy as sp

import astropy.constants as constants

# Use custom pretty plotting
smarter.utils.plot_setup()

# Where we are going to save SMART input/output
place = os.path.join(os.path.abspath("."), "smart_io_miles")

# Make sure there is a directory to save smart outputs
if not os.path.exists(place):
    os.mkdir(place)
    print("Created dir: %s" %place)
else:
    print("Dir already exists: %s" %place)
    
    
# Wavelength bounds for retrieval/data
lammin = 0.7
lammax = 0.8


# Create default SMART model for reflected light spectroscopy
resolution = 5e5
delta_nu = 1e4 / (lammin * resolution)

# Set default SMART parameters
smartin = smart.interface.SmartIn(out_format = 1,                  # IMPORTANT: No transit spectrum
                                  source  = 3,                     # IMPORTANT: Solar and Thermal source functions
                                  grav    = 9.81,                  # Surface gravity (Earth's)
                                  r_AU    = 1.0000,                # Semi-major axis (Earth's)
                                  radius  = 6371.,                 # Planet radius (Earth's in km)
                                  maxwn   = 1e4 / (0.9*lammin),    # Max wavenumber (with padding)
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



# Use the default solar spectrum
solar_spectrum = smart.interface.StellarSpectrum.default_solar_spectrum()

# Load the text file into memory 
solar_spectrum.load_file()


# Use the default composite Earth surface
surface_albedo = smart.interface.Surface.default_earth_composite1()

# Load the text file into memory 
surface_albedo.load_file()


### Construct an initial vertical atmospheric structure and composition   

P0 = 1.013 * 1e5    # Surface pressure
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
d["H2O"] = 3e-3 * np.ones_like(Pnew)
d["CO2"] = 10.**(-4.0) * np.ones_like(Pnew)
d["O2"] = 0.21 * np.ones_like(Pnew)
d["O3"] = 7e-7 * np.ones_like(Pnew)

# Convert dictionary to a pandas.DataFrame
df = pd.DataFrame(d)

# Convert DataFrame to AtmDataFrame for use with smart
atm = smart.interface.AtmDataFrame(df)


# Define smart smiulation object
sim = smart.interface.Smart(tag        = "reflect00",    # Simulation name for i/o
                            smartin    = smartin,        # SMART parameters
                            lblin      = lblin,          # LBLABC Parameters
                            atmosphere = atm,            # Atmospheric TP and composition
                            surface    = surface_albedo, # Planet surface albedo
                            stellar    = solar_spectrum) # Stellar spectrum

# Automatically set SMART and LBLABC exes
sim.set_executables_automatically()

# Tell smart to run in our predetermined "place"
sim.set_run_in_place(place)



## Define an Reflectance Spectrum forward model and run it 

theta_names = ["As", "Temp", "H2O", "CO2", "O2", "O3", "Vsys", "Kp"]
# Define the forward model
fx_highres = smarter.forward_models.FxReflect_HRCCS("isothermal",               # Type of TP profile
                                      #"strato",                   # Type of cloud 
                                       "default", # Default cloud to default smart model
                                       theta_names=theta_names,
                                       smart    = sim,            # Default SMART model
                                       pmin     = 1e-2,           # Min pressure [Pa]
                                       use_existing_pgrid = True, # Use the input atm pressure grid
                                       plvls = plvls,             # Number of pressure levels, if fitting for surface pressure
                                       adapt_mu = True,           # Self-consistent Mean Molec. Weight
                                       fill_gas = "N2",           # Use a VMR filling gas
                                       Vbary = np.zeros(10),
                                       phi = np.linspace(0, np.pi, 10))

# These are our paramaters and their initial values  
As = np.log10(0.05)       # Surface albedo (log)
#Pt_cld = np.log10(0.6e5)  # Cloud top pressure (log Pa)
#dP_cld = np.log10(0.1e5)  # Cloud pressure thickness (log Pa)
#tau_cld = np.log10(13.0)  # Cloud optical depth (log)
f_cld = 0.               # Cloud fraction (linear)
h2o = np.log10(3e-3)      # H2O VMR (log)
co2 = -4.0                # CO2 VMR (log)
o2 = np.log10(0.21)       # O2 VMR (log)
o3 = np.log10(7e-7)       # O3 VMR (log)
Vsys = 0.
Kp = 100. 

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
          Vsys,
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
    
    # RV params
    UniformPrior(-50., 50., theta_name="Vsys"),
    UniformPrior(50., 150., theta_name="Kp"),
    
    
    
]

# Make sure these have the same length and are in the same order!!
assert len(truths) == len(priors)

# Set the names so that we can run fx outside a retrieval 
fx_highres.theta_names = smarter.priors.get_theta_names(priors)


# Run the forward model
xhr, yhr = fx_highres.evaluate(truths)


# Generate some data based on forward model for testing. 
# Kind of backwards but good for testing! 

Rp = sim.smartin.radius      # Already in km for SMART
a = sim.smartin.r_AU * units.AU.in_units(units.km)
scale_factor = (Rp / a)**2.0

def generate_data(fx_highres):
    
    
    if f_cld == 0.:
        Fp_total = fx_highres.rad.pflux
    else:
        Fp_total = f_cld * fx_highres.rad_cld.pflux + (1 - f_cld) * fx_highres.rad_clr.pflux
    
    # calculate total rv of planet at each frame
    RVplanet = Vsys + fx_highres.Vbary + Kp * np.sin(fx_highres.phi)

    # get spline coefficients for planet spectrum
    if f_cld == 0.:
        cs_p = sp.interpolate.splrep(fx_highres.rad.lam, Fp_total, s=0.0)  #spline coeffs for planet, also doing the Rp
    else:
        cs_p = sp.interpolate.splrep(fx_highres.rad_clr.lam, Fp_total, s=0.0)  #spline coeffs for planet, also doing the Rp


    # create series of forward models by Doppler shifting yhr to corresponding Vp of each frame
    Fp_shifted_arr = []
    
    for i in range(len(RVplanet)):
        if f_cld == 0.:
            lam_shifted = fx_highres.rad.lam * (1.0 - RVplanet[i]*1e3/constants.c.value)
        else:
            lam_shifted = fx_highres.rad_clr.lam * (1.0 - RVplanet[i]*1e3/constants.c.value)
        Fp = sp.interpolate.splev(lam_shifted, cs_p, der=0)
        
        Fp_shifted_arr.append(Fp)

    F_shifted_arr = np.array(Fp_shifted_arr)
    
    
    # generate a data lambda grid
    data_wl, data_dwl = cg.noise_routines.construct_lam(lammin, lammax, 100000)
    
    
    Fp_downbinned = []
    if f_cld == 0.:
        Fs_downbinned = cg.downbin_spec(fx_highres.rad.sflux, xhr, data_wl, dlam = data_dwl)
    else:
        Fs_downbinned = cg.downbin_spec(fx_highres.rad_clr.sflux, xhr, data_wl, dlam = data_dwl)
    # interpolate noised up data onto data lambda grid
    for i in range(len(RVplanet)):
        Fp_downbinned.append(cg.downbin_spec(Fp_shifted_arr[i], xhr, data_wl, dlam = data_dwl))

    Fp_downbinned = np.array(Fp_downbinned)
    
    # add noise 
    Fp_downbinned_noisy = Fp_downbinned + 100*np.sqrt(Fp_downbinned)*np.random.randn(Fp_downbinned.shape[0], Fp_downbinned.shape[1])
    data_y = Fp_downbinned_noisy / Fs_downbinned
    
    
       
    data_y *= scale_factor
    
    
    
    return data_wl, data_y

data_x, data_y = generate_data(fx_highres)

# set up a data object for retrieval
data_hrccs = smarter.Data(x = data_x, 
                    y=data_y, 
                    yerr=np.nan*np.ones_like(data_y), 
                    name = "Testing HRCCS", 
                    ylabel = "Planet/Star Flux $(F_p/F_s)$"
                   )

# Finally, run the retrieval 

retrieval = smarter.Retrieval(tag = os.path.join(place, fx_highres.smart.tag),
                              data = data_hrccs,
                              priors = priors,
                              forward = fx_highres,
                              instrument = smarter.instruments.NaiveDownbin(),
                              verbose = True)

#retrieval.lnlike_hrccs(truths)
method = str(sys.argv[1])
start_time = time.time()
if method == "minimize":
    print("Using call_minimize_hrccs()")
    retrieval.call_minimize_hrccs(truths)#, options={"maxiter" : 100, "maxfev":100})
elif method == "minimize_parallel":
    print("Using call_minimize_hrccs()")
    retrieval.call_many_minimize_hrccs(N=None,theta0=truths)#, options={"maxiter" : 2})
elif method == "emcee":
    print("Using emcee")
    def lnprob(theta): return retrieval.lnprob_hrccs(theta)
    retrieval.call_mcmc(lnprob, nsteps = 100, nwalkers = 32, p0 = None,
                        processes = None, cache = True, overwrite = False,
                        nodes = 1, Pool = None)
elif method == "dynesty":
    print("Using dynesty")
    retrieval.call_dynesty_hrccs(maxiter=100)
else:
    print("METHOD NOT SPECIFIED")

print("------------")
print("ELAPSED TIME")
print(round(time.time() - start_time)/60/60, "hr")
print("------------")
