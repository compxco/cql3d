# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 19:31:08 2013
Purpose: read file *.nc produced by CQL3D,
plot toroidal electric field, for each time step;
all iterations (Ampere-Faraday eqns) are plotted in one figure for a given t.
@author: YuP
"""

from numpy import *
from mpl_toolkits.mplot3d import Axes3D

from pylab import *
from matplotlib import rc 
from matplotlib.pyplot import cm,figure,axes,plot,xlabel,ylabel,title,savefig,show

import os
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time
import pylab as pylab
import scipy.io.netcdf as nc
#import pandas as pd # to merge datasets ?
#matplotlib.interactive(True) # no plots on screen
matplotlib.interactive(False) # with plots on screen
print 'matplotlib version:', matplotlib.__version__

# Detect which OS you are using; By default, it is 'windows'
ios='windows' #Will be changed to 'linux', if Python detects it, see below. 
# Some of functionality (such as using Latex) depends on OS
if sys.platform.startswith('win32'):
    print 'OS is windows'
    ios='windows'    
elif sys.platform.startswith('linux'):
    print 'OS is linux'
    ios='linux'
    #Render with externally installed LateX:
    matplotlib.rc('text', usetex = True) #ONLY FOR LINUX !does not work for windows
# Other possible values for sys.platform.startswith(): 
#     'darwin' for MacOS, 'cygwin' for Windows/Cygwin, 'aix' for AIX

#-----------------------------------------------
# NetCDF issues: machine-dependent
# Try netcdf=4 to envoke netCDF4,
# Or try netcdf=2 to work with older netCDF.
netcdf=4
#-----------------------------------------------
if netcdf==4: from netCDF4 import Dataset # YuP
#-----------------------------------------------
e0 = time.time()  # elapsed time since the epoch
c0 = time.clock() # total cpu time spent in the script so far


# Constants
pi=3.14159265358979
clight= 2.99792458e10   # speed of light [cm/s]
charge= 4.8032e-10      # e-charge [cgs]
e     = 4.8032e-10      # e-charge [cgs]
p_mass= 1.67262158e-24  # proton mass    [gram]
proton= 1.67262158e-24  #    [gramm]
e_mass= 9.1095e-28      #    [gramm]
ergtkev=1.6022e-09


#For plots: set fonts and line thicknesses  ------------------------------
fnt  = 19 #16 #20 #12     # font size for axis numbers (see 'params=' below) 
linw = 1.0    # LineWidth for plots
view_azim=-110 #-140 #-135 # degrees  For Axes3D() mesh plots over (rho,time)
view_elev=55 #30 #+72 # degrees  For Axes3D() mesh plots over (rho,time)

imesh=0 # 0 for contour plots; 1 for mesh (over rho,time 2D grid)
Ncont=50 # Number of contour levels, in case of imesh=0
nt_plot=200 #40 #80 #40 #20 #501 # When nt is too large (say, 500 time steps), 
           # it is better to omit some points.
           # Specify nt_plot for the approximate number of steps for plotting.
           # The stride in time index is set below as
           # ntstride= np.floor(nt/nt_plot) #  floor()=nearest-lower integer.
           #For example, (nt/20) means: leave only ~20 time points for plotting.
           # If you want to plot ALL time steps, simply set nt_plot to a very
           # large value (nstop, or larger); then ntstride will be 1.
t_low_lim=0. #810 #[ms] Lower limit for plots of mesh over (t,rho).
           #        Normally, it should be 0, but setting to a larger value
           #         allows zooming-in.

each_step=0 #Set to 1 to make&save Array(rho) plots for each time instant. 
i_ko=1 # Set to 1, to plot KO source  (0-noplots)
i_ra=1 # Set to 1, to plot Runaway-related plots
i_ampf=0
i_dbb=1  #(deltaB/B)^2 from data files (from NIMROD)
i_plasma_profiles=1 # to plot plasma profiles (density, T, <energy>, Zeff)
ksp=0 # those plasma profiles are made for ksp species (in python counting)

#Specify time steps for which plots like E(rho) will be shown, at these n-steps:
#it_select=    [ 0, 14, 48, 80, 114,148, 180] #, 22, 25, 28, 31] 
# For NIMROD-CQL3D 2ms runs:
#it_select=    [ 0, 10, 20, 30, 40, 50,  60, 70, 80, 90, 100]
#col_select_it=['r','b','g','m','c','k', 'r','b','g','m','c'] # color for each it_select
# For NIMROD-CQL3D 6ms runs (720 steps):
#it_select=    [  0, 100,150,200,250,300,320,340,360,380,400,420,440,460,480,500,600,700] 
# For NIMROD-CQL3D 8ms runs (1270 steps):
it_select=    [  0, 200,300,400,500,600,700,800,900,1000,1050,1100,1150,1200,1250,1300,1350,1400] 
col_select_it=[ 'r','b','g','m','c','k','r','b','g','m', 'c', 'k', 'r', 'b', 'g', 'm', 'c', 'k']

# For combined-files, specify selected time slices to show:
#t_combined_select=   [1.14, 1.17,1.19,1.21,1.23,1.25,1.27,1.29] # [sec]
#col_t_combined_select=['g', 'm', 'c', 'k', 'r' ,'b', 'g', 'm' ] #,'c']
#t_combined_select=[0.90,0.92,0.94,1.000,1.050,1.100,1.160] # [sec]
t_combined_select=[1.160, 1.180, 1.200, 1.220, 1.240] # [sec]
#t_combined_select=   [1.00, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0, 2.4, 2.8, 3.0] # [sec]
#t_combined_select=   [0.010,0.015,0.020,0.025,0.030,0.035,0.040] # [sec]
col_t_combined_select=['r','b','g','m','c','k','r','b','g','m','c','k','r','b','g','m'] 

#Similarly, Plots of a func.vs.t will be made for these     
# radial indexes (up to 12 radial points)
ir_select=  [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 30] #for lrz=22
#ir_select=  [ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36] #for lrz=40
#ir_select= [ 0,  8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 120] 
col_select=['r','b','g','m','c','k','r','b','g','m','c','k','r']

# For plots of 'favr_thet0' (distr func averaged over pitch, at all time steps)
i_favr_thet0=1 # To plot favr_thet0 (run CQL3D with netcdfshort='long_jp')
f_plot_mx=2.e-10 # 2.e-5 #2.e-10 # Max limit in plots of distr.functions.
f_plot_range=1.e16 # Range for plots, i.e. f_plot_mn= f_plot_mx/f_plot_range.

# For plotting the curve A*exp(-lam^2 *t / (4pi*sigma/c^2)) 
# Specify:
a=    70. # cm  # Compare with printout of rgeomp (see below)
#lam=  2.4048/a  # 2.4/a means that J0(r*lam) gets to zero at r=a
lam= 5.52008/a # second root of J0(a*lam)=0
#lam= 8.6537/a # 3rd root of J0(a*lam)=0
#lam= 11.79153444/a # 4th root of J0(a*lam)=0
#lam= 14.93091771/a # 5th root of J0(a*lam)=0
resist_phi_0= 3.123e-16 #8.6E-16 # resistivity (cgs) at r=0 
#(printed during CQL3D run as resist_phi(lr) )
resist_phi_a= 3.215e-16 #9.5E-16 # resistivity (cgs) at plasma edge (r=a)
# Compare to printout of RESISTIVITY <E_phi/R>/<j_phi/R> (see below)
#-------------------------------------------------------------

# Optionally: Calculate the Spitzer el. conductivity,
# See NRL (p.37)   sigma_par = 2*tau_ei * n_e * e**2 /e_mass
# Specify: 
T_e= 2140. # eV
n_e= 1.0e13 # cm-3
CL=  18 # Coulomb log. == gamaset
Z_i= 1. #10.  # ==bnumb(ion)
# Then,
tau_ei= 3.44e5* T_e**1.5 / (n_e *CL* Z_i)
sigma_par = 2*tau_ei * n_e * e**2 /e_mass
print ' tau_ei = 3.44e5* T_e**1.5 / (n_e *CL* Z_i) ==',  tau_ei
print ' sigma_par = 2*tau_ei * n_e * e**2 /e_mass ==', sigma_par
# HOWEVER: the 2* factor in front of tau_ei may have a dependence on Z_i.
# It should be changed to 3 for large Z_i (book by S.V.Mirnov)


# Evaluate tau_resistivity factors for further plotting:
sigma_0 = 1.0/resist_phi_0  # conductivity (cgs) at r=0
sigma_a = 1.0/resist_phi_a  # conductivity (cgs) at r=a
print ' sigma_0 = 1.0/resist_phi_0  ==', sigma_0 
print ' sigma_a = 1.0/resist_phi_a  ==', sigma_a 

iplot_tau_decay=0 #1 #Set to 1 if you want to add these plots (test/verification)
# The decay time of initially ~parabolic profile of E field 
# (with max value at r=0, and zero at plasma edge r=a)
tau_decay_0 = 1e3*(4*pi*sigma_0/clight**2)/lam**2 # CONVERT to msec
tau_decay_a = 1e3*(4*pi*sigma_a/clight**2)/lam**2 # CONVERT
# Based on sigma_par, see above:
tau_decay_par=1e3*(4*pi*sigma_par/clight**2)/lam**2 # CONVERT to msec
print ' tau_decay at r=0 and r=a [msec]=', tau_decay_0, tau_decay_a
# So now we can plot A*exp(-t/tau_decay) curves.
# Factor A will be taken from the peak value of Itor(time) curve.

# For plots:
#---------------------------------------
params = {
    'axes.linewidth': linw,
    'lines.linewidth': linw,
    'axes.labelsize': fnt+4,
    'text.fontsize': fnt+4,
    'legend.fontsize': fnt,
    'xtick.labelsize':fnt,
    'ytick.labelsize':fnt,
    'xtick.linewidth':linw,
    'ytick.linewidth':linw,
    'font.weight'  : 'regular'
}
plt.rcParams.update(params)
#rc.defaults() #to restore defaults
mpl.rcParams['font.size']=fnt+2  # set font size for text in mesh-plots
#----------------------------------------------------------------------



# Specify netcdf file name, with sub-directory if any:
###file_name= 'freidberg_full_eps_AF.0.1.nc'

#file_name= 'freidberg_full_eps_AF.0.8fixed.nc'
#file_name= 'freidberg_full_eps_AF.0.8conserv.nc'
#file_name= 'freidberg_full_eps_AF_5th_root_J0.4.1.6.nc'
#file_name= 'freidberg_full_eps_AF.0.10.8_170821.nc' # see /splines_2nd_root_dtr/

#file_name= 'freidberg_full_eps_AF.0.10.8_170821parab.nc'
#file_name='AF.0.11.7.nc'
#file_name='AF.0.11.8.6.nc'

#file_name= 'AF-EC-Edc.1.0.nc' # bad test with ECCD, fow_ver.170314.3 (or.2)
#file_name= 'AF-EC-Edc.1.0_YuP170821.nc' # 21-08-2017 test with ECCD (rayech file)
#file_name= 'AF-EC-Edc.1.0_YuP170821_nt20.nc' # 21-08-2017 test with ECCD (rayech file)
#file_name= 'AF-EC-Edc.1.0_YuP170821_nt20_colmodl1.nc' # 21-08-2017 test with ECCD (rayech file)
#file_name= 'AF-EC-Edc.1.0.nc'
#file_name='AF-EC-Edc.1.2.nc' #'AF-EC-Edc.1.4.nc' #'AF-EC-Edc.1.3.1.nc'

#file_name='AF-EC-Edc.1.5.nc'

#file_name='tdep_ko_amp-far.4.nc'  # Bob's run 171008 on runaway.
#file_name='tdep_ko_amp-far.5.1.nc'  # Bob's run 171008 on runaway.
#file_name='tdep_ko_amp-far_Drr.0.nc'  # Bob's run 171008 on runaway.
#file_name='tdep_ko_amp-far_Drr.1c.nc'  # Bob's run 171008 on runaway.
#file_name='tdep_ko_amp-far_Drr.1c.nc'  # Bob's run 171008 on runaway.
#file_name='tdep_ko_amp-far.5.1.1c.nc'
#file_name='tdep_ko_NIMRODfiles_Tmin10eV_nstop680_dtr0.1em4_conserv.nc'

#file_name='cmod1101201020_AF.23.nc'  # Bob's runs for CMOD (with LHCD)

# Short reruns, nstop=40..80 [2019-09-10],  DIIID, flat profiles
#file_name='tdep_ko_amp-far.4.3_short_test_short_YuP_Intel_R8_fpconstant.nc'
#file_name='tdep_ko_amp-far.5.1.1c_short_nstop80.nc' # lrz=11 Bad oscillations
#file_name='tdep_ko_amp-far.5.1.1c_short_nstop80_lrz51.nc' # Good
#file_name='tdep_ko_amp-far.5.1.1c_short_nstop80_lrz21.nc' # Not bad; small oscillations
#file_name='tdep_ko_amp-far.5.1.1c_short_nstop80_lrz21_pellet7.nc'

#file_name='tdep_ko_amp-far.5.1.1c_lrz21_noPellet.nc' # just for reference
#file_name='tdep_ko_amp-far.5.1.1c_lrz21_Pellet.nc'
# after 2019-10-03 :
#file_name='tdep_ko_amp-far.5.1.1c_lrz51_Pellet.nc' #very nice smooth plots
#file_name='tdep_ko_amp-far.5.1.1c_lrz51_Pellet_dt01ms.nc'  # run20
#file_name='tdep_ko_amp-far.5.1.1c_lrz51_Pellet_dt005ms.nc' # run21 and 21nb
# Starting from 21nb, Ne,bound are added into KO source.
#file_name='tdep_ko_amp-far.5.1.1c_lrz51_Pellet_dt0025ms.nc' #run22 temp_expt_tau=0.1d-3
#file_name='tdep_ko_amp-far.5.1.1c_lrz51_Pellet_dt0010ms.nc' #run22a, with smaller dtr

#file_name='tdep_ko_amp-far.5.1.1c_lrz101_Pellet_dt0020ms.nc'

#file_name='tdep_ko_amp-far_lrz51_Pellet_Te10keV_dt0010ms.nc' #bad

# Switched to lrz=101  More stable --------------------------------
#file_name='tdep_ko_amp-far_lrz101_Pellet_Te10keV_dt0020ms.nc' # PC: good RE
#file_name='tdep_ko_amp-far_lrz101_Pellet_Te10keV_dt0020ms_1.nc' #mpi32, nampfmax2-8
#file_name='tdep_ko_amp-far_lrz101_Pellet_Te10keV_dt0020ms_1.nc'

#----------------- Good for presentation/APS2019:
# In /lrz101_AMPF_pellet_Te2keV_tau1ms_b/    (10/09/2019)
# Stable run on PC, with dtr=0.05e-3
# temp_expt_tau0= 30.0d-3 ![sec]slow decay time of Te(t) 
# temp_expt_tau1=  1.0d-3 ![sec]fast decay time of Te(t)(for Thermal Quench)
#file_name='tdep_ko_amp-far_lrz101_Pellet_Te2keV_dt0050ms.nc' # slow tau1, for slides

# In /lrz101_AMPF_pellet_Te2keV_tau01ms_a/   (10/10/2019)
# Stable run on PC, with dtr=0.05e-3,   fast tau1 (but very small RE)
# temp_expt_tau0= 30.0d-3 ![sec]slow decay time of Te(t) 
# temp_expt_tau1=  0.1d-3 ![sec]fast decay time of Te(t)(for Thermal Quench)
###file_name='tdep_ko_amp-far_lrz101_Pellet_Te2keV_dt0050ms_tau01.nc'
# Also see /lrz101_AMPF_pellet_Te2keV_tau01ms_b_coll_nohesslow/
#  similar run, but hesslow gscreen and hbethe are disabled.

# Good 10keV run, /lrz101_AMPF_pellet_Te10keV_tau01ms_b_coll_nohesslow/ (10/11/2019)
# or /lrz101_AMPF_pellet_Te10keV_tau01ms_a/
#file_name='tdep_ko_amp-far_lrz101_Pellet_Te10keV_dt0050ms_tau01.nc' #tau1=0.1ms

# Even faster tau1=0.05ms
#file_name='tdep_ko_amp-far_lrz101_Pellet_Te10keV_dt0050ms_tau005.nc'
# Somewhat more conversion to RE, from larger rho (rho<0.2 - same RE current)

# Also faster tau1=0.05ms with Te0=2.5keV case
#file_name='tdep_ko_amp-far_lrz101_Pellet_Te2keV_dt0050ms_tau005.nc'
# Small fraction is converted (<30kA)  Not taken for APS
# But see lrz101_AMPF_pellet_Te2keV_tau005ms_V200: Vpell->200m/s:
# 95kA of RE

# Replot NIMROD-CQL3D runs from 
#file_name='tdep_ko_NIMRODfiles_Tmin10eV_nstop680_dtr0.1em4_conserv.nc' #2018-09-24
#file_name='test_read_nimrod_enorm10MeV_v2.nc'
#file_name='test_read_nimrod_enorm10MeV_v4r.nc'
#file_name='test_read_nimrod_enorm10MeV_v5r.nc' # "r" = R-outboard profiles (not FSA)
#file_name='test_read_nimrod_enorm10MeV_v5fsa.nc' # FSA (very small RE here)
#file_name='test_read_nimrod_enorm10MeV_v5r_tran.nc' #with drr_scale=1e-5
# See /cql3d_nimrod_v5fsa_ext_REcorr_transp1m3/
#file_name='test_read_nimrod_enorm10MeV_v5fsa_tran_ext.nc' #FSA,long,drr_scale=1e-3
#file_name='test_read_nimrod_enorm10MeV_v5fsa_notran_ext.nc'
#file_name='test_read_nimrod_enorm10MeV_v5fsa_tran_ext_KOdis.nc' # knockon='disabled'
#----------- Better: enorm-->40MeV, jx-->300 (was 150)  --------------
#file_name='test_read_nimrod_enorm40MeV_v5fsa_tran_ext.nc' #1m3 (drr_scale=1.d-3)
#file_name='test_read_nimrod_enorm40MeV_v5fsa_tran_ext_KOdis.nc' # knockon='disabled'
##file_name='test_read_nimrod_enorm100MeV_v5fsa_tran_ext.nc'
#file_name='test_read_nimrod_enorm40MeV_v5fsa_notran_ext.nc'
#file_name='test_read_nimrod_enorm40MeV_v5fsa_notran_ext_KOdis.nc'
#file_name='test_read_nimrod_enorm40MeV_v5fsa_tran1m4.nc'
#Trying dB/B scaled by 1/20 (Drr by 2.3e-3):
#file_name='test_read_nimrod_enorm40MeV_v5fsa_tran2m3.nc' # Noise at 1.4-1.6ms
# [2022-01-06] Added 1/gamma^5
#file_name='test_read_nimrod_enorm40MeV_v5fsa_tran_ext_gammi5.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi5_itl_tran1m3.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi5_itl_tran1m2.nc'
file_name='cql_nimrod_v5fsa_Epressau2_gammi5_itl_tran000.nc'

# Series of runs with different drr_scale and different 1/gamma^n scale
#file_name='cql_nimrod_v5fsa_Epressau2_gammi0_itl_tran1m3.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi1_itl_tran1m3.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi2_itl_tran1m3.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi3_itl_tran1m3.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi4_itl_tran1m3.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi5_itl_tran1m3.nc'

#file_name='cql_nimrod_v5fsa_Epressau2_gammi0_itl_tran1m2.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi1_itl_tran1m2.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi2_itl_tran1m2.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi3_itl_tran1m2.nc'
file_name='cql_nimrod_v5fsa_Epressau2_gammi4_itl_tran1m2.nc' #--- Good for paper
#file_name='cql_nimrod_v5fsa_Epressau2_gammi5_itl_tran1m2.nc'
#gammi4_tran1m2 : 
# noise at rho= 0.75(g,1.45-1.55ms)--0.84(m,1.35-1.6ms)--0.93(c,t=1.2-1.8ms)
#(for gammi5_tran1m2 - all plots are very similar).

#file_name='cql_nimrod_v5fsa_Epressau2_gammi0_itl_tran4m2.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi1_itl_tran4m2.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi2_itl_tran4m2.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi3_itl_tran4m2.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi4_itl_tran4m2.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi5_itl_tran4m2.nc'

#file_name='cql_nimrod_v5fsa_Epressau2_gammi0_itl_tran1m1.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi1_itl_tran1m1.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi2_itl_tran1m1.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi3_itl_tran1m1.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi4_itl_tran1m1.nc'
#file_name='cql_nimrod_v5fsa_Epressau2_gammi5_itl_tran1m1.nc'

# NIMROD rerun[2024-05-14]
#file_name='cql_nimrod_v5fsa_Epressau2_gammi5_itl_tran000_r2024.nc'
#--- New case with Ar SPI [2024-05-14]
#file_name='cql_nimrod_Ar_tran000.nc' # New run with Ar
#file_name='cql_nimrod_Ar_tran000_jx600.nc' # same but jx=600
# UPDATED eqdsk file to g160606.02990_m3q_257
file_name='cql_nimrod_Ar_tran000_r4_eqdsk.nc'
# NIMROD, new data (and using geqdsk)
file_name='cql_nimrod_Ar177053_tran000_r3.nc'
file_name='cql_nimrod_Ar177053_tran000_r3_flcq_lrz44.nc' #r3, but lrz doubled
file_name='cql_nimrod_Ar177053_tran000_r3_rycq_lrz44.nc' #r3, but lrz doubled
file_name='cql_nimrod_Ar177053_tran000_r4_rycq_lrz44.nc' #/4CQL3D_45_r4_rycq_lrz44/
file_name='cql_nimrod_Ar177053_tran000_r5_rycq_lrz44.nc' #/4CQL3D_shallow_r5_rycq_lrz44/
#[Jul/12]Trying elecfld_data='readdata' (direct data for E from files):
file_name='cql_nimrod_Ar177053_tran000_r3a_flcq_lrz44_edata.nc' #Ire=1.6kA
file_name='cql_nimrod_Ar177053_tran000_r6_rycq_lrz44.nc' #4CQL3D_axis/ 2.6kA
file_name='cql_nimrod_Ar177053_tran000_r7_rycq_lrz44.nc' #7kA
#2024-07-25 Added rho~sqrt(psi); corrected def. of rho=1.
file_name='cql_nimrod_Ar177053_tran000_r7a_rycq_lrz44.nc' # 12.8kA
# elecfld_data=readdata:
file_name='cql_nimrod_Ar177053_tran000_r7b_rycq_lrz44.nc' # ??? Too high
# Back to elecfld_data='jspitzer', but no Hesslow corrections (gamafac='disabled')
file_name='cql_nimrod_Ar177053_tran000_r7c_rycq_lrz44.nc'
file_name='cql_nimrod_Ar177053_tran000_r7d_rycq_lrz44.nc' #17kA
#run7f  Same as 7a but Use flcq* data
file_name='cql_nimrod_Ar177053_tran000_r7f_flcq_lrz44.nc'
#-----------------------------------------------
# BACK to 160606 with Argon.
file_name='cql_nimrod_Ar160606_tran000_r4a_flcq_lrz44_sqpolflx.nc'
file_name='cql_nimrod_Ar160606_tran000_r4b_flcq_Esptz_har08_n0.nc'
file_name='cql_lrz44_nimrod_flcq_CQLlowetamax_Esptz_noTrapped.nc'
# 08/22 C.Kim fixed issues with E :
file_name='cql_nimrod_Ar160606_r4d_flcq_Esptz.nc'
file_name='cql_nimrod_Ar160606_r4d_flcq_Edata.nc'
file_name='cql_nimrod_160606_cql42Ar_r5a_flcq_Esptz.nc' #70kA RE
file_name='cql_nimrod_160606_cqlNe14_r6a_flcq_Esptz.nc'
file_name='cql_nimrod_Ar177053_Ar50_run8a_flcq.nc'
file_name='cql_nimrod_Ar177053_Ar50_run8b_flcq_gamafacENABLED.nc'
file_name='cql_nimrod_160606_Ar_cqlLV_r7a_flcq_Esptz.nc'
file_name='cql_nimrod_160606_Ar_cqlHV_r8a_flcq_Esptz.nc' #RE=120kA
file_name='cql_nimrod_160606_Ar_cqlHV_r8b_flcq_Edata.nc'
file_name='cql_nimrod_160606_Ar_cqlLV_r7b_flcq_Edata.nc'



#file_name='test_read_nimrod_enorm10MeV_v5ext.nc' # "r" = R-outboard profiles (not FSA)
#file_name='test_read_nimrod_enorm10MeV_v5fsa_ext.nc' # FSA profiles

#file_name='test_read_nimrod_enorm10MeV_v6fsa_notran.nc' # Val Izzo data
#file_name='test_read_nimrod_enorm10MeV_v6fsa_tran1m3.nc' # Val Izzo data
#file_name='test_read_nimrod_enorm10MeV_v6fsa_tran1m2.nc' # Val Izzo data
#file_name='test_read_nimrod_enorm10MeV_v6fsa_tran1.nc' # Val Izzo data scale=1
#file_name='test_read_nimrod_enorm10MeV_v6fsa_tran10.nc' # Val Izzo data scale=10
#file_name='test_read_nimrod_enorm10MeV_v6_notran.nc' #-- Ok
#file_name='test_read_nimrod_enorm10MeV_v6_tran4m1.nc' # ok
#file_name='test_read_nimrod_enorm10MeV_v6_tran6m1.nc' # 0.6  ok



file_one=file_name


#-------- Bob's runs for C-mod -------------------------------
#file_name='cmod1101201020_AF_LH.2_pellet.1.nc'
# With AMP-FAR:
#file_name='cmod1101201020_AF_LH.2_pellet.3.nc' #AMP-FAR: bad
# No AMP-FAR, good run:
#file_name='cmod1101201020_AF_LH_Wpellet.nc' #see run W8a 200ms range looks good


#file_one='cmod1101201020_AF_LH.2.1_yup_short_nstop200_iprocur_norf_neohh.nc'

#file_one='freidberg_cyl_PoP2008_yup2_noTdrop.nc'

#file_one='JET_PoP2008_r26h_AF_Tdrop_n210_nampfmax2_lrz11_Z1_Eflat.nc'
# After 2020-01-28:
# Constant gamaset=12. (but I0 is too large =2.63MA; converted to RE~1.45MA):
#file_one='JET_PoP2008_r26h_AF_Tdrop_n210_nampfmax2_lrz11_Z1_Eflat_L12.nc'
# Same but trapredc=0.99 (to get rid of trap cone):
#file_one='JET_PoP2008_r26k_AF_Tdrop_n210_nampfmax2_Z1_Eflat_L16_trapredc.nc'
#(strange - results are exactly same as in previous, ***_Eflat_L12.nc)
# CYLINDER:
#file_one='JET_PoP2008_r38_AF_Tdrop_n210_Z1_L12_E144_eqdsk_neobscd.nc'
#file_one='JETcirc0_r42_AF_Z1_L12_E201__tauT_05ms_nampfmax2_mpi32_O3.nc'
#file_one='JETcirc0_r42_AF_Z1_L12_E201__tauT_05ms_nampfmax2_e10K_jx800.nc'
#file_one='JETcirc0_r43_AF_Z1_L12_E201__tauT_05ms_nampfmax2_e5K_lrz30.nc'
#file_one='JETcirc0_r45_AF_Z1_L12_E201__noTdrop_e5K_lrz30.nc'
#file_one='JETcirc0_r46_AF_Z1_L12_E201__noTdrop_e5K_lrz30_ampfadd_neosigma.nc'
#file_one='JETcirc0_r47_AF_Z1_L12_E201__noTdrop_e5K_lrz30_ampfadd_neobscd.nc'
#file_one='JETcirc0_r48_AF_Z1_L12_E201__Tdrop_e5K_lrz30_iy320.nc'

# 02-12 Changed usage of subr.resthks: use zresstau1 and zresstau2 (Sauter eqns):
#file_one='JETcirc0_r53_AF_Z1_L12_E201__tauT_05ms_nampfmax2_e5K_lrz30.nc'
#file_one='JETcirc0_r54_AF_Z1_L12_E177__tauT_05ms_nampfmax2_e5K_lrz30_neobs.nc'

# Now back to tau_T=0.1ms (see r32) but increased enorm,jx,lrz:
#file_one='JETcirc0_r55_AF_Z1_L12_E201__tauT_01ms_nampfmax2_e5K_lrz30.nc'
#file_one='JETcirc0_r56_AF_Z1_L12_E201__tauT_01ms_nampfmax2_e5K_lrz30_neobs.nc'
#file_one='JETcirc0_r53a_AF_Z1_L12_E201__tauT_05ms_nampfmax2_e5K_lrz30_mx10.nc'

#file_one='JETcirc0_r66_AF_Z1_L12_E201__tauT_0025ms_nampfmax2_e5K_lrz30.nc' #n=1000
#file_one='JETcirc0_r66_AF_Z1_L12_E201__tauT_0025ms_nampfmax2_e5K_lrz30_neobs.nc' #n=1000

#file_one='JETcirc0_r64_AF_Z1_L12_E201__tauT_005ms_nampfmax2_e5K_lrz30.nc' #n=400
#file_one='JETcirc0_r64_AF_Z1_L12_E201__tauT_005ms_nampfmax2_e5K_lrz30_neobs.nc' #n=400
#file_one='JETcirc0_r65_AF_Z1_L12_E201__tauT_005ms_nampfmax2_e5K_lrz30.nc' #n=600
#file_one='JETcirc0_r65_AF_Z1_L12_E201__tauT_005ms_nampfmax2_e5K_lrz30_neobs.nc' #n=600

#file_one='JETcirc0_r60_AF_Z1_L12_E201__tauT_02ms_nampfmax2_e5K_lrz30.nc' #n=290
#file_one='JETcirc0_r60_AF_Z1_L12_E201__tauT_02ms_nampfmax2_e5K_lrz30_neobs.nc' #n=290
#file_one='JETcirc0_r59_AF_Z1_L12_E201__tauT_03ms_nampfmax2_e5K_lrz30.nc'
#file_one='JETcirc0_r59_AF_Z1_L12_E201__tauT_03ms_nampfmax2_e5K_lrz30_neobs.nc'

#file_one='JETcirc0_r61_AF_Z1_L12_E201__tauT_04ms_nampfmax2_e5K_lrz30.nc'
#file_one='JETcirc0_r61_AF_Z1_L12_E201__tauT_04ms_nampfmax2_e5K_lrz30_neobs.nc'
#Changes are needed for higher tau_T. Larger enorm, smaller xfac, longer nstop:
#file_one='JETcirc0_r62_AF_Z1_L12_E201__tauT_06ms_nampfmax2_e10K_xfac005.nc' #n=350
#file_one='JETcirc0_r62_AF_Z1_L12_E201__tauT_06ms_nampfmax2_e10K_xfac005_neobs.nc' #n=350
#file_one='JETcirc0_r63_AF_Z1_L12_E201__tauT_07ms_nampfmax2_e10K_xfac005.nc' #n=400
#file_one='JETcirc0_r63_AF_Z1_L12_E201__tauT_07ms_nampfmax2_e10K_xfac005_neobs.nc' #n=400


#---------- ITER runs, with tau0_T = 1ms (for T-drop) ------------------------
# First few runs - with ne=const
#file_one='ITERcirc0_r01_AF_L12_tauT1ms_e10K_xfac005_n120_ne_const__tst8ms.nc'
# Ifinal~10MA, steady state after t~18ms :
#file_one='ITERcirc0_r03_AF_L12_tauT1ms_e10K_xfac005_n120_ne_const__tst8ms.nc'
# Next - with slow ne growth (ne is almost const over modeling time of 28ms)
#Slow growth of ne, 800-8=792ms (bctime= 0.d0,8.d-3,0.8 ! [sec]) Ifinal~10MA
#file_one='ITERcirc0_r04_AF_L12_tauT1ms_e10K_xfac005_n120_ne_grows__tst8ms.nc'

#Faster growth of ne (bctime=0.d0,8.d-3,15.d-3 [sec])(7ms). Ifinal~6MA (reduced)
#file_one='ITERcirc0_r05_AF_L12_tauT1ms_e10K_xfac005_n120_ne_grows__tst8ms.nc'

# ne_grow7ms, but 4x smaller dtr1. Final RE current is larger ~8MA
#file_one='ITERcirc0_r06_AF_L12_tauT1ms_e10K_xfac005_n420_ne_grows__tst8ms.nc'

#Shorter ne_grow2ms, 10-8=2ms (bctime= 0.d0,8.d-3,10.d-3 [sec]). Ifinal~3.5MA
##file_one='ITERcirc0_r07_AF_L12_tauT1ms_e10K_xfac005_n420_ne_grows2ms.nc'
#Run ne_grow7ms looks more natural; ne sim.to Te drop (saturated at t=15ms)

#Try to reduce dtr1 even further, and try nampfmax=4 (was 2 in all above runs)

#As in r06, but nampfmax=4.  Itot: Goes to 10MA, then to9MA (RE: 7MA then to 6MA)
#file_one='ITERcirc0_r06_AF_L12_tauT1ms_e10K_xfac005_n420_ne_grows7ms_nampfmax4.nc'

#Twice smaller time step.  Itot: 9.2MA then drops to 8.4MA
#file_one='ITERcirc0_r06_AF_L12_tauT1ms_e10K_xfac005_n820_ne_grows7ms_nampfmax2.nc'
#as prev., but nampfmax=4. Itot: 11.1-->9.1MA (RE:7.4MA then drops to 4.5MA)
#file_one='ITERcirc0_r06_AF_L12_tauT1ms_e10K_xfac005_n820_ne_grows7ms_nampfmax4.nc'

#ITER/In general: still sensitive to dtr1 and nampfmx


#--------------------------------------------------------------------------
#file_one='cmod1101201020__LH.3.2_yup2.nc'
#file_one='cmod1101201020_AF_LH.3.2.nc'
#file_one='cmod1101201020__LH.3.2_yup2_ampfar.nc' # corrupted
#file_one='cmod1101201020__LH.3.2_yup3_ampfar.nc' # 20steps at dt=0.5ms

#file_one='cmod1101201020_3.2_AF_BS.nc'
#file_one='cmod1101201020_3.2_AF_tdboothi.nc'
#file_one='cmod1101201020_3.2_AF_dbscurm.nc' # added bscurm into AMPFAR Eqn
#file_one='cmod1101201020_3.2_AF_dsig.nc' # added ALL: bscurr, dcurr, dsig
#file_one='cmod1101201020_3.2_AF_dsig_n20.nc'
#file_one='cmod1101201020_AF_LH.4_yup6.nc'
#file_one='cmod1101201020_AF_LH.4.0.nc'
#file_one='cmod1101201020_AF_LH.4.nc'
#file_one='cmod1101201020_AF_LH.4.0.1.1_nstop10.nc'
# 2020-02-11  See /cmod1101201020_AF_LH.4.0.1/
#file_one='cmod1101201020_AF_LH.4.0_nstop1000.nc' #=distrfunc.nc for restart
#file_one='cmod1101201020_AF_LH.4.0.1_nstop10_yup2.nc' #restart, using large step

# 2020-10-29
#file_one='cmod1101201020_AF_LH.7.nc' #[0--900ms]
#file_one='cmod1101201020_AF_LH.7_rerun1.nc' #[0--900ms]
#-- From here (*.8, ...) using gamaset=-1.(NRL)
#file_one='cmod1101201020_AF_LH.8.nc' #[0--900ms] gamaset=-1(NRL)
#file_one='cmod1101201020_AF_LH.9.nc' #[0-900ms],ampfadd="enabled",totcrt=-0.425d6
#file_one='cmod1101201020_AF_LH.10.nc' #[0-900ms],ampfadd="neo+bscd",totcrt=-0.425d6
#file_one='cmod1101201020_AF_LH.11.nc' #lrz=121,ampfadd="neo+bscd",totcrt=-0.400d6
#file_one='cmod1101201020_AF_LH.10.1.nc'

#file_one='cmod1101201020_AF_LH.7.0.nc' #dtr=0.1d-3 sec --> +100ms total
#file_one='cmod1101201020_AF_LH.7.1.nc' #dtr=0.1d-3 sec --> +160ms total(pre-flake)

#--file_one='cmod1101201020_AF_LH.7.2.0.nc' # nstop=400
#--file_one='cmod1101201020_AF_LH.7.2.1.nc' # nstop=100 
#--file_one='cmod1101201020_AF_LH.7.2.2.nc' # nstop=200 
#--file_one='cmod1101201020_AF_LH.7.2.0.2.nc' #curr_fit; new genray.nc_run.4-4.1
#--file_one='cmod1101201020_AF_LH.7.2.0.3.nc' #curr_fit; new genray.nc_run.4-4.1
#file_one='cmod1101201020_AF_LH.7.2.0.5.nc' #curr_fit; new genray.nc_run.4-4.1 GOOD
#file_one='cmod1101201020_AF_LH.10.2.0.5.nc' #curr_fit;genray.nc_run.4-4.1
#file_one='cmod1101201020_AF_LH.10.2.0.5.1.nc' # genray.nc_run.4-4.0.nc(Zeff=2)

#--file_one='cmod1101201020_AF_LH.6.0.4.nc' #difusr=50, nstop=330

#  Now using  genray.nc_run.4-4.1, and Zeff=const=4.1; From 1160ms:
#file_one='cmod1101201020_AF_LH.7.2.2.3.nc' #Zeff=4.1, halved dt at n>600. GOOD
#(need smaller dtr1 after n=600, otherwise A-F unstable -> crash at t=1225ms)
#file_one='cmod1101201020_AF_LH.7.2.2.5.nc' #from t=1240.25ms,+17.5, dtr=0.025ms

#file_one='cmod1101201020_AF_LH.7.2.3.0.nc' #noAF; expt T-drop; 1160-1360ms 
#file_one='cmod1101201020_AF_LH.7.2.3.1.nc' #noAF; expt T-drop; 1160-1260ms, tau=30ms 
#file_one='cmod1101201020_AF_LH.7.2.3.2.nc' #noAF; expt T-drop; 1160-1260ms, tau=20ms 
#file_one='cmod1101201020_AF_LH.7.2.3.3.nc' #noAF; expt T-drop; 1160-1260ms, tau=10ms 
#file_one='cmod1101201020_AF_LH.7.2.3.4.nc' #noAF; expt T-drop; 1160-1260ms, tau=5ms 
#file_one='cmod1101201020_AF_LH.7.2.3.5.nc' #noAF; expt T-drop; 1160-1260ms, tau=1ms 

#file_one='cmod1101201020_AF_LH.7.2.4.2.nc' # W pellet, using exp(drop), noAF
#file_one='cmod1101201020_AF_LH.7.2.4.3.nc' # W using iprote(i)='spline-t' Better
#file_one='cmod1101201020_AF_LH.7.2.4.6.nc' #Added Zi=100 to get Zeff=4.1
#file_one='cmod1101201020_AF_LH.7.2.4.8.nc' #Zeff=4.1, and E(0)=3*E(a)
#file_one='cmod1101201020_AF_LH.7.2.4.9.nc' #Zeff=4.1, and E(0)=2*E(a) GOOD
#file_one='cmod1101201020_AF_LH.7.2.4.9_mpi.nc' #Fixed bugs in ADC**[2020-11-16]
#file_one='cmod1101201020_AF_LH.10.2.4.9.nc' #Zeff=1.92, and E(0)=2*E(a)
#file_one='cmod1101201020_AF_LH.10.2.4.9.1.nc' #genray.nc_run.4-4.0.nc(Zeff=2

#file_one='cmod1101201020_AF_LH.7.2.5.0.nc' #Bob's run with A-F, 2020-11-14 
#(very short physical time; pellet just enters rho=1). 

#file_one='cmod1101201020_AF_LH.7.2.5.1.nc' #Wflake+AF , from 1160ms, n=900 to 75ms
#file_one='cmod1101201020_AF_LH.7.2.5.1_nampfmax4.nc' # somewhat better than nampfmax=2,
#(current drops to ~0.28MA vs 0.2MA) but becomes unstable at t~1239ms
#file_one='cmod1101201020_AF_LH.10.2.5.1.nc' #Wflake+AF , from 1160ms, n=900 to 75ms
#file_one='cmod1101201020_AF_LH.10.2.5.1.1.nc' #genray.nc_run.4-4.0.nc(Zeff=2)


#file_one='test_read_nimrod_enorm500_v2.nc'
#file_one='test_read_nimrod_enorm2000_v2.nc'
#file_one='test_read_nimrod_enorm10MeV_v2.nc'
#file_one='test_read_nimrod_enorm10MeV_v4r.nc'
#file_one='test_read_nimrod_enorm10MeV_v5r.nc'
#file_one='test_read_nimrod_enorm10MeV_v5ext.nc'
#file_one='test_read_nimrod_enorm10MeV_v5fsa_ext.nc'
#file_one='test_read_nimrod_enorm10MeV_v5r_ext.nc'

# Set of runs. Set the list below, set nfiles accordingly.
# If you want JUST ONE file, leave one out of list, and set nfiles to 0
file=[]
#----- Start the list:
file.append(file_one)  # If just one, leave this line, comment other lines
#setabove:file.append('cmod1101201020_AF_LH.7.nc') #start(set above: file_one=)
#file.append('cmod1101201020_AF_LH.7.0.nc') # curr_fit, up to 1000ms. GOOD
#-- From here (*.8,...) using gamaset=-1.(NRL) 
#file.append('cmod1101201020_AF_LH.8.0.nc') # curr_fit, up to 1000ms.
#file.append('cmod1101201020_AF_LH.9.0.nc') #curr_fit,to1000ms,ampfadd="enabled"
#file.append('cmod1101201020_AF_LH.10.0.nc') #curr_fit,to1000ms,ampfadd="neo+bscd"

#file.append('cmod1101201020_AF_LH.7.1.nc') # curr_fit, up to 1160ms. GOOD
#file.append('cmod1101201020_AF_LH.8.1.nc') # curr_fit, up to 1160ms.
#file.append('cmod1101201020_AF_LH.10.1.nc') # curr_fit, up to 1160ms.

#--file.append('cmod1101201020_AF_LH.7.2.0.nc') #old genray.nc
#--file.append('cmod1101201020_AF_LH.7.2.0.2.nc') #genray.nc_run.4-4.1. Crash at 1220ms
#file.append('cmod1101201020_AF_LH.7.2.0.3.nc') #genray.nc_run.4-4.1. curr_fit, Crash at 1243ms
#file.append('cmod1101201020_AF_LH.7.2.2.3.nc') #from 1160.25ms;Z=4.1,dt=0.10->0.05ms at n>600
#file.append('cmod1101201020_AF_LH.7.2.2.5.nc') #from 1240.25ms,+17.5,dtr=0.025ms
#-----
#file.append('cmod1101201020_AF_LH.7.2.4.9.nc') #1160->; Zeff=4.1, E(0)=2*E(a), T,n from experim
#----- End of list.




# Automatic procedure, if many files with same prefix (same file_base):
#for ifile in range(0, nfiles):
#    file.append(file_base+str(ifile)+'.nc') 
print file
#print shape(file)
nfiles=0 #3-1 #4-1 #0 #5-1 #7-1 # number of files in the list above, minus 1
#Now file[] contains list of all *.nc files from above    

#Put netCDF structures into list.
ifile=0
#snetcdf =nc.netcdf_file(file[ifile],'r') # Does not work for YuP
s_file_cql3d= Dataset(file[ifile], 'r', format='NETCDF4')
# to read the basic stuff, common for all files:
lrz=s_file_cql3d.variables['lrz'].getValue()   
lrz=np.asscalar(lrz)  
i_R_stop=lrz
iy=s_file_cql3d.variables['iy'].getValue()
iy=np.asscalar(iy)
jx=s_file_cql3d.variables['jx'].getValue()  
jx=np.asscalar(jx)      
print 'lrz,iy,jx =',lrz,iy,jx
dvol= array(s_file_cql3d.variables['dvol'])  # cm^3
dvol= np.asarray(dvol)
darea=array(s_file_cql3d.variables['darea']) # cm^2
darea= np.asarray(darea)
#x=s_file_cql3d.variables['x'][:]
x= array(s_file_cql3d.variables['x'])
x= np.asarray(x)

t_combined_low_lim=0.0 #0.650 #0. #0.750 #[sec] For plots like Ip_combined vs time_combined:
# set the lower limit in time axis.


time_combined=[]
Ip_combined=[]
I_bs_combined=[]
I_starnue_combined=[]
I_starnue0_combined=[]
I_RE_combined=[]
I_dsE_bs_combined=[] # To hold delta_sigma*E + I_bs
elecfld_combined=[]
curr_combined=[]
zeff_r_combined=[] # Zeff at lr=1 FP surface
for ifile in range(0,nfiles+1):
    print 'file:', file[ifile]
    snetcdf = Dataset(file[ifile], 'r', format='NETCDF4')
    #snetcdf =nc.netcdf_file(file[ifile], 'r')
    #  'bctshift' May not be available in *.nc files from older runs
    bctshift= snetcdf.variables['bctshift'].getValue()  #getValue() for scalar
    time_ifile= array(snetcdf.variables['time']) +bctshift #[sec]
    time_ifile=np.asarray(time_ifile)
    print 'ifile, shape(time_ifile),MIN/MAX,bctshift=' , ifile, shape(time_ifile),\
    np.min(time_ifile),np.max(time_ifile), bctshift
    time_combined.extend(time_ifile)
    #Now Total plasma current
    ccurtor= array(snetcdf.variables['ccurtor'])  # [Amps] 
    # CONVERT to kA:
    #c     ccurtor(lr_) is the cumulative toroidal current, integrating
    #c                   curtor(lr) in poloidal cross-section (Amps),
    #c                   accounting for pol variation of tor current.
    #c                   See tddiag
    ccurtor=ccurtor/1000. # kA
    # i.e. size= (nstop+1,lrz)
    #The value of ccurtor[itime,lrz] corresponds to the total integral of current:
    Ip_ifile=ccurtor[:,lrz-1] # [kA]   as a function of time.   
    if ifile==3:
        Ip_ifile[0]=Ip_ifile[1] # Adjust the glitch in n=1 data point for this run
    #print 'Ip_ifile: shape, min/max',Ip_ifile.shape,np.min(Ip_ifile),np.max(Ip_ifile) 
    Ip_combined.extend(Ip_ifile)
    #--- Now RE current
    curra=array(snetcdf.variables['curra']) #shape: (nstop+1,lrz)
    #curra= np.absolute(curra) # Sometimes it is <0 (neg.tor.dir). Plot |curra|
    # 'Runaway FSA parallel cur density above ucrit'   'Amps/cm**2'
    I_RE_ifile= np.dot(curra,darea)/1e6  # 1/e6 is to MA
    I_RE_combined.extend(I_RE_ifile) # MA
    #--- Now additional current: I_dsE_bs_combined= delta_sigma*E + I_BS
    bscurr_e_gen=array(snetcdf.variables['bscurr_e_gen'])  # [A/cm^2]
    I_bs_e_gen= np.dot(bscurr_e_gen,darea)/1e6  # 1/e6 is to MA
    bscurr_i_gen=array(snetcdf.variables['bscurr_i_gen'])  # [A/cm^2]
    I_bs_i_gen= np.dot(bscurr_i_gen,darea)/1e6  # 1/e6 is to MA
    bscurr_e_maxw=array(snetcdf.variables['bscurr_e_maxw'])  # [A/cm^2]
    I_bs_e_maxw= np.dot(bscurr_e_maxw,darea)/1e6  # 1/e6 is to MA
    bscurr_i_maxw=array(snetcdf.variables['bscurr_i_maxw'])  # [A/cm^2]
    I_bs_i_maxw= np.dot(bscurr_i_maxw,darea)/1e6  # 1/e6 is to MA
    I_bs_combined.extend(I_bs_e_gen+I_bs_i_maxw)
    currpar_starnue=array(snetcdf.variables['currpar_starnue'])  # [A/cm^2]
    I_starnue= np.dot(currpar_starnue,darea)/1e6  # 1/e6 is to MA
    I_starnue_combined.extend(I_starnue)
    currpar_starnue0=array(snetcdf.variables['currpar_starnue0'])  # [A/cm^2]
    I_starnue0= np.dot(currpar_starnue0,darea)/1e6  # 1/e6 is to MA
    I_starnue0_combined.extend(I_starnue0)
       
    #---- Additional current [MA] (can be negative):
    I_add_ifile= (I_bs_e_gen+I_bs_i_maxw) + (I_starnue-I_starnue0)  # 
    I_dsE_bs_combined.extend(I_add_ifile) # MA    
    #--- Now E field (matrix)
    elecfld= array(snetcdf.variables['elecfld'])  # V/cm   (nstop+1,lrz+1)
    # evaluated at bin centers (use rho) !  Parallel Electric Field
    elecfld=elecfld*100. # CONVERT to V/m
    elecfld_combined.extend(elecfld[:,1:lrz+1])  # V/m, And [0] is omitted
    #print 'curra  : ',curra.shape   #'(nstop+1,lrz)'
    #print 'elecfld: ',elecfld.shape #'(nstop+1,lrz+1)'
    # Now current density
    curr=array(snetcdf.variables['curr']) #shape: (nstop+1,lrz)
    curr= np.absolute(curr) # Sometimes it is <0 (neg.tor.dir). Plot |curr|
    curr_combined.extend(curr)  # A/cm^2
    currv=array(snetcdf.variables['currv']) #shape: (lrz,jx)
    print 'currv', currv.shape   #(lrz,jx) if netcdfshort='disabled'
    # or (lrz,jx,nstop+1) if netcdfshort='long_jp'
    zeff_ifile=array(snetcdf.variables['zeff']) #shape: (nstop+1,lrz)
    zeff_r_combined.extend(zeff_ifile[:,:]) # 2nd index is for lr

time_combined=   np.asarray(time_combined)    # 1D array
Ip_combined=     np.asarray(Ip_combined)      # 1D array
I_bs_combined=   np.asarray(I_bs_combined)
I_starnue_combined= np.asarray(I_starnue_combined)
I_starnue0_combined= np.asarray(I_starnue0_combined)

elecfld_combined=np.asarray(elecfld_combined) # 2D array 
curr_combined=   np.asarray(curr_combined)    # 2D array
zeff_r_combined=np.asarray(zeff_r_combined) # 2D array (nstop+1,lrz)
time_combined_min= np.min(time_combined)
time_combined_max= np.max(time_combined)
print 'time_combined=',shape(time_combined),time_combined_min,time_combined_max  #sec
print 'Ip_combined=',  shape(Ip_combined),np.min(Ip_combined),np.max(Ip_combined)  #kA
print 'elecfld_combined=',  shape(elecfld_combined),np.min(elecfld_combined),np.max(elecfld_combined)  #V/m
print 'curr_combined=',  shape(curr_combined),np.min(curr_combined),np.max(curr_combined)  #A/cm^2
print 'zeff_r_combined=',  shape(zeff_r_combined),np.min(zeff_r_combined),np.max(zeff_r_combined) 
nt_combined= time_combined.size


# Adjust t_low_lim, to be less than max of timecode:
t_combined_low_lim= min(t_combined_low_lim, time_combined_max*0.99)
# Adjust t_combined_low_lim, to be not less than min of time_combined:
t_combined_low_lim= max(t_combined_low_lim, time_combined_min)
print 'Adjusted t_combined_low_lim to ', t_combined_low_lim

# Find indexes it_combined_select corresponding to t_combined_select[] values
# in the time_combined array (with some accuracy)
t_combined_select=np.asarray(t_combined_select)
it_combined_select=[]
for it in range(0,t_combined_select.size): # scan the list
    t_combined_select1= t_combined_select[it] # a given slice
    tdiff= np.abs(time_combined[:]-t_combined_select1)
    it_found=[] # reset
    it_found= np.min(np.where(tdiff==np.min(tdiff)))  # What if cannot find?
    it_found= np.asscalar(it_found)
    print it_found
    it_combined_select.append(it_found)
it_combined_select=np.asarray(it_combined_select)
print 't_combined_select ',t_combined_select
print 'it_combined_select',it_combined_select
print 'time_combined[it_combined_select]',time_combined[it_combined_select]


# Read data file (*.nc) :   
    
print 'The input file contains:'
print '========================================'
print "The global attributes: ",s_file_cql3d.dimensions.keys()        
print "File contains variables: ",s_file_cql3d.variables.keys()
print '========================================'


# FSA Parallel current (A/cm^2)  as a func of (time,rho)
# Note: curr array may not exist in *.nc, so try to read it:
try:
    try:
        curr=array(s_file_cql3d.variables['curr'])  # j_par(time,lr)
        # also could plot 'curtor' (time,lr)
    except:
        print('No data on curr')
        i_curr=0
    else:
        i_curr=1
        #curr=array(s_file_cql3d.variables['curr'])   # j_par(time,lr)
        print 'curr:', curr.shape
finally:
    print '----------------------------------------'  


# Note: nstates may not exist in *.nc (added in 2019-09), so try to read it:
try:
    try:
        nstates=s_file_cql3d.variables['nstates'].getValue() # scalar
        nstates=np.asscalar(nstates)
    except:
        print('No data on nstates')
        nstates=0
    else:
        print 'nstates=', nstates
finally:
    print '----------------------------------------'  

unorm=s_file_cql3d.variables['vnorm'].getValue()  #getValue() for scalar
unorm=np.asscalar(unorm)
unorm2=unorm**2
unorm3=unorm*unorm2
unorm4=unorm2*unorm2

renorm= 1e-12*unorm3  
# In CQL3D, 'f' is norm-ed so that 
# INTEGRAL  [f_cql3d  * x^2 dx * 2pi*sin(pitch)dpitch] = n[cm^-3]
# We want to change 'f' into f_SI so that
# INTEGRAL  [f_SI  * u[m/s]^2 du * 2pi*sin(pitch)dpitch] = n[m^-3]
# This is achieved by rearranging:
# (f_cql3d/unorm[cm/s]^3) *1e6*(u[cm/s]/100)^2 * d(u[cm/s]/100) gives n[cm^-3]
# and another 1e6 to have n[m^-3].
# Then, we divide f_cql3d by unorm[cm/s]^3 and multiply by 1e12 (cm^-3-->m^-3)
# So that f_SI= f_cql3d/renorm

#  'bctshift' May not be available in *.nc files from older runs
bctshift=s_file_cql3d.variables['bctshift'].getValue()  #getValue() for scalar
print 'bctshift=',bctshift  # 'Time shift of bctime(), for restarts' [sec]

#rfpwr=array(s_file_cql3d.variables['pwrrf'])
#print 'rfpwr:', rfpwr.shape

rmag=s_file_cql3d.variables['rmag'].getValue()
rmag=np.asscalar(rmag)
print 'Rmag[cm]=',rmag

btor=s_file_cql3d.variables['btor'].getValue()
btor=np.asscalar(btor)
print 'Nominal tor mag fld at radmaj btor[Gauss]=',btor

rgeomp=s_file_cql3d.variables['rgeomp'].getValue()
rgeomp=np.asscalar(rgeomp)
print '0.5*(max-min) of major radius   rgeomp[cm]=' , rgeomp

restp=array(s_file_cql3d.variables['restp']) # [nt+1,lrz+1]
restp=np.asarray(restp)
#print restp.shape # [nt+1,lrz+1]
print 'RESISTIVITY <E_phi/R>/<j_phi/R> [cgs] restp(t=0, all ir):', restp[1,:]

sptzrp=array(s_file_cql3d.variables['sptzrp']) # [nt+1,lrz+1]
sptzrp=np.asarray(sptzrp)
# Spitzer resistivity, incl Zeff dependence [cgs: seconds]

rya=array(s_file_cql3d.variables['rya'])
# Normalized radial mesh at bin centers !
rya=np.asarray(rya)
print 'rya:', rya


bnumb=array(s_file_cql3d.variables['bnumb'])
# Atomic charge 
bnumb=np.asarray(bnumb)
print 'Atomic charge bnumb:', bnumb

fmass=array(s_file_cql3d.variables['fmass'])
# mass [gram] 
fmass=np.asarray(fmass)
print 'fmass [gram]:', fmass

ntotal= fmass.size  # Number of species, including general and Maxw.
print 'Number of species: ntotal=', ntotal
ngen= s_file_cql3d.variables['ngen'].getValue()
ngen= np.asscalar(ngen)
print 'ngen=',ngen

Rp=array(s_file_cql3d.variables['Rp']) #[cm] Major rad. of surface at outerboard
Rp=np.asarray(Rp)
#print 'Rp:', Rp

Rm=array(s_file_cql3d.variables['Rm']) #[cm] Major rad. of surface at innerboard
Rm=np.asarray(Rm)
#print 'Rm:', Rm
rminor=(Rp-Rm)/2  # minor radius, cm
print 'rminor[0:lrz-1] ==(Rp-Rm)/2 [cm] =', rminor

taueeh=array(s_file_cql3d.variables['taueeh']) #'Hinton-Hazeltine(Eq5.4)-ONETWO-taue'
taueeh=np.asarray(taueeh)
print 'shape(taueeh)',shape(taueeh) # (nt,lrz)

timecode= array(s_file_cql3d.variables['time']) +bctshift  #added the shift[sec]

print 'timecode.shape=',timecode.shape
#print 'timecode=',timecode

# CONVERT to msec
timecode=timecode*1e3
nt= timecode.size
print 'Number of time steps  nt=',nt
# Adjust t_low_lim, to be less than max of timecode:
t_low_lim= min(t_low_lim, np.max(timecode)*0.99)
# Adjust t_low_lim, to be not less than min of timecode:
t_low_lim= max(t_low_lim, np.min(timecode))
print 'Adjusted t_low_lim to ', t_low_lim
# Find time index corresponding to t_low_lim:
it0= np.min(np.where(timecode>=t_low_lim) )
print 'it0 that corresponds to t_low_lim:' , it0
print 'timecode[it0]=', timecode[it0]


elecfld= array(s_file_cql3d.variables['elecfld'])  # V/cm
# evaluated at bin centers (use rho) !  Parallel Electric Field
# CONVERT to V/m:
elecfld=elecfld*100. # V/m
print 'elecfld: ',elecfld.shape,'(nstop+1,lrz+1)'

ipellet=0 # to be changed below, if pellet='enabled' in nc file
if nstates>0:
    gamafac=s_file_cql3d.variables['gamafac'][:] # Character
    #gamafac= gamafac[:].tostring()
    print 'gamafac=', gamafac
    pellet=s_file_cql3d.variables['pellet'][:] # Character
    #pellet= pellet[:].tostring()
    print 'pellet=', pellet
    if pellet[0]== "e":  
        ipellet=1
    else:
        ipellet=0
    print 'pellet=', pellet, ipellet
    
    imp_type=s_file_cql3d.variables['imp_type'].getValue()
    imp_type=np.asscalar(imp_type)
    print 'imp_type=', imp_type
    fmass_imp=s_file_cql3d.variables['fmass_imp'].getValue()  #getValue() for scalar
    fmass_imp=np.asscalar(fmass_imp)
    print 'Impurity atom fmass_imp [gram]:', fmass_imp
    print 'Impurity atom fmass_imp/proton:', fmass_imp/proton
    # Atomic charge for each state
    bnumb_imp=array(s_file_cql3d.variables['bnumb_imp'])
    bnumb_imp=np.asarray(bnumb_imp)  # 0:nstates
    print 'Atomic charge for each state bnumb_imp:', bnumb_imp
    # Related to pellet, if any:
    pellet_Cablation=s_file_cql3d.variables['pellet_Cablation'].getValue()
    pellet_Cablation=np.asscalar(pellet_Cablation)
    print 'pellet_Cablation=', pellet_Cablation
    pellet_M0=s_file_cql3d.variables['pellet_M0'].getValue()
    pellet_M0=np.asscalar(pellet_M0)
    print 'pellet_M0 [gram]=', pellet_M0
    # rho(t) for pellet:
    pellet_rho=array(s_file_cql3d.variables['pellet_rho'])    
    pellet_rho=np.asarray(pellet_rho)  # 0:nt
    print 'pellet_rho:', pellet_rho.shape
    # Gablation(t) for pellet [gram/sec]:
    Gablation=array(s_file_cql3d.variables['Gablation'])    
    Gablation=np.asarray(Gablation)  # 0:nt
    print 'Gablation:', Gablation.shape
    # Remaining mass(t) for pellet:
    pellet_Mrem=array(s_file_cql3d.variables['pellet_Mrem'])    
    pellet_Mrem=np.asarray(pellet_Mrem)  # 0:nt
    print 'pellet_Mrem[gram]:MIN/MAX', np.min(pellet_Mrem), np.max(pellet_Mrem)
    # Find time index corresponding to instant when pellet_Mrem~0:
    it_Mgone_arr= np.where(pellet_Mrem/pellet_M0<1.e-5) 
    print 'it_Mgone_arr=',it_Mgone_arr
    if np.size(it_Mgone_arr)==0:
        print ' No time point satisfying pellet_Mrem/pellet_M0<1.e-5'
        print ' which means pellet got to inner side.'
        print ' Setting it_Mgone to nt-1'
        it_Mgone=nt-1
    else:
        it_Mgone= np.min(it_Mgone_arr)
    print 'it_Mgone=',it_Mgone,' nt=',nt,' pellet_Mrem[it_Mgone]=',pellet_Mrem[it_Mgone]

    dens_imp_allstates=array(s_file_cql3d.variables['dens_imp_allstates']) 
    # 'Density of impurity, all charge states together'   '1/cm**3'
    print 'dens_imp_allstates: ',dens_imp_allstates.shape #shape: (nstop+1,lrz)
    dens_imp=array(s_file_cql3d.variables['dens_imp']) 
    # 'Density of impurity, for each charge state (incl.Z=0)'   '1/cm**3'
    print 'dens_imp: ',dens_imp.shape #shape: (nstop+1,lrz,0:nstates)
    dens_imp_max=np.max(dens_imp)
    print 'MAX of dens_imp =', dens_imp_max
    print 'MIN of dens_imp =', np.min(dens_imp)
    
    #----- Ne_bound INVENTORY 
    #[SUM(Nb(kstate)*density(rho,kstate))*dvol(rho) for each rho and t]
    kstate_max= nstates
    nb= dens_imp_allstates*0 # initialize shape: [ir,it]
    for kstate in range(0,kstate_max+1): # loop in charge states
        nb= nb +dens_imp[:,0:lrz,kstate]*(nstates-bnumb_imp[kstate]) 
        #SUM over n(Zstate)*(Zatom-Zstate)
        # nb # density of bound e
    print 'nb: ',nb.shape #shape: (nstop+1,lrz)

if i_ampf==1:
    elecfldn= array(s_file_cql3d.variables['elecfldn'])  # statVolts/cm
    # evaluated at bin centers !!!
    print 'elecfldn: ',elecfldn.shape,'(niter,nstop+1,lrz+2)'
    # CONVERT to V/m:
    elecfldn=elecfldn*300.*100. # V/m
    niter= elecfldn[:,0,0].size  # Number of iterations
    print 'Number of saved iterations(including it=0)  niter=',niter 
    elecfld_min= np.min(elecfldn)
    elecfld_max= np.max(elecfldn)
    print 'min/max of elecfldn for all r, t, iterations:',elecfld_min,elecfld_max

# When nt is too large (say, 500), it is better to omit some points.
ntstride= np.floor(nt/nt_plot) # rint()= nearest integer, floor()=nearest-lower
# the above (nt/20) means: leave only ~20 points for plotting.
# ntstride=5  # Or set explicitly (the stride in time index)
ntstride= int(max(ntstride,1)) # To make sure it is >0 



# runaway-related arrays
if i_ra==1:
    runaway_rate=array(s_file_cql3d.variables['runaway_rate']) #shape: (nstop+1,lrz)
    # 'Runaway rate, determined from e flux off grid'
    # 'Runaway rate = 1/n * dn/dt / nu_Kulsrud'
    # 'Unitless'
    print 'runaway_rate: ',runaway_rate.shape
    denra=array(s_file_cql3d.variables['denra']) #shape: (nstop+1,lrz)
    # 'Runaway FSA density above ucrit'   '1/cm**3'
    print 'denra: ',denra.shape
    curra=array(s_file_cql3d.variables['curra']) #shape: (nstop+1,lrz)
    #curra= np.absolute(curra) # Sometimes it is <0 (neg.tor.dir). Plot |curra|
    # 'Runaway FSA parallel cur density above ucrit'   'Amps/cm**2'
    print 'curra: ',curra.shape
    ucrit=array(s_file_cql3d.variables['ucrit']) #shape: (nstop+1,lrz)
    # 'Critical momentum per mass for runaway'   'Normalized to vnorm'
    print 'ucrit: ',ucrit.shape
    edreicer=array(s_file_cql3d.variables['edreicer']) #shape: (nstop+1,lrzmax)
    # 'E_D Dreicer elec fld, e.g., Kulsrud PRL(1973)'   'Volts/cm'
    print 'edreicer: ',edreicer.shape
    # CONVERT to V/m:
    edreicer=edreicer*100. # V/m

if i_ko==1:
    srckotot=array(s_file_cql3d.variables['srckotot']) #shape: (nstop+1,lrz)
    # 'FSA Knockon source density rate'   '#/cm**3*sec'
    print 'srckotot: ',srckotot.shape
    eoe0=array(s_file_cql3d.variables['eoe0']) #shape: (nstop+1,lrz)
    # 'Elecfld/Critical knockon electric field'   'no units'
    print 'eoe0: ',eoe0.shape
    denfl=array(s_file_cql3d.variables['denfl']) #shape: (nstop+1,lrz)
    # 'FSA Elec Density from KO Reduced Distn'   '#/cm**3'
    print 'denfl: ',denfl.shape
if i_dbb==1:
    dbb2=array(s_file_cql3d.variables['dbb2']) #shape: (nstop+1,lrz)
    # (deltaB/B)^2 from data files
    print 'dbb2: ',dbb2.shape


# density, temp, <energy> of all species, as a func. of t and rho 
density=array(s_file_cql3d.variables['density']) #shape: (nstop+1,lrz,ntotal)
print 'density [1/cm**3]: ',density.shape    #    '1/cm**3'
temp=array(s_file_cql3d.variables['temp']) #shape: (nstop+1,lrz,ntotal)
print 'temp [keV]: ',temp.shape    #    'keV'
energy=array(s_file_cql3d.variables['energy']) #shape: (nstop+1,lrz,ntotal)
print 'energy [keV]: ',energy.shape    # 'keV'   'FSA Energy per particle'
#ntotal= density[0,0,:].size  # Number of species, including general and Maxw.
#print 'Number of species: ntotal=', ntotal
#  Also, Zeff
zeff=array(s_file_cql3d.variables['zeff']) #shape: (nstop+1,lrz)
print 'zeff: ',zeff.shape    # Zeff


ccurtor=array(s_file_cql3d.variables['ccurtor'])  # [Amps] 
# CONVERT to kA:
#c     ccurtor(lr_) is the cumulative toroidal current, integrating
#c                   curtor(lr) in poloidal cross-section (Amps),
#c                   accounting for pol variation of tor current.
#c                   See tddiag
ccurtor=ccurtor/1000. # kA
print 'ccurtor:', ccurtor.shape  # Shape is same as for curr,
# i.e. size= (nstop+1,lrz)
#The value of ccurtor[itime,lrz] corresponds to the total integral of current:
Itor=ccurtor[:,lrz-1] # [kA]   as a function of time.
#print 'Itor[kA] as a func of time:', Itor
#print Itor.shape, timecode.shape

bscurr_e_gen=array(s_file_cql3d.variables['bscurr_e_gen'])  # [A/cm^2]
I_bs_e_gen= np.dot(bscurr_e_gen,darea)/1e6  # 1/e6 is to MA
bscurr_i_gen=array(s_file_cql3d.variables['bscurr_i_gen'])  # [A/cm^2]
I_bs_i_gen= np.dot(bscurr_i_gen,darea)/1e6  # 1/e6 is to MA
bscurr_e_maxw=array(s_file_cql3d.variables['bscurr_e_maxw'])  # [A/cm^2]
I_bs_e_maxw= np.dot(bscurr_e_maxw,darea)/1e6  # 1/e6 is to MA
bscurr_i_maxw=array(s_file_cql3d.variables['bscurr_i_maxw'])  # [A/cm^2]
I_bs_i_maxw= np.dot(bscurr_i_maxw,darea)/1e6  # 1/e6 is to MA

currpar_starnue=array(s_file_cql3d.variables['currpar_starnue'])  # [A/cm^2]
I_starnue= np.dot(currpar_starnue,darea)/1e6  # 1/e6 is to MA
currpar_starnue0=array(s_file_cql3d.variables['currpar_starnue0'])  # [A/cm^2]
I_starnue0= np.dot(currpar_starnue0,darea)/1e6  # 1/e6 is to MA

#---------- Plot Itor vs time
# Add a plot of  A*exp(-t/tau_decay) curves.
# Factor A will be taken from the peak value of Itor(time) curve.
A=0.
it_Imax=1
isign=1.0 # To reverse all currents (e.g., for CMOD data, use -1.0)
for itime in range(0,nt): # loop in time index
    if abs(Itor[itime])>=abs(A):
        A=Itor[itime]
        it_Imax=itime
print ' Max or Min of Itor is at time step itime=it_Imax=',it_Imax
fig0=plt.figure(0) 
ax = plt.subplot(111)
txt= ' $Thin-Black:curr,$ $Red:curra,$ '+\
     r"$Blue: \delta \sigma E$" 
plt.xlabel('$time$  $(msec)$')
plt.ylabel('$I$ $(MA)$ $ bold/black:$ $all$ $together$')
plt.title(txt,y=1.03)
plt.grid(True)
plt.hold(True) 
plt.minorticks_on() # To add minor ticks
plt.tick_params(which='both',  width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4, color='k')
xlim((np.min(timecode),np.max(timecode)*1.05))
ylim((0.0, 2.0)) # ylim((0.0, 1.3)) # in MA
# Flatten I for the initial few time steps, where Amp-Far was not turned on yet
it_flat= 0 #10-1 # use nonampf value here
for it in range(0,it_flat):
    Itor[it]=Itor[it_flat]
plot(timecode, isign*(Itor)/1e3, '-', color='k', linewidth=1) #1/e3 to MA
# This line is very close to Itor:
#plot(timecode, isign*np.dot(curr,darea)/1e6 ,'g', linewidth=1) # 1/e6 is to MA

#plot(timecode, isign*I_starnue , 'r.', linewidth=0.5) 
#plot(timecode, isign*I_starnue0 ,'m.', linewidth=0.5)
plot(timecode, isign*(I_starnue-I_starnue0),'b--', linewidth=1) #= delta_sigma*Ephi
plot(timecode, isign*(I_bs_e_gen+I_bs_i_maxw) , 'g--', linewidth=1) # Bootstrap
I_all=(Itor)/1e3 + (I_bs_e_gen+I_bs_i_maxw) + (I_starnue-I_starnue0)
plot(timecode, isign*I_all, 'o-', color='k', linewidth=3) #1/e3 to MA

# Add RE current, if available:
if i_ra==1:
    I_RE= np.dot(curra,darea)/1e6  # 1/e6 is to MA
    print 'min/max of I_RE [MA]', np.min(I_RE), np.max(I_RE)
    plot(timecode, isign*I_RE , 'r', linewidth=1)
    txt= r"$I_{RE}(t_{end})=$" + r"%1.2e" %(I_RE_combined[nt-1]) +"$MA$"
    ntt=max(1,nt-130)
    xpos_txt= timecode[ntt]
    ypos_txt= isign*I_RE[ntt]+0.2 # in MA
    #plt.text(xpos_txt, ypos_txt, txt, fontsize=fnt,color='r')
# Add a plots of  A*exp(-t/tau_decay) curves. 
#(Start from time step =it_Imax where Itor reaches Max value A)
if iplot_tau_decay==1:
    t00= timecode[it_Imax]
    plot(timecode[it_Imax:nt], A*exp(-(timecode[it_Imax:nt]-t00)/tau_decay_0),color='r',linewidth=1)
    plot(timecode[it_Imax:nt], A*exp(-(timecode[it_Imax:nt]-t00)/tau_decay_a),color='k',linewidth=1)
    plot(timecode[it_Imax:nt], A*exp(-(timecode[it_Imax:nt]-t00)/tau_decay_par),color='g',linewidth=1)
savefig('Itor_time'+'.png')
show() #--------------------------------------------------------------------------
it_st=0
print 'timecode[it_st]=',timecode[it_st], 'ms'
print 'Itor[it_st]=',  Itor[it_st]*1e-3, 'MA'
print 'I_all[it_st]=', I_all[it_st], 'MA'
n1=320+200 # For NIMROD+CQL runs: 2.0ms [2022]
n1=200+800 # [2024-05 runs with Ar SPI] 200+800-->1ms+1.6ms
print 't1 [ms]  =', timecode[n1]
print 'tend [ms]=', timecode[nt-1]
print 'I_RE[t1] =', I_RE[n1], 'MA'
print 'I_RE[end]=', I_RE[nt-1], 'MA'
print '------------------------------------------------'
print 'I_RE[end]/Itor[it_st]=', I_RE[nt-1]/(Itor[it_st]*1e-3)
print 'Itor[end]/Itor[it_st]=', Itor[nt-1]/(Itor[it_st])
print '------------------------------------------------'
print 'I_RE[end] /I_all[it_st]=', I_RE[nt-1]/(I_all[it_st])
print 'I_all[end]/I_all[it_st]=', I_all[nt-1]/(I_all[it_st])
print '------------------------------------------------'
#stop
#-------------------------------------------------------------------------------
fig0=plt.figure(0) 
ax = plt.subplot(111)
txt= '$Black:I_{ALL},$ $Blue:curr[I_{code}],$ $Red:curra$'
txt= '$Black:I_{ALL},$ $Blue:curr[I_{code}]$ $and$ $\delta \sigma E$'
plt.xlabel('$time$  $(ms)$')
if isign<0:
    plt.ylabel('$-I$  $(MA)$')
else:
    plt.ylabel('$I$  $(MA)$')
#plt.title(txt,y=1.045)
plt.grid(True)
plt.hold(True) 
plt.minorticks_on() # To add minor ticks
plt.tick_params(which='both',  width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4, color='k')
plt.xlim(( t_combined_low_lim*1e3, time_combined_max*1e3 )) #msec
#plt.xlim((650.,10+time_combined_max*1e3))
plt.ylim((0.0,1.6)) # MA #--------- Adjust limits ----------
plot(time_combined*1e3,isign*I_starnue_combined, 'k--', linewidth=1.0) 
plot(time_combined*1e3,isign*I_starnue0_combined,'b--', linewidth=1.0)
#plot(time_combined*1e3,isign*(I_starnue_combined-I_starnue0_combined),'b--', linewidth=1) #= delta_sigma*Ephi
plot(time_combined*1e3,isign*I_bs_combined,'g-',linewidth=1) # Bootstrap
plot(time_combined*1e3,isign*(Ip_combined/1e3),'-',color='b',linewidth=1) #1/e3 to MA
#plot(time_combined*1e3,isign*(Ip_combined/1e3 -I_starnue0_combined),'--',color='c',linewidth=1) #1/e3 to MA
print 'min/max I_starnue0_combined', np.min(I_starnue0_combined),np.max(I_starnue0_combined)
print 'min/max I_starnue_combined', np.min(I_starnue_combined),np.max(I_starnue_combined)
print 'min/max Ip_combined [MA]', np.min(Ip_combined/1e3),np.max(Ip_combined/1e3)

#Add I_dsE_bs_combined to form total current I_All :
I_All_combined= (Ip_combined)/1e3 + I_dsE_bs_combined  # MA
plot(time_combined*1e3, np.abs(I_All_combined),'-',color='k',linewidth=2) # MA
# Add RE current, if available:
if i_ra==1:
    print 'min/max of I_RE [MA]', np.min(I_RE_combined), np.max(I_RE_combined)
    plot(time_combined*1e3, np.abs(I_RE_combined), 'r', linewidth=1)
    
#--- Also plot Zeff(time); with axis and labels at right-edge of plot
ax2= ax.twinx()
for lr in range(0,lrz,1):
    ax2.plot(time_combined*1e3,zeff_r_combined[:,lr],'-',color='m',linewidth=2)
plt.ylim((0, 10.)) #--------- Adjust limits ----------
plt.ylabel('$Z_{eff}$',color='m') # make same color as the plotted line 
plt.minorticks_on() # To add minor ticks
plt.tick_params(which='both',  width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4, color='k')
plt.xlim(( t_combined_low_lim*1e3, time_combined_max*1e3 )) # msec
#plt.xlim((650.,10+time_combined_max*1e3))

savefig('Itor_time_combined'+'.png')
savefig('Itor_time_combined'+'.eps')
show() #--------------------------------------------------------------------------
#stop


if i_ampf==1:
    #---------- Plot E(r=0) vs time
    itera=niter-1 # Plot Efld for this iteration (or select other <niter)
    # itera can be 0,1,...,niter-1  (niter-1 is equal to nampfmax in CQL3D)
    ir=1 # ir=0 or 1 corresponds to r~0 (plasma core)
    Epk=np.transpose(elecfldn[itera,:,ir])    # Efld[time] at rho=0
    E_rho_t=elecfldn[itera,:,:] # as 2D array (rho,t)
    print 'shape of E_rho_t', np.shape(E_rho_t)  # [ntime,lrz]
    Emean_vs_t=np.mean(E_rho_t[:,1:],axis=1) #func of time. (averaged over rho at each t)
    print 'shape of Emean_vs_t', np.shape(Emean_vs_t)  #
    Emin_vs_t=np.min(E_rho_t[:,1:],axis=1) #func of time. (MIN over rho at each t)
    Emax_vs_t=np.max(E_rho_t[:,1:],axis=1) #func of time. (MAX over rho at each t)

    fig0=plt.figure(1) 
    ax = plt.subplot(111)
    txt= "$-o-E(r=0);$  $Green:MEAN;$ $Blue/Red:MIN/MAX$   $(V/m)$"  
    plt.xlabel('$time$  $(msec)$')
    plt.ylabel('$(V/m)$')
    plt.title(txt,y=1.02)
    plt.grid(True)
    plt.hold(True) 
    plot(timecode, Epk, 'o', color='k')
    plot(timecode, Emean_vs_t, '-', color='g', linewidth=2)
    plot(timecode, Emin_vs_t,  '-', color='b', linewidth=2)
    plot(timecode, Emax_vs_t,  '-', color='r', linewidth=2)
    # Add a plots of  A*exp(-t/tau_decay) curves. 
    #(Start from time step =it_Emax where Epk reaches Max value A)
    if (iplot_tau_decay==1):
        # Add a plot of  A*exp(-t/tau_decay) curves.
        # Factor A will be taken from the peak value of Epk(time) curve.
        A=0.
        it_Emax=1
        for itime in range(0,nt): # loop in time index
            if abs(Epk[itime])>=abs(A):
                A=Epk[itime]
                it_Emax=itime
        print ' Max or Min of Epk is at time step itime=it_Emax=',it_Emax
        it_Emax=it_Imax+1 # Better use the time step corresponding to peak of I(time)
        it_Emax=min(it_Emax,nt-1)
        A=Epk[it_Emax]
        t00= timecode[it_Emax] # msec
        plot(timecode[it_Emax:nt], A*exp(-(timecode[it_Emax:nt]-t00)/tau_decay_0),color='r',linewidth=1)
        plot(timecode[it_Emax:nt], A*exp(-(timecode[it_Emax:nt]-t00)/tau_decay_a),color='k',linewidth=1)
        plot(timecode[it_Emax:nt], A*exp(-(timecode[it_Emax:nt]-t00)/tau_decay_par),color='g',linewidth=1)
    savefig('E_time'+'.png')
    show() #--------------------------------------------------------------------------



# Problem: rho was absent in nc file. 
#rho=np.arange(0,1.0001,1.0/(lrz-1))
rho=rya  # 'rya' is saved from index 1 to lrz in CQL3D/netcdfrw2.f
# Limits for plots:
rho_min= 0.0 #min(rho)
rho_max= 1.0 #max(rho)
#  For mesh-type plots: 
time_adj=timecode[it0:nt:ntstride] # msec
R,T = np.meshgrid(time_adj,rho)  # 2D grids
R_combined,T_combined = np.meshgrid(time_combined,rho) # 2D grids for combined t
print 'min of time_adj grid:', np.min(time_adj)

zdir = (None) # direction for plotting text (title)
xdir = (None) # direction
ydir = (None) # direction

#  Mesh-type plots: Jpar[rho,time] A/cm^2
Jpar=np.transpose(curr[it0:nt:ntstride,:])    # Jpar[rho,time]
Jpar=np.abs(Jpar) # Sometimes it is negative. Plot |Jpar|
print T.shape, R.shape, Jpar.shape
Jpar_min=np.min(Jpar)
Jpar_max=np.max(Jpar)
print 'min/max of |Jpar| (A/cm^2):', np.min(Jpar),np.max(Jpar)
A=Jpar
fig0=plt.figure(3)
if imesh==1:
    ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
    ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
    #ax.set_zscale('log') #------------- LOG10 SCALE --------------
    #ax.set_zlim3d(A_max/1e4, A_max)
    ax.set_xlim3d(0.,1.0)
    ax.set_ylim3d(t_low_lim,timecode[nt-1] )
    zdir = (None) # direction for plotting text (title)
    xdir = (None) # direction
    ydir = (None) # direction
    ax.set_ylabel(r"$time$  $(msec)$", xdir) 
    ax.set_xlabel(r"$\rho$", ydir) 
else: # imesh=0
    plt.axis([0.,1.1, t_low_lim, timecode[nt-1]])
    #levels=np.arange(0.,300.,10) # set values, for detailed zoom-in
    #CS=plt.contour(T,(R),(A),levels,linewidth=linw,cmap=plt.cm.jet)
    CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
    CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
    if nstates>0:
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
    plt.ylabel(r"$time$  $(msec)$", xdir) 
    plt.xlabel(r"$\rho$", ydir)
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
plt.title('$|J_{||}|$  $(A/cm^2)$',y=1.02)
savefig('Jpar_2D'+'.png')
plt.show()        


# Jpar(rho) plots for each time instant,  all together. 


# Define line thickness, different for different time slices
linw_mx= 9*linw # largest thickness
linw_mn= 0.5    # smallest
itt=0
for itime in range(0,nt,ntstride): # loop in time index
    itt=itt+1
linw_reducing=zeros((itt)) # initialize
if itt>1:
    dlinw= (linw_mx-linw_mn)/(itt-1)
else:
    dlinw=0
itt=0
for itime in range(0,nt,ntstride): # loop in time index
    linw_reducing[itt]=linw_mx-itt*dlinw
    itt=itt+1



W=np.abs(curr[:,:]) # [t,r]

fig0=plt.figure(4) 
ax = plt.subplot(111)
W_max=np.max(W) 
W_min=np.min(W)
xlim((0.,rho_max))
ylim_min=min(0.,W_min/2)
ylim_max=W_max*1.05
ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
txt= "$|J_{||}|$  $(A/cm^2)$"   
plt.xlabel(r'$\rho$', fontsize=34)
plt.title(txt,y=1.02)
plt.grid(True)
plt.hold(True) 
icount=0
for it in it_select:
    if it<nt:
        W1=W[it,:]
        for ir in range(0,lrz): # loop in rho index
            W1[ir]= max(W1[ir], ylim_min/10) # all r: impose lower limit
        linww= 4*linw-icount # Start with bold line, then reduce
        linww=max(linww,0.75) # line thickness: not lower than 0.75pt
        plot(rho,W1,linewidth=linww,color=col_select_it[icount])
        txt1= r"$it=$"+r"$%3i$" %(it+1) # 
        txt2= r" $t[ms]=$"+r"$%1.3f$" %(timecode[it])
        xpos_txt=rya[lrz-1]
        ypos_txt=W1[lrz-1]
        #plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
        #color=col_select_it[icount]
        q0=(btor/rmag)/(0.2*3.14*W1[0]) #W1[0] is current dens at m.axis[A/cm^2]
        print 'J vs rho: Added it,time[ms]=',it,timecode[it],\
        ' Max=',np.max(W1),' q0=',q0
        #plt.semilogy(rya,W1,linewidth=linww,color=col_select_it[icount])
        plot(rya,W1,linewidth=linww,color=col_select_it[icount])
        icount=icount+1
savefig('Jpar_selected_it'+'.png')
show()


fig0=plt.figure(5) 
ax = plt.subplot(111)
#------- Plot different types of current density -------
print 'shape(curr)=',shape(curr), ' nt=',nt, ' nt_combined=',nt_combined
j_code=zeros((lrz))
#print shape(j_code), lrz
j_code= isign*curr[nt-1,:] # [t,r]
j_bs=   isign*( bscurr_e_gen[nt-1,:] + bscurr_i_maxw[nt-1,:] )
j_ohm0= isign*currpar_starnue0[nt-1,:]
j_ohm=  isign*currpar_starnue[nt-1,:]
j_re=   isign*curra[nt-1,:]
j_cd=   j_code-j_ohm0
j_all=  j_code + j_bs + (j_ohm-j_ohm0)

W_max=np.max(j_all) 
W_min=min(np.min(j_all), np.min(j_bs))
xlim((0.,rho_max))
ylim_min=min(0.,W_min)
ylim_max=W_max*1.05
ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
txt= "$-J_{||}$  $(A/cm^2)$"   
plt.xlabel(r'$\rho$', fontsize=32)
plt.title(txt,y=1.04)
plt.grid(True)
plt.hold(True) 

plot(rho,j_all, '-',color='k',linewidth=3)
plot(rho,j_code,'-',color='b',linewidth=1)
plot(rho,j_re,  '-',color='r',linewidth=1)
plot(rho,j_ohm0,'--',color='b',linewidth=1)
plot(rho,j_cd,  '--',color='c',linewidth=1)
plot(rho,j_bs,  '-',color='g',linewidth=1)
plot(rho,(j_ohm-j_ohm0),  '--',color='b',linewidth=1)

savefig('Jpar_nstop'+'.png')
show()

#stop



# E(rho) plots for selected it_combined over combined time_combined axis
W=np.abs(curr_combined[:,:]) # [t,r]
print 'shape of W=A=E',shape(W)
fig0=plt.figure(190) # # Jpar(rho) plots for selected it
ax = plt.subplot(111)
W_max=np.max(W) 
W_min=np.min(W)
xlim((0.,rho_max))
ylim_min= min(W_min,0)
ylim_max= W_max*1.05
ylim((ylim_min, ylim_max)) 
txt= "$|J_{||}|$  $(A/cm^2)$"   
plt.xlabel(r'$\rho$', fontsize=34)
#plt.ylabel('$A/cm^2$')
plt.title(txt,y=1.02)
plt.grid(True)
plt.hold(True) 
icount=0
for it in it_combined_select:
    if it<nt_combined:
        W1=W[it,:]
        for ir in range(0,lrz): # loop in rho index
            W1[ir]= max(W1[ir], ylim_min/10) # all r: impose lower limit
        linww= 4*linw-icount # Start with bold line, then reduce
        linww=max(linww,0.75) # line thickness: not lower than 0.75pt
        plot(rho,W1,linewidth=linww,color=col_t_combined_select[icount])
        txt1= r"$it=$"+r"$%3i$" %(it+1) # 
        txt2= r"$%1.3f$" %(time_combined[it]) +"$s$"
        ir_txt=0 #icount+np.mod(icount,3)
        xpos_txt=rya[ir_txt]
        ypos_txt=W1[ir_txt]
        plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                 color=col_t_combined_select[icount] )
        icount=icount+1
savefig('Jpar_combined_selected_it'+'.png')
show()


fig0=plt.figure(14) 
ax = plt.subplot(111)
Jpar_min=np.min(Jpar)
Jpar_max=np.max(Jpar)*1.1
ax.axis([0.,rho_max, 0.,Jpar_max])
txt= "$|J_{||}|$  $(A/cm^2)$"   
plt.xlabel(r'$\rho$', fontsize=34)
plt.title(txt,y=1.02)
plt.grid(True)
plt.hold(True) 
icount=0
itt=0
for itime in range(0,nt,ntstride): # loop in time index
    #print 'time step=',itime,':'
    # Plot with different thickness of lines, depending on iteration
    if remainder(itt,6)==0: col='b'
    if remainder(itt,6)==1: col='g'
    if remainder(itt,6)==2: col='r'
    if remainder(itt,6)==3: col='c'    
    if remainder(itt,6)==4: col='m' 
    if remainder(itt,6)==5: col='k'  
    # start with thick line, then - gradually reducing thickness
    Jpara=np.abs(curr[itime,:])
    #print np.shape(rho), np.shape(Jpara)
    plot(rho, Jpara, '-',color=col, linewidth=linw_reducing[itt])
    itt=itt+1
savefig('Jpar_All_nt'+'.png')
show()


#--------------------------------------------------------------------------

if i_ampf==1:
    #  Mesh-type plots: Efld[rho,time]
    itera=niter-1 # Plot Efld for this iteration (or select other <niter)
    # itera can be 0,1,...,niter-1  (niter-1 is equal to nampfmax in CQL3D)
    print 'PLOTTING Efld for itera=', itera
    Efld=np.transpose(elecfldn[itera,it0:nt:ntstride,1:lrz+1])    # Efld[rho,time] # V/m
    print T.shape, R.shape, Efld.shape
    Efld_min=np.min(Efld)
    Efld_max=np.max(Efld)
    #print np.min(Efld),np.max(Efld)

    fig0=plt.figure(15)
    ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
    ax.plot_wireframe(T,(R),(Efld),rstride=200,cstride=1,cmap=cm.jet)
    ax.set_xlim3d(0.,1.0)
    ax.set_ylim3d(t_low_lim,timecode[nt-1] )
    ax.set_zlim3d(min(Efld_min,0.), max(Efld_max,0.))
    zdir = (None) # direction for plotting text (title)
    xdir = (None) # direction
    ydir = (None) # direction
    ax.set_ylabel(r"$time$  $(msec)$", xdir) 
    ax.set_xlabel(r"$\rho$", ydir) 
    #ax.set_zlabel('$E_{tor}$  $(V/m)$', zdir)
    plt.title('$E_{tor}$  $(V/m)$',y=1.02)
    savefig('Efld_rho_2D'+'.png')
    plt.show()

    # Efld(rho) plots for each time instant,  all together. 
    fig0=plt.figure(17) 
    ax = plt.subplot(111)
    el_min= np.min(elecfldn[itera,it0:nt:ntstride,:])
    el_max= np.max(elecfldn[itera,it0:nt:ntstride,:])
    ax.axis([0.,rho_max, el_min, el_max])
    txt= "$E_{tor}$ $(V/m)$"   
    plt.xlabel(r'$\rho$', fontsize=34)
    #plt.ylabel('$E_{tor}$  $(V/m)$')
    plt.title(txt,y=1.02)
    plt.grid(True)
    plt.hold(True) 
        
    itt=0
    for itime in range(it0,nt,ntstride): # loop in time index
        print 'time step=',itime,':', timecode[itime],'ms'
        # Plot with different thickness of lines, depending on iteration
        if remainder(itt,6)==0: col='b'
        if remainder(itt,6)==1: col='g'
        if remainder(itt,6)==2: col='r'
        if remainder(itt,6)==3: col='c'    
        if remainder(itt,6)==4: col='m' 
        if remainder(itt,6)==5: col='k'  
        itera=niter-1 # for the last iteration only.
        # start with thick line, then - gradually reducing thickness
        plot(rho, elecfldn[itera,itime,1:lrz+1], '-',color=col, linewidth=linw_reducing[itt])
        itt=itt+1
    savefig('Efld_rho_All_nt'+'.png')
    #--------------------------------------------------------------------------


    # Plot and Save Efld(rho) plots for each time instant. 
    # All iterations are shown on one plot.
    if each_step==1:
        for itime in range(0,nt,ntstride): # loop in time index
            print 'time step=',itime,':'
            fig0=plt.figure(20+itime) 
            ax = plt.subplot(111)
            el_min=min(0,elecfld_min)*1.1
            el_max=max(0,elecfld_max)*1.1
            #ax.axis([0.,rho_max, el_min,el_max]) # COMMENT if you want auto-limits
            txt= r"$E_{tor}$ $(V/m)$  $at$ $t=$" + r"%1.6e" %(timecode[itime]) +"$msec$" 
            plt.xlabel(r'$\rho$')
            plt.ylabel('$E_{tor}$  $(V/m)$')
            plt.title(txt,y=1.02)
            plt.grid(True)
            plt.hold(True) 
            # Plot with different thickness of lines, depending on iteration
            linw_mx= 9*linw # largest thickness
            linw_mn= 0.5    # smallest
            if niter>1:
                dlinw= (linw_mx-linw_mn)/(niter-1)
            else:
                dlinw=0
            for itera in range(0,niter): # loop in iterations for a given time step
                if remainder(itera,6)==0: col='b'
                if remainder(itera,6)==1: col='g'
                if remainder(itera,6)==2: col='r'
                if remainder(itera,6)==3: col='c'    
                if remainder(itera,6)==4: col='m' 
                if remainder(itera,6)==5: col='k'  
                plot(rho, elecfldn[itera,itime,1:lrz+1], 'o-', color=col, linewidth=linw_mx-itera*dlinw)
                print 'elecfldn',itime,itera,np.min(elecfldn[itera,itime,:]),np.max(elecfldn[itera,itime,:])
            #---------
            plot(rho,elecfld[itime,1:lrz+1],'r',linewidth=linw) #Added 'elecfld' profile
            #print 'rho.shape=',rho.shape,' elecfld.shape[itime,1:lrz+1]=',elecfld[itime,1:lrz+1].shape
            # In CQL3D: elecfld is saved as (0:lrz) array,
            #    rho==rya is saved from index 1 to lrz in netcdfrw2.f 
            #---------
            savefig('Efld_rho'+'_itime_'+str(itime)+'.png')
        show() #--------------------------------------------------------------------------



# elecfld() is saved in any case:
#--------------
A=np.transpose(elecfld[it0:nt:ntstride,1:lrz+1]) #to V/m; [0] point is omitted
#A=np.abs(A) # Plot |E|. Better E (can change sign as a func of time)
#print T.shape, R.shape, ' E (V/m), elecfld:', A.shape
A_min=np.min(A)
A_max=np.max(A)
fig0=plt.figure(190)
if imesh==1:
    ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
    ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
    #ax.set_zscale('log') #------------- LOG10 SCALE --------------
    #ax.set_zlim3d(A_max/1e4, A_max)
    ax.set_xlim3d(0.,1.0)
    ax.set_ylim3d(t_low_lim,timecode[nt-1] )
    zdir = (None) # direction for plotting text (title)
    xdir = (None) # direction
    ydir = (None) # direction
    ax.set_ylabel(r"$time$  $(msec)$", xdir) 
    ax.set_xlabel(r"$\rho$", ydir) 
else: # imesh=0
    plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
    print 'A_min,A_max=',A_min,A_max
    if A_max>A_min:
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
    if nstates>0:
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
    plt.ylabel(r"$time$  $(msec)$", xdir) 
    plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
plt.title(r'$E$ $(V/m)$',y=1.02)
print '---------- max of E(V/m)=', A_max
it_Amax= np.min(np.where(A==A_max)) 
print 'it_Amax=', it_Amax, '  t[ms]=',timecode[it_Amax]
#print 'lr_Amax=', lr_Amax, '  rya=',rya[lr_Amax]
savefig('E_rho_2D'+'.png')
plt.show()        


#-------------- E Combined over all files (time_combined axis) ------
#print 'shapes of T_combined,R_combined,elecfld_combined',\
#shape(T_combined),shape(R_combined),shape(elecfld_combined)
A=np.transpose(elecfld_combined) #to V/m; [0]is omitted
#A=np.abs(A) # Plot |E|. Better E (can change sign as a func of time)
#print T.shape, R.shape, ' E (V/m), elecfld_combined:', A.shape
A_min=np.min(A)
A_max=np.max(A)
fig0=plt.figure(190)
if imesh==1:
    ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
    ax.plot_wireframe(T_combined,(R_combined),(A),rstride=200,cstride=1,cmap=cm.jet)
    #ax.set_zscale('log') #------------- LOG10 SCALE --------------
    #ax.set_zlim3d(A_max/1e4, A_max)
    ax.set_xlim3d(0.,1.0)
    ax.set_ylim3d(t_combined_low_lim,time_combined_max)
    zdir = (None) # direction for plotting text (title)
    xdir = (None) # direction
    ydir = (None) # direction
    ax.set_ylabel(r"$time$  $(sec)$", xdir) 
    ax.set_xlabel(r"$\rho$", ydir) 
else: # imesh=0
    plt.axis([0.,1.1, t_combined_low_lim, time_combined_max])
    if A_max>A_min:
        CS=plt.contour(T_combined,(R_combined),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
    if nstates>0:
        plt.plot(pellet_rho[1:it_Mgone],time_combined[1:it_Mgone],color='r',linewidth=linw*2)
        plt.plot(pellet_rho[1:it_Mgone],time_combined[1:it_Mgone],'k.')
    plt.ylabel(r"$time$  $(sec)$", xdir) 
    plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
plt.title(r'$E$ $(V/m)$',y=1.02)
savefig('E_2D_combined'+'.png')
plt.show()        



# E(rho) plots for selected it 
W=np.transpose(elecfld[:,1:lrz+1]) #converted to V/m # [r,it], [0] omitted
print 'shape of W=A=E',shape(W)
fig0=plt.figure(190) # # E(rho) plots for selected it
ax = plt.subplot(111)
W_max=np.max(W) 
W_min=np.min(W)
if W_max==W_min:
    W_max=W_max+0.1
    W_min=W_min-0.1
xlim((0.,rho_max))
ylim_min= W_min-0.1 #min(0.,W_min/2)
ylim_max= W_max+0.1
ylim((ylim_min, ylim_max))  
txt= "$E$  $(V/m)$"   
plt.xlabel(r'$\rho$', fontsize=34)
plt.ylabel('$V/m$')
plt.title(txt,y=1.02)
plt.grid(True)
plt.hold(True) 
icount=0
for it in it_select:
    if it<nt:
        W1=W[:,it]
        #for ir in range(0,lrz): # loop in rho index
        #    W1[ir]= max(W1[ir], ylim_min/10) # all r: impose lower limit
        linww= 4*linw-icount # Start with bold line, then reduce
        linww=max(linww,0.75) # line thickness: not lower than 0.75pt
        plot(rho,W1,linewidth=linww,color=col_select_it[icount])
        txt1= r"$it=$"+r"$%3i$" %(it+1) # 
        txt2= r"$%1.1f$" %(timecode[it]) +"$ms$"
        lr_mid= np.floor(lrz/2)
        xpos_txt=rya[lr_mid]
        ypos_txt=W1[lr_mid]
        plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
        color=col_select_it[icount])
        xpos_txt=rya[0]
        ypos_txt=W1[0]
        plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
        color=col_select_it[icount])
        print 'E vs rho: Added it,time[ms]=',it,timecode[it],\
        ' Max=',np.max(W1)
        #plt.semilogy(rya,W1,linewidth=linww,color=col_select_it[icount])
        plot(rya,W1,linewidth=linww,color=col_select_it[icount])
        icount=icount+1
savefig('E_selected_it'+'.png')
show()


# E(rho) plots for selected it_combined over combined time_combined axis
W=isign*np.transpose(elecfld_combined) #converted to V/m # [r,it], [0] omitted
print 'shape of W=A=E',shape(W)
fig0=plt.figure(190) # # E(rho) plots for selected it
ax = plt.subplot(111)
W_max=np.max(W) 
W_min=np.min(W)
if W_max==W_min:
    W_max=W_max+0.01
    W_min=W_min-0.01
xlim((0.,rho_max))
ylim_min= W_min-0.01 #min(0.,W_min/2)
ylim_max= W_max+0.01
ylim((ylim_min, ylim_max)) 
txt= "$E$  $(V/m)$"   
if isign<0:
    txt= "$-E$  $(V/m)$"  
plt.xlabel(r'$\rho$', fontsize=34)
plt.ylabel('$V/m$')
plt.title(txt,y=1.02)
plt.grid(True)
plt.hold(True) 
icount=0
for it in it_combined_select:
    if it<nt_combined:
        W1=W[:,it]
        #for ir in range(0,lrz): # loop in rho index
        #    W1[ir]= max(W1[ir], ylim_min/10) # all r: impose lower limit
        linww= 4*linw-icount # Start with bold line, then reduce
        linww=max(linww,0.75) # line thickness: not lower than 0.75pt
        plot(rho,W1,linewidth=linww,color=col_t_combined_select[icount])
        txt1= r"$it=$"+r"$%3i$" %(it+1) # 
        txt2= r"$%1.3f$" %(time_combined[it]) +"$s$"
        ir_txt=np.mod(2.5*icount,lrz-1)
        xpos_txt=rya[ir_txt]
        ypos_txt=W1[ir_txt]
        plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                 color=col_t_combined_select[icount] )
        #print 'E vs rho: Added it,time[s]=',it,time_combined[it],\
        #' Max=',np.max(W1)
        icount=icount+1
savefig('E_combined_selected_it'+'.png')
show()
#stop


#--------------
# add extra room along horizontal axis in plots with ir_select :
dtw= 0.2*(np.max(timecode)- np.min(timecode))
fig0=plt.figure(191)
plt.hold(True)
plt.grid(True)
C= np.transpose((elecfld[:,1:lrz+1]))    # A[rho,time] ALL TIME STEPS  [V/m]
C_max=np.max(C) 
C_min=np.min(C)
print 'min/max of elecfld:' , C_min, C_max
plt.xlabel('$time$ $(msec)$')
xlim((np.min(timecode), np.max(timecode)+dtw  ))
ylim_min=C_min-0.1 #/1e5
ylim_max=C_max+0.1 #*2
ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
icount=0
C1=timecode*0 
for ir in ir_select:  
    if ir<lrz:
        it_pk=1075
        t_pk=timecode[it_pk] #timecode[nt-1]
        C1_pk=0.
        for it in range(0,nt): # loop in time index
            C1[it]= C[ir,it] #max(C[ir,it], ylim_min/10) # all t: impose lower limit
            #if C1[it]>=C1_pk:
            #    t_pk=timecode[it]
            #    C1_pk=C1[it]
        linww= 4*linw-icount # Start with bold line, then reduce
        linww=max(linww,0.75) # line thickness: not lower than 0.75pt
        plt.plot(timecode,C1,linewidth=linww,color=col_select[icount])
        txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
        txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
        #it_txt= nt-lrz+ir #int(nt/2)
        xpos_txt= t_pk
        ypos_txt= C1_pk
        xpos_txt=timecode[it_pk]
        ypos_txt=C1[it_pk]
        if ypos_txt>ylim_min :
            plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
            color=col_select[icount])
        #print 'E_vs_time:  Added ir=',ir,rya[ir],' Min/Max=',np.min(C1),np.max(C1)
        icount=icount+1
plt.title(' $E$  $(V/m)$',y=1.03)
savefig('E_vs_time'+'.png')
plt.show()    
#--------------


#--------------
# add extra room along horizontal axis in plots with ir_select :
dtw_combined= 0.2*(time_combined_max - t_combined_low_lim)
fig0=plt.figure(191)
plt.hold(True)
plt.grid(True)
C= np.transpose(np.abs(elecfld_combined)) # A[rho,time] ALL TIME STEPS  [V/m]
C_max=np.max(C) 
C_min=np.min(C)
plt.xlabel('$time$ $(ms)$')
xlim((t_combined_low_lim*1e3, (time_combined_max+dtw_combined)*1e3  ))
ylim_min=C_max/1e4
ylim_max=C_max*2
ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
icount=0
C1=time_combined*0 
for ir in ir_select:  
    if ir<lrz:
        t_pk=time_combined[nt_combined-1]
        C1_pk=0.
        for it in range(0,nt_combined): # loop in time index
            C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
            if C1[it]>=C1_pk:
                t_pk=time_combined[it]
                C1_pk=C1[it]
        linww= 4*linw-icount # Start with bold line, then reduce
        linww=max(linww,0.75) # line thickness: not lower than 0.75pt
        plt.semilogy(time_combined*1e3,C1,linewidth=linww,color=col_select[icount])
        txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
        txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
        xpos_txt= t_pk*1e3
        ypos_txt= C1_pk
        xpos_txt= (time_combined[nt_combined-1])*1e3
        ypos_txt=C1[nt_combined-1]
        if ypos_txt>ylim_min :
            plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
            color=col_select[icount])
        #print 'E_vs_time:  Added ir=',ir,rya[ir],' Min/Max=',np.min(C1),np.max(C1)
        icount=icount+1
plt.title(' $|E|$  $(V/m)$',y=1.03)
savefig('E_vs_time_combined'+'.png')
plt.show()    
#--------------

#stop

#-------------- Resistivity from restp[] array # == E/j  [cgs: sec] 
A_min=np.min(restp)
A_max=np.max(restp)
print 'MIN/MAX of restp [sec] original:', A_min,A_max
# Sometimes restp has negative values (numerical instability?)
# Make adjustment
A_min= np.min(sptzrp)/100 # set lower limit, based on Spitzer resistivity 
A_max= np.max(sptzrp)*100 # set upper limit, based on Spitzer resistivity 
for it in range(0,len(restp[0,:])):
    for ir in range(0,len(restp[:,0])):
        restp[ir,it]=max(restp[ir,it],A_min)
        restp[ir,it]=min(restp[ir,it],A_max)
A=np.transpose(restp[it0:nt:ntstride,0:lrz]) # == E/j  [cgs: sec]  
A_min=np.min(A)
A_max=np.max(A)
print 'MIN/MAX of restp [sec] after adjustment:', A_min,A_max
fig0=plt.figure(193)
if imesh==1:
    ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
    ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
    #ax.set_zscale('log') #------------- LOG10 SCALE --------------
    #ax.set_zlim3d(A_max/1e4, A_max)
    ax.set_xlim3d(0.,1.0)
    ax.set_ylim3d(t_low_lim,timecode[nt-1] )
    zdir = (None) # direction for plotting text (title)
    xdir = (None) # direction
    ydir = (None) # direction
    ax.set_ylabel(r"$time$  $(msec)$", xdir) 
    ax.set_xlabel(r"$\rho$", ydir) 
else: # imesh=0
    plt.axis([0.,1.1, t_low_lim, timecode[nt-1]])
    CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
    CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
    if nstates>0:
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
    plt.ylabel(r"$time$  $(msec)$", xdir) 
    plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(r'$resistivity$ $from$ $E/j$  $(cgs: sec)$',y=1.02)
savefig('restp_rho_2D'+'.png')
plt.show()        

#-------------- Resistivity from sptzrp[] array
A=np.transpose(sptzrp[it0:nt:ntstride,0:lrz]) # from Spitzer formula [cgs: sec]  
A_min=np.min(A)
A_max=np.max(A)
print 'MIN/MAX of sptzrp [sec]:', A_min,A_max
fig0=plt.figure(194)
if imesh==1:
    ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
    ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
    #ax.set_zscale('log') #------------- LOG10 SCALE --------------
    #ax.set_zlim3d(A_max/1e4, A_max)
    ax.set_xlim3d(0.,1.0)
    ax.set_ylim3d(t_low_lim,timecode[nt-1] )
    zdir = (None) # direction for plotting text (title)
    xdir = (None) # direction
    ydir = (None) # direction
    ax.set_ylabel(r"$time$  $(msec)$", xdir) 
    ax.set_xlabel(r"$\rho$", ydir) 
elif A_max>A_min: # imesh=0
    plt.axis([0.,1.1, t_low_lim, timecode[nt-1]])
    CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
    CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
    if nstates>0:
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
    plt.ylabel(r"$time$  $(msec)$", xdir) 
    plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(r'$Spitzer$ $resistivity$  $(cgs: sec)$',y=1.02)
savefig('sptzrp_rho_2D'+'.png')
plt.show()   



#-------------- tau for Decay of current, based on E/j resistivity
lam1= 2.4/(0.1*rgeomp)  #[1/cm] 0.1*rgeomp. Or maybe rminor[]=(Rp-Rm)/2 ???
tau_decay_restp= 1e3*(4*pi/restp/clight**2)/lam1**2 # 1e3 is to CONVERT to msec
tau_decay_sptz= 1e3*(4*pi/sptzrp/clight**2)/lam1**2 # 1e3 is to CONVERT to msec
#print 'shape of tau_decay_restp', np.shape(tau_decay_restp) #[nt+1; lrz+1]
A= np.transpose(tau_decay_restp[it0:nt:ntstride,0:lrz]) #  [msec]  
A_min=np.min(A)
A_max=np.max(A)
print 'MIN/MAX of tau_decay_restp [msec]:', A_min,A_max
fig0=plt.figure(195)
if imesh==1:
    ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
    ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
    #ax.set_zscale('log') #------------- LOG10 SCALE --------------
    #ax.set_zlim3d(A_max/1e4, A_max)
    ax.set_xlim3d(0.,1.0)
    ax.set_ylim3d(t_low_lim,timecode[nt-1] )
    zdir = (None) # direction for plotting text (title)
    xdir = (None) # direction
    ydir = (None) # direction
    ax.set_ylabel(r"$time$  $(msec)$", xdir) 
    ax.set_xlabel(r"$\rho$", ydir) 
else: # imesh=0
    plt.axis([0.,1.1, t_low_lim, timecode[nt-1]])
    CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
    CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
    if nstates>0:
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
    plt.ylabel(r"$time$  $(msec)$", xdir) 
    plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(r'$\tau_{decay}$ $for$ $current$  $(msec)$',y=1.02)
savefig('tau_decay_current_2D'+'.png')
plt.show()   

#-------------- 1/tau for Decay of current, based on E/j resistivity
B= np.transpose(tau_decay_restp[it0:nt:ntstride,0:lrz]) #  [1/msec]  
A= 1/B
A_min=np.min(A)
A_max=np.max(A)
print 'MIN/MAX of 1/tau_decay_restp [1/msec]:', A_min,A_max
fig0=plt.figure(196)
if imesh==1:
    ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
    ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
    #ax.set_zscale('log') #------------- LOG10 SCALE --------------
    #ax.set_zlim3d(A_max/1e4, A_max)
    ax.set_xlim3d(0.,1.0)
    ax.set_ylim3d(t_low_lim,timecode[nt-1] )
    zdir = (None) # direction for plotting text (title)
    xdir = (None) # direction
    ydir = (None) # direction
    ax.set_ylabel(r"$time$  $(msec)$", xdir) 
    ax.set_xlabel(r"$\rho$", ydir) 
else: # imesh=0
    plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
    CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
    CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
    if nstates>0:
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
    plt.ylabel(r"$time$  $(msec)$", xdir) 
    plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(r'$1/\tau_{decay}$ $for$ $current$  $(1/msec)$',y=1.02)
savefig('one_tau_decay_current_2D'+'.png')
plt.show()   



if i_ra==1:   # RA-related
    # Mesh plots over(rho,time)
    A=np.abs(np.transpose(curra[it0:nt:ntstride,0:lrz]))    # curra==A[rho,time] 
    curra_adj= A # Save this
    A_max=400. #np.max(A)
    A_min=0.1 # A_max/1e5 #/1e15
    print 'curra_adj min/max=', np.min(curra_adj),np.max(curra_adj)
    for it in range(0,len(A[0,:])):
        for ir in range(0,len(A[:,0])):
            A[ir,it]=max(A[ir,it],A_min)
    #A=np.log10(A)
    A_min=np.min(A)
    A_max=np.max(A)
    
    fig0=plt.figure(201)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),log10(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contourf(T,(R),log10(A),20,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=1.0, format='%.1f')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('$log10(curra)$  $(A/cm^2)$',y=1.02)
    savefig('curra_rho_2D'+'.png')
    plt.show()    
    #--------------

    A=np.transpose(denra[it0:nt:ntstride,0:lrz])    # denra==A[rho,time] 
    denra_adj=A # Save this
    A_max=np.max(A)
    A_min= A_max/1e15
    for it in range(0,len(A[0,:])):
        for ir in range(0,len(A[:,0])):
            A[ir,it]=max(A[ir,it],A_min)
    #A=np.log10(A)
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(202)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),log10(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contourf(T,(R),log10(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('$log10(denra)$  $(1/cm^3)$',y=1.02)
    savefig('denra_rho_2D'+'.png')
    plt.show()    
    #--------------

    A=np.transpose(ucrit[it0:nt:ntstride,0:lrz])    # ucrit[rho,time]/vnorm 
    temp_e=np.transpose(temp[it0:nt:ntstride,0:lrz,0])    # Te[rho,time] 
    vth_e=sqrt(temp_e*ergtkev/fmass[0])  #[cm/s] Thermal v of electrons
    A=A*unorm/vth_e  # Now ucrit/Vth_e
    print T.shape, R.shape, ' ucrit.shape:',A.shape
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(203)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        ax.set_zlim3d(0.,10.0)
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    elif A_max>A_min:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        levels2=np.arange(0.0, 40., 40./Ncont)
        CS=plt.contour(T,(R),(A),levels2,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%1.1f')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('$Ucrit/V_{th,e}$',y=1.02)
    savefig('ucrit_rho_2D'+'.png')
    plt.show()    
    #--------------
    
    A=np.transpose(runaway_rate[it0:nt:ntstride,0:lrz])    # A[rho,time] 
    #print T.shape, R.shape, A.shape
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(204)
    ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
    ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
    ax.set_xlim3d(0.,1.0)
    ax.set_ylim3d(t_low_lim,timecode[nt-1] )
    ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
    zdir = (None) # direction for plotting text (title)
    xdir = (None) # direction
    ydir = (None) # direction
    ax.set_ylabel(r"$time$  $(msec)$", xdir) 
    ax.set_xlabel(r"$\rho$", ydir) 
    #ax.set_zlabel('$ $  $ $', rotation=-90)
    plt.title(r'$runaway$ $rate$  $(1/n)(dn/dt)/$$\nu_{Kulsrud}$',y=1.02)
    savefig('runaway_rate_rho_2D'+'.png')
    plt.show()    
    #--------------
    A=np.transpose(edreicer[it0:nt:ntstride,0:lrz]) # A[rho,time] # converted to V/m
    print T.shape, R.shape, ' Edreicer.shape:',A.shape
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(205)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    elif A_max>A_min:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(r'$E_{Dreicer}$ $(V/m)$',y=1.02)
    savefig('edreicer_rho_2D'+'.png')
    plt.show()    
    #--------------
    E2D=np.transpose(elecfld[it0:nt:ntstride,1:lrz+1]) # conv to V/m, [0]omitted
    print T.shape, R.shape, ' E2D:', E2D.shape
    B=np.absolute(E2D/A) # here A[rho,time] is from the above : Edreicer
    A= B #np.log10(B)    
    print T.shape, R.shape, ' Efld/Edreicer:', A.shape
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(206)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(r'$E/E_{Dreicer}$',y=1.02)
    print '---------- max of E/Edreicer=', A_max
    savefig('E_Edreicer_rho_2D'+'.png')
    plt.show()    
    #--------------
    # Plots of a func.vs.t will be made for these     
    # radial indexes (up to 12 radial points)
    #ir_select=[0,1,2,3,4,5,10,15,20,25,30,35] 
    #col_select=['r','b','g','m','c','k', 'r','b','g','m','c','k']
    fig0=plt.figure(211)
    plt.hold(True)
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    C= np.abs(np.transpose(curra[:,0:lrz]))    # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) #*2 # give some extra space
    C_min=np.min(C)
    print 'min/max of curra:' , C_min, C_max
    plt.xlabel('$time$ $(msec)$')
    xlim((np.min(timecode), np.max(timecode)+dtw  ))
    ylim_min=C_max/1e15 #/1e17
    ylim_max=C_max*10
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            it_pk=1075
            t_pk=timecode[it_pk]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            if np.min(C1)>0:
                plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
                txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
                txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
                #it_txt= nt-lrz+ir #int(nt/2)
                xpos_txt= t_pk
                ypos_txt= C1_pk
                xpos_txt=timecode[it_pk]
                ypos_txt=C1[it_pk]
                if ypos_txt>ylim_min :
                    plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                    color=col_select[icount])
                #print 'curra_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
                icount=icount+1
    plt.title(' $RA$ $current$ $density$ $(A/cm^2)$',y=1.03)
    savefig('curra_vs_time'+'.png')
    #savefig('curra_vs_time'+'.eps')
    plt.show()    
    #--------------
    fig0=plt.figure(213)
    plt.hold(True)
    plt.grid(True)
    C= np.transpose(np.abs(curr[:,0:lrz]))    # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) #*2 # give some extra space
    C_min=np.min(C)
    print 'min/max of curr:' , C_min, C_max
    plt.xlabel('$time$ $(msec)$')
    xlim((np.min(timecode), np.max(timecode)+dtw  ))
    ylim_min=C_max/1e4
    ylim_max=C_max*2
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            it_pk=1075
            t_pk=timecode[it_pk]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            #it_txt= nt-lrz+ir #int(nt/2)
            xpos_txt= t_pk
            ypos_txt= C1_pk
            xpos_txt=timecode[it_pk]
            ypos_txt=C1[it_pk]
            if ypos_txt>ylim_min :
                plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                color=col_select[icount])
            #print 'curr_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
            icount=icount+1
    plt.title('$current$ $density$ $(A/cm^2)$',y=1.03)
    savefig('curr_vs_time'+'.png')
    #savefig('curr_vs_time'+'.eps')
    plt.show()    
    #--------------
    fig0=plt.figure(214)
    plt.hold(True)
    plt.grid(True)
    C= np.transpose(denra[:,0:lrz])    # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) #*2 # give some extra space
    C_min=np.min(C)
    print 'min/max of denra:' , C_min, C_max
    plt.xlabel('$time$ $(msec)$')
    xlim((np.min(timecode), np.max(timecode)+dtw  ))
    ylim_min=C_max/1e17
    ylim_max=C_max*10
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    icount=0
    for ir in ir_select:  
        if ir<lrz:
            it_pk=1075   
            t_pk=timecode[it_pk]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10)   # all t steps
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            if np.min(C1)>0:
                plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
                txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
                txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
                it_txt= 5 #int(nt/2)
                xpos_txt= t_pk
                ypos_txt= C1_pk
                xpos_txt=timecode[it_pk]
                ypos_txt=C1[it_pk]
                if ypos_txt>ylim_min :
                    plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                    color=col_select[icount])
                icount=icount+1
                #print 'denra_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
    plt.title(' $RA$ $density$ $(cm^{-3})$',y=1.03)
    savefig('denra_vs_time'+'.png')
    #savefig('denra_vs_time'+'.eps')
    plt.show()    
    #-------------


nfree=np.transpose(density[it0:nt:ntstride,0:lrz,ksp]) # ne,free
# BE SURE THAT ksp corresponds to electrons here !
if nstates>0:
    nbound=np.transpose(nb[it0:nt:ntstride,0:lrz]) # density of bound e
else:
    nbound=nfree*0

if i_ko==1:
    # Mesh plots over(rho,time)
    A=np.transpose(srckotot[it0:nt:ntstride,0:lrz]/density[it0:nt:ntstride,0:lrz,0])    # A[rho,time] 
    #print T.shape, R.shape, A.shape
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(301)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),log10(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        levels_ko= np.arange(1e-5, 10e-5, (11e-5-1e-5)/Ncont)
        CS=plt.contourf(T,(R),log10(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('$S_A/n_e (srckotot/density)$  $(1/s)$',y=1.02)
    savefig('srckotot_rho_2D'+'.png')
    plt.show()    
    #stop
    
    
    #--------------
    A=np.transpose(eoe0[it0:nt:ntstride,0:lrz])    # A[rho,time] 
    # BE SURE THAT ksp corresponds to electrons here !
    # Adjust E0==Ec to Eceff, according to Hesslow et al.
    kappa_hesslow=2. # According to PPCF-2018, it is between 1 and 2
    #Eceff= E0*(nfree+kappa_hesslow*nbound)/nfree # can be >> than E0, when nbound are present
    # Then, our ratio of E/E0 should be adjusted to E/Eceff,
    A=A*(nfree/(nfree+kappa_hesslow*nbound))  # E/Eceff now
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(302)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('$E/E_{c,eff}$',y=1.02)
    print '---------- max of E/Eceff=', A_max
    #savefig('eoe0_rho_2D.png')   # Was E/E0
    savefig('E_Eceff_rho_2D.png') # Now E/Eceff
    plt.show()   
    #stop
    
    #--------------
    A=np.transpose(denfl[it0:nt:ntstride,0:lrz])    # A[rho,time] 
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(303)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('$target(cold)$ $el.density$ $for$ $KO$ $(cm^{-3})$',y=1.02)
    savefig('denfl_rho'+'_2D'+'.png')
    plt.show()    
    #--------------
    if each_step==1:
        # Plot and Save srckotot(rho) plots for each time instant. 
        # All iterations are shown on one plot.
        srckotot_min=np.min(srckotot)
        srckotot_max=np.max(srckotot)
        for itime in range(0,nt,ntstride): # loop in time index
            print 'srckotot:',itime,np.min(srckotot[itime,:]),np.max(srckotot[itime,:])
            fig0=plt.figure(10+itime) 
            ax = plt.subplot(111)
            ra_min=min(0,srckotot_min)*1.1
            ra_max=max(0,srckotot_max)*1.1
            ax.axis([0.,rho_max, ra_min,ra_max]) # COMMENT if you want auto-limits
            txt= r" $KO$ $Src$  $at$ $t=$" + r"%1.6e" %(timecode[itime]) +"$msec$" 
            plt.xlabel(r'$\rho$')
            plt.ylabel(r'$KO$ $Src$  $electrons/cm^3/sec$')
            plt.title(txt,y=1.02)
            plt.grid(True)
            plt.hold(True) 
            #---------
            plot(rho[0:lrz],srckotot[itime,0:lrz],'r',linewidth=linw) 
            # rho=='rya' is saved from index 1 to lrz in CQL3D/netcdfrw2.f
            # and srckotot - also.
            #---------
            savefig('runaway'+'_itime_'+str(itime)+'.png')
        show() #--------------------------------------------------------------------------


#----------- (deltaB/B)^2 from data files
if i_dbb==1:
    # Mesh plots over(rho,time)
    A=np.sqrt(np.transpose(dbb2[it0:nt:ntstride,0:lrz]))  # dB/B == A[rho,time] 
    #print T.shape, R.shape, A.shape
    A_min=np.min(A)
    A_max=np.max(A)
    print 'dbb min/max', A_min,A_max
    fig0=plt.figure(321)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1]*1.05 ])
        CS=plt.contourf(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(r"$\delta B/B$",y=1.02) 
    savefig('dBB_rho'+'_2D'+'.png')
    plt.show()    
    #--------------
#stop

if ipellet>0:
    fig0=plt.figure(333) 
    ax = plt.subplot(311) #-----------------------
    ylim((0, 2))
    plt.grid(True)
    plt.hold(True)
    plt.plot(timecode[1:it_Mgone],pellet_rho[1:it_Mgone],color='r',linewidth=linw*2)
    plt.plot(timecode[1:it_Mgone],pellet_rho[1:it_Mgone],'k.')
    plt.ylabel(r"$\rho$")
    #plt.xlabel(r"$time$  $(ms)$")
    ax = plt.subplot(312) #-----------------------
    plt.plot(timecode[1:it_Mgone],Gablation[1:it_Mgone],color='r',linewidth=linw*2)
    plt.plot(timecode[1:it_Mgone],Gablation[1:it_Mgone],'k.')
    plt.ylabel(r"$G_{abl}$ $(gram/s)$")
    #plt.xlabel(r"$time$  $(ms)$")
    plt.grid(True)
    plt.hold(True) 
    ax = plt.subplot(313) #-----------------------
    plt.plot(timecode[1:it_Mgone],pellet_Mrem[1:it_Mgone]*1e3,color='r',linewidth=linw*2)
    plt.plot(timecode[1:it_Mgone],pellet_Mrem[1:it_Mgone]*1e3,'k.')
    ylim((0, pellet_Mrem[1]*1.05*1e3))   # 1e3 to convert to mgram
    plt.ylabel(r"$M_{pell}$ $(mg)$")
    plt.xlabel(r"$time$  $(ms)$")
    plt.grid(True)
    plt.hold(True) 
    savefig('pellet_ablation_rho_vs_t.png')
    plt.show()
    #--------------------------------------------------------------------------
    
if i_plasma_profiles==1:   # plasma profiles 
    # Mesh plots over(rho,time)
    #ksp=0 # species number, in Python counting (set in the beginning of script)
    # Print this species atomic charge and weight:
    txt1=  '  $Z=$'+r"%3i" %(bnumb[ksp])
    txt2=  '  $m/m_p=$'+r"%1.6f" %(fmass[ksp]/p_mass)  
    #----- density
    A=np.transpose(density[it0:nt:ntstride,0:lrz,ksp])    # A[rho,time] 
    dens_adj=A # Save this
    A_max=np.max(A)
    A_min= A_max/1e15
    for it in range(0,len(A[0,:])):
        for ir in range(0,len(A[:,0])):
            A[ir,it]=max(A[ir,it],A_min)
    #A=np.log10(A)
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(401)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        print 'A_min,A_max=',A_min,A_max
        if A_max>A_min+1e10 :
            CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
            CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('$n$ $(cm^{-3})$'+txt1+txt2,y=1.02)
    savefig('dens_rho_2D_ksp'+str(ksp+1)+'.png') # ksp+1 to match CQL3D count
    plt.show()    
    #----- Ne_free INVENTORY [density(rho)*dvol(rho) for each rho and t]
    print 'min/max dens_adj=', np.min(dens_adj),np.max(dens_adj)
    A=A*0 # clear up; important
    print len(A[:,0]), len(A[0,:])
    for it in range(0,len(A[0,:])):
        #print 'MIN/max dens_adj[it]=', np.min(dens_adj[:,it]),np.max(dens_adj[:,it])
        for ir in range(0,len(A[:,0])):
            A[ir,it]=dens_adj[ir,it]*dvol[ir] # ptcls/cm^3 * cm^3    
    fig0=plt.figure(402)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(r'$N_{e,free}$ $inventory$ $(dn_e(\rho)*dvol(\rho))$',y=1.02)
    savefig('Ne_free_rho_2D_ksp'+str(ksp+1)+'.png') # ksp+1 to match CQL3D count
    plt.show()  
    
    #--------------  T
    A=np.transpose(temp[it0:nt:ntstride,0:lrz,ksp])    # A[rho,time] 
    temp_adj=A # Save this
    A_max=np.max(A)
    A_min= A_max/1e15
    for it in range(0,len(A[0,:])):
        for ir in range(0,len(A[:,0])):
            A[ir,it]=max(A[ir,it],A_min)
    #A=np.log10(A)
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(405)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    elif A_max>A_min:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('$T$ $(keV)$ $for$'+txt1+txt2,y=1.02)
    savefig('temp_rho'+'_2D_ksp'+str(ksp+1)+'.png') # ksp+1 to match CQL3D count
    plt.show()    
    #-------------- <energy>
    A=np.transpose(energy[it0:nt:ntstride,0:lrz,ksp])    # A[rho,time] 
    energy_adj=A # Save this
    A_max=np.max(A)
    A_min= A_max/1e15
    for it in range(0,len(A[0,:])):
        for ir in range(0,len(A[:,0])):
            A[ir,it]=max(A[ir,it],A_min)
    #A=np.log10(A)
    A_min=np.min(A)
    A_max=np.max(A)
    fig0=plt.figure(406)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    elif A_max>A_min:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('$<energy>$ $(keV)$'+txt1+txt2,y=1.02)
    savefig('energy_rho'+'_2D_ksp'+str(ksp+1)+'.png') # ksp+1 to match CQL3D count
    plt.show()        
    #--------------    Zeff
    A=np.transpose(zeff[it0:nt:ntstride,0:lrz])    # A[rho,time] 
    zeff_adj=A # Save this
    A_min=1.0
    for it in range(0,len(A[0,:])):
        for ir in range(0,len(A[:,0])):
            A[ir,it]=max(A[ir,it],A_min)
    #A_min=0 #np.min(A)
    A_max=np.max(A)+1
    print 'A min,max=', A_min, A_max
    print 'Zeff min,max=', np.min(zeff), np.max(zeff)
    #print 'zeff.shape', shape(zeff)
    fig0=plt.figure(407)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    elif np.max(zeff)-np.min(zeff)>0.1:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir)     
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    #ax.set_zlabel('$ $  $ $', rotation=-90)
    plt.title('$Zeff$',y=1.02)
    savefig('zeff_rho_2D'+'.png') 
    plt.show()    
    #--------------
    
    W=temp[:,0:lrz,ksp] # [t,r]
    fig0=plt.figure(502) # # temp(rho) plots for selected it
    ax = plt.subplot(111)
    W_max=np.max(W) 
    W_min=np.min(W)
    xlim((0.,rho_max))
    ylim_min=0. #W_min/2
    ylim_max=W_max*1.05
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    txt= "$T$  $(keV)$"   
    plt.xlabel(r'$\rho$', fontsize=34)
    plt.ylabel('$keV$')
    plt.title(txt,y=1.02)
    plt.grid(True)
    plt.hold(True) 
    icount=0
    for it in it_select:
        if it<nt:
            W1=W[it,:]
            for ir in range(0,lrz): # loop in rho index
                W1[ir]= max(W1[ir], ylim_min/10) # all r: impose lower limit
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plot(rho,W1,linewidth=linww,color=col_select_it[icount])
            txt1= r"$%3i$" %(it) # 
            txt2= r"$%1.3f$" %(timecode[it]) +"$ms$"
            ir_txt=0
            xpos_txt=rya[ir_txt]
            ypos_txt=W1[ir_txt]
            plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
            color=col_select_it[icount])
            print 'T vs rho: Added it,time[ms]=',it,timecode[it]
            #' pellet_rho=',pellet_rho[it]
            #plt.semilogy(rya,W1,linewidth=linww,color=col_select_it[icount])
            plot(rya,W1,linewidth=linww,color=col_select_it[icount])
            icount=icount+1
    savefig('temp_selected_it'+'.png')
    show()

    W=density[:,0:lrz,ksp] # [t,r]
    fig0=plt.figure(502) # # dens(rho) plots for selected it
    ax = plt.subplot(111)
    W_max=np.max(W) 
    W_min=np.min(W)
    xlim((0.,rho_max))
    ylim_min=0. #W_min/2
    ylim_max=W_max*1.05
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    txt= "$n$  $(cm^{-3})$"   
    plt.xlabel(r'$\rho$', fontsize=34)
    plt.ylabel('$cm^{-3}$')
    plt.title(txt,y=1.02)
    plt.grid(True)
    plt.hold(True) 
    icount=0
    for it in it_select:
        if it<nt:
            W1=W[it,:]
            for ir in range(0,lrz): # loop in rho index
                W1[ir]= max(W1[ir], ylim_min/10) # all r: impose lower limit
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plot(rho,W1,linewidth=linww,color=col_select_it[icount])
            txt1= r"$%3i$" %(it) # 
            txt2= r"$%1.2f$" %(timecode[it]) +"$ms$"
            ir_txt=0
            xpos_txt=rya[ir_txt]
            ypos_txt=W1[ir_txt]
            plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
            color=col_select_it[icount])
            print 'n vs rho: Added it,time[ms]=',it,timecode[it]
            #' pellet_rho=',pellet_rho[it]
            #plt.semilogy(rya,W1,linewidth=linww,color=col_select_it[icount])
            plot(rya,W1,linewidth=linww,color=col_select_it[icount])
            icount=icount+1
    savefig('dens_selected_it'+'.png')
    show()

    #stop    
    
    #=================================== Now same, but All nt steps in one plot
    # density(rho) plots for each time instant,  all together. 
    fig0=plt.figure(501) 
    ax = plt.subplot(111)
    A_min=0 #np.min(dens_adj)*0.9
    A_max=np.max(dens_adj)*1.1
    ax.axis([0.,rho_max, A_min,A_max])
    txt= "$density$ $(cm^{-3})$ $For$ $ksp=$"+ r"%3i" %(ksp+1)
    plt.xlabel(r'$\rho$', fontsize=34)
    plt.title(txt,y=1.02)
    plt.grid(True)
    plt.hold(True) 
    itt=0
    for itime in range(0,nt,ntstride): # loop in time index
        #print 'time step=',itime,':'
        # Plot with different thickness of lines, depending on iteration
        if remainder(itt,6)==0: col='b'
        if remainder(itt,6)==1: col='g'
        if remainder(itt,6)==2: col='r'
        if remainder(itt,6)==3: col='c'    
        if remainder(itt,6)==4: col='m' 
        if remainder(itt,6)==5: col='k'  
        # start with thick line, then - gradually reducing thickness
        plot(rho, density[itime,:,ksp], '-',color=col, linewidth=linw_reducing[itt])
        itt=itt+1
    savefig('density_rho'+'_All_nt__ksp'+str(ksp+1)+'.png')
    plt.show()
    #--------------------------------------------------------------------------
    
    # temp(rho) plots for each time instant,  all together. 
    fig0=plt.figure(502) 
    ax = plt.subplot(111)
    A_min=0 
    A_max=np.max(temp_adj)*1.1
    ax.axis([0.,rho_max, A_min,A_max])
    txt= "$T$ $(keV)$ $For$ $ksp=$"+ r"%3i" %(ksp+1)
    plt.xlabel(r'$\rho$', fontsize=34)
    plt.title(txt,y=1.02)
    plt.grid(True)
    plt.hold(True) 
    itt=0
    for itime in range(0,nt,ntstride): # loop in time index
        #print 'time step=',itime,':'
        # Plot with different thickness of lines, depending on iteration
        if remainder(itt,6)==0: col='b'
        if remainder(itt,6)==1: col='g'
        if remainder(itt,6)==2: col='r'
        if remainder(itt,6)==3: col='c'    
        if remainder(itt,6)==4: col='m' 
        if remainder(itt,6)==5: col='k'  
        # start with thick line, then - gradually reducing thickness
        plot(rho, temp[itime,:,ksp], '-',color=col, linewidth=linw_reducing[itt])
        itt=itt+1
    savefig('temp_rho'+'_All_nt__ksp'+str(ksp+1)+'.png')
    plt.show()
    #--------------------------------------------------------------------------
    # energy(rho) plots for each time instant,  all together. 
    fig0=plt.figure(503) 
    ax = plt.subplot(111)
    A_min=0
    A_max=np.max(energy_adj)*1.1
    ax.axis([0.,rho_max, A_min,A_max])
    txt= "$<energy>$ $(keV)$ $For$ $ksp=$"+ r"%3i" %(ksp+1)
    plt.xlabel(r'$\rho$', fontsize=34)
    plt.title(txt,y=1.02)
    plt.grid(True)
    plt.hold(True) 
    itt=0
    for itime in range(0,nt,ntstride): # loop in time index
        #print 'time step=',itime,':'
        # Plot with different thickness of lines, depending on iteration
        if remainder(itt,6)==0: col='b'
        if remainder(itt,6)==1: col='g'
        if remainder(itt,6)==2: col='r'
        if remainder(itt,6)==3: col='c'    
        if remainder(itt,6)==4: col='m' 
        if remainder(itt,6)==5: col='k'  
        # start with thick line, then - gradually reducing thickness
        plot(rho, energy[itime,:,ksp], '-',color=col, linewidth=linw_reducing[itt])
        itt=itt+1
    savefig('energy_rho'+'_All_nt__ksp'+str(ksp+1)+'.png')
    plt.show()
    
    #--------------
    # Plots of a func.vs.t will be made for these     
    # radial indexes (up to 12 radial points)
    #ir_select=[0,1,2,3,4,5,10,15,20,25,30,35] 
    #col_select=['r','b','g','m','c','k', 'r','b','g','m','c','k']
    fig0=plt.figure(510)
    plt.hold(True)
    plt.grid(True)
    C= np.transpose(temp[:,0:lrz,ksp])    # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) 
    C_min=np.min(C)
    plt.xlabel('$time$ $(msec)$')
    #xlim((np.min(timecode), np.max(timecode)+dtw  ))
    xlim=((0.0, 2.5))
    ylim_min=C_min/2
    ylim_max=C_max*2
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')    
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            it_pk=1075
            t_pk=timecode[it_pk]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            xpos_txt= t_pk
            ypos_txt= C1_pk
            xpos_txt=timecode[it_pk]
            ypos_txt=C1[it_pk]
            #if ypos_txt>ylim_min :
            #    plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
            #    color=col_select[icount])
            #print 'temp_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
            icount=icount+1
    plt.title(' $T$ $(keV)$',y=1.03)
    savefig('temp_vs_time'+'.png')
    #savefig('temp_vs_time'+'.eps')
    plt.show()    
    #stop
    #--------------
    fig0=plt.figure(511)
    plt.hold(True)
    plt.grid(True)
    C= np.transpose(density[:,0:lrz,ksp])    # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) 
    C_min=np.min(C)
    plt.xlabel('$time$ $(msec)$')
    #xlim((np.min(timecode), np.max(timecode)+dtw  ))
    xlim=((0.0, 2.5))
    ylim_min=C_min/2
    ylim_max=C_max*2
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')    
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            it_pk=1075
            t_pk=timecode[it_pk]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            xpos_txt= t_pk
            ypos_txt= C1_pk
            xpos_txt=timecode[it_pk]
            ypos_txt=C1[it_pk]
            #if ypos_txt>ylim_min :
            #    plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
            #    color=col_select[icount])
            #print 'dens_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
            icount=icount+1
    plt.title(' $n$ $(cm^{-3})$',y=1.03)
    savefig('dens_vs_time'+'.png')
    #savefig('dens_vs_time'+'.eps')
    plt.show()    
    #--------------
    fig0=plt.figure(512)
    plt.hold(True)
    plt.grid(True)
    C= np.transpose(tau_decay_restp[:,0:lrz])    # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) 
    C_min=np.min(C)
    print 'MIN/MAX tau_decay_restp, all t-steps:', C_min,C_max
    plt.xlabel('$time$ $(msec)$')
    #xlim((np.min(timecode), np.max(timecode)+dtw  ))
    ylim_min=C_min/2
    ylim_max=C_max*2
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            it_pk=1075
            t_pk=timecode[it_pk]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            xpos_txt= t_pk
            ypos_txt= C1_pk
            xpos_txt=timecode[it_pk]
            ypos_txt=C1[it_pk]
            if ypos_txt>ylim_min :
                plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                color=col_select[icount])
            #print 'tau_decay_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
            icount=icount+1
    plt.title(r' $\tau_{decay}$ $for$ $current$  $(msec)$',y=1.03)
    savefig('tau_decay_vs_time'+'.png')
    #savefig('tau_decay_vs_time'+'.eps')
    plt.show()    
    #--------------
    fig0=plt.figure(513)  #  current decay time based on Spitzer resistivity
    plt.hold(True)
    plt.grid(True)
    C= np.transpose(tau_decay_sptz[:,0:lrz])    # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) 
    C_min=np.min(C)
    plt.xlabel('$time$ $(msec)$')
    #xlim((np.min(timecode), np.max(timecode)+dtw  ))
    ylim_min=C_min/2
    ylim_max=C_max*2
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            it_pk=1075
            t_pk=timecode[it_pk]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            xpos_txt= t_pk
            ypos_txt= C1_pk
            xpos_txt=timecode[1] #timecode[nt-1]
            ypos_txt=C1[1] #C1[nt-1]
            if ypos_txt>ylim_min :
                plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                color=col_select[icount])
            #print 'tau_decay_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
            icount=icount+1
    plt.title(r' $\tau_{decay}$ $for$ $current$  $(msec)$  $[Spitzer-based]$',y=1.03)
    savefig('tau_decay_sptz_vs_time'+'.png')
    #savefig('tau_decay_vs_time'+'.eps')
    plt.show()    
    
    #--------------
    fig0=plt.figure(513)  #  'taueeh' 
    plt.hold(True)
    plt.grid(True)
    C= 1e3*np.transpose(taueeh[:,0:lrz])    #at ALL TIME STEPS, converted to ms
    C_max=np.max(C) 
    C_min=np.min(C)
    plt.xlabel('$time$ $(msec)$')
    #xlim((np.min(timecode), np.max(timecode)+dtw  ))
    ylim_min=C_min/2
    ylim_max=C_max*2
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            it_pk=1075
            t_pk=timecode[it_pk]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            xpos_txt= t_pk
            ypos_txt= C1_pk
            xpos_txt=timecode[1] #timecode[nt-1]
            ypos_txt=C1[1] #C1[nt-1]
            if ypos_txt>ylim_min :
                plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                color=col_select[icount])
            icount=icount+1
    plt.title(r' $\tau_{eeh}$ $(msec)$',y=1.03)
    savefig('taueeh_vs_time'+'.png')
    #savefig('taueeh_vs_time'+'.eps')
    plt.show()    
     
    #--------------
    Tkev= np.transpose(temp[:,0:lrz,ksp]) # A[rho,time]
    ncm3= np.transpose(density[:,0:lrz,ksp]) # A[rho,time] ALL TIME STEPS 
    CL=14. # Coulomb log, take from gamaset.
    v_vte_ratio=1.0
    # Use NRL_2016, p.32, nu_s_ee for Fast electrons,
    eps_ev= 1e3*Tkev*0.5*v_vte_ratio**2
    tau_ee= (eps_ev**1.5)/(7.7e-6*CL*ncm3) # tau_ee Scales as v_vte_ratio^3
    Ekev= np.transpose(energy[:,0:lrz,ksp]) # Aver. energy
    tau_decay_sptz_E= np.transpose(tau_decay_sptz)*(1.5*Tkev/Ekev)**1.5
    #print 'shape(tau_ee)',shape(tau_ee)
    #print 'shape(tau_decay_sptz_E)',shape(tau_decay_sptz_E)
    #print np.min(1.5*Tkev/Ekev), np.max(1.5*Tkev/Ekev)
    fig0=plt.figure(513)  #---------------------------------  'tau_slow_down' 
    C= 1e3*tau_ee    #at ALL TIME STEPS, converted to ms
    C_max=np.max(C)*6.0**3 
    C_min=np.min(C)*3.0**3
    plt.xlabel('$time$ $(msec)$')
    #txt3= r" $v/v_{th,e}=$"+r"$%1.1f$" %(v_vte_ratio)
    plt.title(r'$\tau_s^{ee}$ $(msec)$ $for$ $v/v_{th,e}=3(solid)$ $and$ $6(dashed)$',y=1.03)
    #xlim((np.min(timecode), np.max(timecode)+30.0  ))
    ylim_min=C_min/2
    ylim_max=C_max*2
    #ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    plt.hold(True)
    plt.grid(True)
    icount=0
    C1=timecode*0 
    ir_select1=[7,27]
    for ir in ir_select1:  
        if ir<lrz:
            C1[:]= C[ir,:] # all t
            t_pk=timecode[nt-5]
            C1_pk=C1[nt-5]
            T1=Tkev[ir,:]
            #time of T decay (normally negative, but can be positive at later t)
            tau_T= np.abs(T1*np.gradient(timecode)/np.gradient(T1))  #[ms]
            tau_res_E= tau_decay_sptz_E[ir,:]
            print 'rho, tau_T [ms] min/max=', rho[ir], np.min(tau_T),np.max(tau_T)
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.semilogy(timecode,C1*3.0**3,'--',linewidth=linww,color=col_select[icount])
            plt.semilogy(timecode,C1*6.0**3,'-',linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            plt.text(t_pk,C1_pk*3.0**3, txt2, fontsize=fnt,color=col_select[icount])
            #plt.text(t_pk,C1_pk*6.0**3, txt2, fontsize=fnt,color=col_select[icount])
            plt.semilogy(timecode,tau_T,':',linewidth=linww,color=col_select[icount])
            plt.semilogy(timecode,tau_res_E,'-',linewidth=linww,color='k')
            icount=icount+1
    savefig('tauee_vs_time'+'.png')
    #savefig('tauee_vs_time'+'.eps')
    plt.show()    
    #stop



    #--------------
    fig0=plt.figure(514)  
    temp_e= temp[:,0:lrz,0] # k=0 for (e)
    subplot(3,1,1) #--------- #  Te^(3/2) / Zeff
    plt.hold(True)
    plt.grid(True)
    C= np.transpose((temp_e**1.5)/zeff[:,0:lrz])  # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) 
    C_min=np.min(C)
    #plt.xlabel('$time$ $(msec)$')
    plt.title('$T_e^{3/2}/Z_{eff}$',y=1.03)
    #xlim((np.min(timecode), np.max(timecode)+dtw  ))
    ylim_min=C_min/2
    ylim_max=C_max*2
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            t_pk=timecode[nt-1]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            xpos_txt= t_pk
            ypos_txt= C1_pk
            xpos_txt=timecode[nt-1]
            ypos_txt=C1[nt-1]
            if ypos_txt>ylim_min :
                plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                color=col_select[icount])
            #print 'tau_decay_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
            icount=icount+1
    subplot(3,1,2) #--------- #  Zeff
    plt.hold(True)
    plt.grid(True)
    C= np.transpose(zeff[:,0:lrz])  # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) 
    C_min=np.min(C)
    #plt.xlabel('$time$ $(msec)$')
    plt.ylabel('$Z_{eff}$   ',y=1.03)
    #xlim((np.min(timecode), np.max(timecode)+dtw  ))
    ylim_min=0.95 #C_min/2
    ylim_max=C_max*1.05
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            t_pk=timecode[nt-1]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.plot(timecode,C1,linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            xpos_txt= t_pk
            ypos_txt= C1_pk
            xpos_txt=timecode[nt-1]
            ypos_txt=C1[nt-1]
            if ypos_txt>ylim_min :
                plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                color=col_select[icount])
            #print 'tau_decay_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
            icount=icount+1
    subplot(3,1,3) #--------- #  Te[keV]
    plt.hold(True)
    plt.grid(True)
    C= np.transpose(temp_e)  # A[rho,time] ALL TIME STEPS
    C_max=np.max(C) 
    C_min=np.min(C)
    plt.xlabel('$time$ $(msec)$')
    plt.ylabel('$T_e[keV]$   ',y=1.03)
    #xlim((np.min(timecode), np.max(timecode)+dtw  ))
    ylim_min=C_min/2
    ylim_max=C_max*2
    ylim((ylim_min, ylim_max))  # few orders of magnitude to show.
    icount=0
    C1=timecode*0 
    for ir in ir_select:  
        if ir<lrz:
            t_pk=timecode[nt-1]
            C1_pk=0.
            for it in range(0,nt): # loop in time index
                C1[it]= max(C[ir,it], ylim_min/10) # all t: impose lower limit
                if C1[it]>=C1_pk:
                    t_pk=timecode[it]
                    C1_pk=C1[it]
            linww= 4*linw-icount # Start with bold line, then reduce
            linww=max(linww,0.75) # line thickness: not lower than 0.75pt
            plt.semilogy(timecode,C1,linewidth=linww,color=col_select[icount])
            txt1= r"$lr=$"+r"$%3i$" %(ir+1) # lr is the radial index in code counting
            txt2= r" $\rho=$"+r"$%1.3f$" %(rho[ir])
            xpos_txt= t_pk
            ypos_txt= C1_pk
            xpos_txt=timecode[nt-1]
            ypos_txt=C1[nt-1]
            if ypos_txt>ylim_min :
                plt.text(xpos_txt, ypos_txt, txt2, fontsize=fnt,\
                color=col_select[icount])
            #print 'tau_decay_vs_time:  Added ir=',ir,' Min/Max=',np.min(C1),np.max(C1)
            icount=icount+1
    #plt.title(r' $T_e^{3/2}/Z_{eff}$',y=1.03)
    savefig('Te32_Zeff_vs_time'+'.png')
    #savefig('tau_decay_vs_time'+'.eps')
    plt.show()    
    #--------------
    #--------------------------------------------------------------------------



#============= favr_thet0 from CQL3D =====================================
#if i_favr_thet0>0:
for lr in range(0,lrz):
    #lr=0 #5-1 #18-1 # in cql3d: lr_=6 -> rho=0.25; lr_=9 -> rho=0.39; lr_=18 -> rho=0.8
    #In NIMROD+CQL3D runs, lrz=22, lr=5[code] corr to rho=0.2045
    # lr=13[code] corr to 0.5682
    istride=10 #20  # SKIP some steps
    tplot_min=1.8 #0.4 #0.78 #1.64 #0.78  #ms
    tplot_max=5.6 #1.0 #1.76 #1.0  #ms
    nstop=len(timecode)-1 # nstop in cqlinput
    it00=0 #220  # it=itstart is time step for T-drop begin
    itend=nstop #220 #nstop+1
    imx=1.0 #0.2 #1.0
    
    fig1= plt.figure() 
    x_axis=  x*(unorm/clight) #VV2_cql  #=VV2_cql= ((u/gammac)/Vt0_cql)**2
    # alternatively, use x*(unorm/clight)==u/c or Enrgy_j
    #Enrgy_MeV= Enrgy_j[:,ksp-1]/1e3 # Horiz axis, to MeV
    #x_axis= Enrgy_MeV
    #---------------
    ax = plt.subplot(1, 1, 1)
    plt.hold(True)
    plt.grid(True)
    #x_axis_mx= VV2_plot_mx
    x_axis_mx= np.max(x_axis)
    plt.xlim((0.,x_axis_mx*imx)) 
    plt.xlim((0.,x_axis_mx*imx))  
    plt.yscale('log') # log 10 scale !!! Comment it, if you want a linear scale    
    if f_plot_mx>0:
        y_axis_mx= f_plot_mx
        y_axis_mn= f_plot_mx/f_plot_range
    else:
        y_axis_mx= 1e-1
        y_axis_mn= 1e-9 # as in Fig.4 of REF
    ylim((y_axis_mn, y_axis_mx)) # for log10 scale
    xlabel("$p/mc$") #,fontsize=fnt+6)
    #plt.text(0.81*x_axis_mx,0.01*f_plot_mx/f_plot_range,"$(u/\gamma/v_{To})^2$")
    #ylabel("$favr_{cql3d}$ $(averaged$ $over$ $pitch-angle)$") #,fontsize=fnt+6)  
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')  
    if ngen>1: # more than one gen.species
        favr_thet0=s_file_cql3d.variables['favr_thet0'][:,ksp-1,lr,:]
    else:  # one gen.species
        favr_thet0=s_file_cql3d.variables['favr_thet0'][:,lr,:]
    print 'SHAPE of favr_thet0', favr_thet0.shape
    favr_thet0=favr_thet0/renorm #f_SI= f_cql3d/renorm
    itt=-1
    print '------------------------'
    for it in range(it00,itend,istride):
        itt=itt+1
        favr_thet0_it= favr_thet0[it,:] # size=jx 
        if remainder(itt,5)==0: col='k'
        if remainder(itt,5)==1: col='b'
        if remainder(itt,5)==2: col='r'
        if remainder(itt,5)==3: col='g'    
        if remainder(itt,5)==4: col='m' 
        if remainder(itt,5)==5: col='c' 
        plt.plot(x_axis,favr_thet0_it,color=col,linewidth=0.5*linw) 
        # Add text/label on each curve 
        txt4= r"$%1.2f$" %(timecode[it]) +"$ms$"  # Time [ms] for a given curve
        #iv_txt= np.floor(jx*0.55)+np.floor(itt*4)   #V-index where to place label
        #iv_txt= np.floor(jx*0.5)+np.floor(itt*4)   #V-index where to place label
        # Where to put time label:
        favr_thet0_level=1.e-22 #1.e-17
        iv_txt= np.where( abs(log(favr_thet0_it)-log(favr_thet0_level))<2)
        if np.size(iv_txt)==0:
            iv_txt=20
        else:
            iv_txt=np.floor(np.mean(iv_txt))
        iv_txt= max(iv_txt,150) #+itt+50 #+itt*4
        iv_txt= min(iv_txt,jx-3)
        xpos_txt= x_axis[iv_txt]
        ypos_txt= favr_thet0_it[iv_txt]        
        print 'iv_txt[adjusted]=',iv_txt, xpos_txt, ypos_txt, txt4
        #if (timecode[it]>3.7)&(timecode[it]<20.)&(remainder(itt,1)==0):
        if (timecode[it]>tplot_min)&(timecode[it]<tplot_max)&(remainder(itt,3)==0):
            plt.text(xpos_txt, ypos_txt, txt4, \
            color=col,fontsize=fnt-5,rotation=90,ha='left',va='bottom')
    #----------- Add text/label  
    txt2= "$t_{end}$="+r"$%1.2f$" %(timecode[nstop]) +"$ms$" # Time at n=nstop
    xpos_txt= x_axis_mx*0.3 
    ypos_txt= y_axis_mx/1e4 
    #plt.text(xpos_txt, ypos_txt, txt3+'  '+txt2, fontsize=fnt+1,color='k')
    #-----------
    #txt2= " $n=$" + r"$%1.3e$" %(density[nstop,lr,ksp-1]*1e6) + "$m^{-3}$" 
    #txt3= " $T=$" + r"$%1.3f$" %(temp[nstop,lr,ksp-1]) + "$keV$" 
    #plt.text(xpos_txt, ypos_txt*15,  "$At$ $t=t_{end}:$ " +txt3)        
    #txt2= " $n=$" + r"$%1.3e$" %(density[0,lr,ksp-1]*1e6) + "$m^{-3}$" 
    #txt3= " $T=$" + r"$%1.3f$" %(temp[0,lr,ksp-1]) + "$keV$" 
    #plt.text(xpos_txt, ypos_txt*200, "$At$ $t=0:$    " +txt3)    
    #txtV= " $v_{cr}(t_{end})/v_{To}=$" + r"$%1.2f$" %(Vcr_Vt0)
    #plt.text(xpos_txt, ypos_txt/100, txtV)    
    plt.title("$f_{cql3d}$ $aver.$ $over$ $pitch$ $angle$  $(m^{-3}/(m/s)^3)$", y=1.03)
    # Save as png plot:
    tend=timecode[itend] #np.max(timecode)
    print 'tend[ms]=',tend
    txt4= "%2.1f" %(tend) +"ms" 
    savefig('favr_thet0_cql3d_renorm__r'+str(lr+1)+'_t'+txt4+'.png')
    #savefig('favr_thet0_cql3d_renorm__r'+str(lr)+'_t'+txt4+'.eps')
    #-------------------- FIGURE IS SAVED -------------------------
    show()    # comment it, to avoid pressing "Enter" after each plot

stop

    
if nstates>0:
    #----- Ne_bound INVENTORY 
    #[SUM(Nb(kstate)*density(rho,kstate))*dvol(rho) for each rho and t]
    kstate_max= nstates
    nb=nbound 
    A=nb*0 #A*0# initialize shape: [ir,it]
    #print 'shape of T, A', np.shape(T), np.shape(A)
    for it in range(0,len(A[0,:])):
        for ir in range(0,len(A[:,0])):
            A[ir,it]= nb[ir,it]*dvol[ir] # ptcls/cm^3 * cm^3
            # Checked that ( ns[ir,it]+dens_adj[ir,0] )*dvol[ir] 
            # where ns= ns+B*bnumb_imp[kstate]   gives same plot
            # as dens_adj[] (Ne_free)
    fig0=plt.figure(403)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        #ax.zaxis.set_scale('log')
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        #ax.set_zlim3d(min(A_min,0.), max(A_max,0.))
        #ax.set_zlim3d(A_min, A_max)
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        if nstates>0:
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
            plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(r'$N_{e,bound}$ $inventory$',y=1.02)
    savefig('Ne_bound_rho_2D_ksp'+str(ksp+1)+'.png') # ksp+1 to match CQL3D count
    plt.show()  
    
    #----- density, All charge states combined
    A=np.transpose(dens_imp_allstates[it0:nt:ntstride,0:lrz])    # A[rho,time] 
    dens_imp_tot=A  # save: {t,rho}
    for itime in range(it0,nt,ntstride):     
        print 'min/max of dens_imp_tot[:,itime]=',\
        timecode[itime],  np.min(dens_imp_allstates[itime,:]),\
        np.max(dens_imp_allstates[itime,:])
    print T.shape,A.shape,' A=transposed(dens_imp_allstates[it,0:lrz]):',A.shape
    A_max=np.max(A)
    A_min=np.min(A)
    print 'MIN,MAX of dens_imp_allstates =', A_min,A_max
    # Print this state charge and weight:
    txt1='  $Z_{imp}=$'+r"%i" %(bnumb_imp[nstates])
    txt2='  $m_{imp}/m_p=$'+r"%5.1f" %(fmass_imp/p_mass)   
    #print txt1+txt2    
    A_max=np.max(A)+1
    fig0=plt.figure(601)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir) 
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title(' $n(cm^{-3})$ $impurity$' +txt1+txt2, y=1.02)
    savefig('dens_impALLstates_rho_2D'+'.png') #
    plt.show() 

    #stop
    
    #----- density, for EACH charge state
    kstate_max= nstates
    Zav= dens_imp_tot*0  # initialize
    for kstate in range(0,kstate_max+1): # loop in charge states
        A=np.transpose(dens_imp[it0:nt:ntstride,0:lrz,kstate])    # A[rho,time] 
        dens_imp_kstate=A  # save: {t,rho}
        Zav= Zav + A*bnumb_imp[kstate]  # SUM over n(z)*Z
        # Print this state charge and weight:
        txt1='  $Z_{imp}=$'+r"%i" %(bnumb_imp[kstate])
        txt2='  $m_{imp}/m_p=$'+r"%5.1f" %(fmass_imp/p_mass)    
        #print T.shape,A.shape,' A=transposed(dens_imp[it,0:lrz]):',A.shape
        A_min=np.min(A)
        A_max=np.max(A)
        print 'kstate=',kstate, '  MIN/MAX of dens_imp:', A_min, A_max
        if A_max>1e7:
            A_min=0. # Lower limit for plots
            A_max=dens_imp_max # Max Over all states. # Or given kstate: A_max+1
            fig0=plt.figure(602+kstate)
            if imesh==1:
                ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
                ax.plot_wireframe(T,(R),(A),rstride=200,cstride=1,cmap=cm.jet)
                ax.set_xlim3d(0.,1.0)
                ax.set_ylim3d(t_low_lim,timecode[nt-1] )
                ax.set_zlim3d(A_min,A_max)
                zdir = (None) # direction for plotting text (title)
                xdir = (None) # direction
                ydir = (None) # direction
                ax.set_ylabel(r"$time$  $(msec)$", xdir) 
                ax.set_xlabel(r"$\rho$", ydir) 
            else:
                plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
                CS=plt.contour(T,(R),(A),Ncont,linewidth=linw,cmap=plt.cm.jet)
                CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
                plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
                plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
                plt.ylabel(r"$time$  $(msec)$", xdir) 
                plt.xlabel(r"$\rho$", ydir) 
        plt.grid(True)
        plt.minorticks_on() # To add minor ticks
        plt.tick_params(which='both',  width=1)
        plt.tick_params(which='major', length=7)
        plt.tick_params(which='minor', length=4, color='k')
        #ax.set_zlabel('$ $  $ $', rotation=-90)
        plt.title('  $n(cm^{-3})$ $of$ $ch.state$' +txt1+txt2, y=1.02)
        savefig('dens_imp_rho_2D_Z'+r"%i" %(bnumb_imp[kstate])+'.png') #
        #plt.show()    
    #---loop in kstate done

    # Average <Z> charge [ NOT  Zeff !!! ]
    #Zav= Zav/dens_imp_tot #== SUM(n(z)*Z) / dens_imp_tot
    nts= Zav[0,:].size  # For selected time steps
    for it in range(0,nts):
        for lr in range(0,lrz):
            if dens_imp_tot[lr,it] > 1e-100:
                Zav[lr,it]= Zav[lr,it]/dens_imp_tot[lr,it]
            else:
                Zav[lr,it]=0.0
                                    
    #print 'Zav.shape', Zav.shape
    print 'MIN/MAX of Zav', np.min(Zav), np.max(Zav)
    fig0=plt.figure(701)
    if imesh==1:
        ax  = Axes3D(fig0,azim=view_azim,elev=view_elev)
        ax.plot_wireframe(T,(R),(Zav),rstride=200,cstride=1,cmap=cm.jet)
        ax.set_xlim3d(0.,1.0)
        ax.set_ylim3d(t_low_lim,timecode[nt-1] )
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        ax.set_ylabel(r"$time$  $(msec)$", xdir) 
        ax.set_xlabel(r"$\rho$", ydir) 
    else:
        plt.axis([0.,1.1, t_low_lim, timecode[nt-1] ])
        CS=plt.contour(T,(R),(Zav),Ncont,linewidth=linw,cmap=plt.cm.jet)
        CB=plt.colorbar(orientation='vertical', shrink=0.9, format='%.2e')
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],color='r',linewidth=linw*2)
        plt.plot(pellet_rho[1:it_Mgone],timecode[1:it_Mgone],'k.')
        plt.ylabel(r"$time$  $(msec)$", xdir) 
        plt.xlabel(r"$\rho$", ydir)         
    plt.grid(True)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    plt.title('  $<Z>$ $of$ $impurity$', y=1.02)
    savefig('Zav_rho_2D'+'.png') #
    plt.show()    
    
    
    
    