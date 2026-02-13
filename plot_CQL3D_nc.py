#ipython
# plot_CQL3D_nc.py (based on plot_DC_multiR-cql3d_format_cqlb_f_forb_110918.py)
#YuP and BH, 2011-2020

# profile plots of different diagnostics (NPA, current, etc),
# mesh or contour plots of 
# distribution functions at different radial points,
# diffusion coeffs at each minor radius, etc.
# Reading the *.nc file specified below by  file_cql3d=...

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
#----------------------------------------------------------------------------

#e0 = time.time()  # elapsed time since the epoch
#c0 = time.clock() # total cpu time spent in the script so far

#-----------------------------------------------
# NetCDF issues: machine-dependent
# Try netcdf=4 to envoke netCDF4,
# Or try netcdf=2 to work with older netCDF.
netcdf=4
#-----------------------------------------------
if netcdf==4: from netCDF4 import Dataset # YuP
#-----------------------------------------------


# Specify data source and plot type:

data_src='cql3d_nc' # specifies data source: 'DC_text' files, assumed
#                   # to have path ../du0u0_grid and ../du0u0_r001, etc.,
#                   # or 'cql3d_nc' netcdf files file_cql3d (with path).
#                   # Just give file names (without .gz even if gzipped)

#file_cql3d='./mnemonic_rf.nc'  # for example, .gz ok, but omit the .gz.

#data_src='cql3d_nc' # specifies data source: 'DC_text' files, assumed


# GDT mirror machine (work started in 2016):
#file_cql3d='mirror_Ti2_NBI_e1000_lrz20_iy200_jx200_noloss_dtr1m3_fus.nc'
#file_cql3d='mirror_NBI.3.6_SS.nc'
#file_cql3d='gdt_nbi.1_5.5.nc'
#file_cql3d='gdt_nbi_hhfw.1_5.5.nc'
#file_cql3d='gdt_ech.3.12.nc'
#file_cql3d='gdt_ech_LF.4.nc'

# Alpha source [08-2016]
#file_cql3d='alpha_FULLorbgyro_lrz38_iyfus240_lz80_nt20_en8000iy102jx200_TRAN.nc'
#file_cql3d='alpha_ZOW_noloss_lrz38_iyfus240_lz80_nt20_en8000iy102jx200_r.nc'
#file_cql3d='alpha_HYBRorbgyro_lrz38_iyfus240_lz80_nt20_en8000iy102jx200.nc' #run 09/02/2016
#file_cql3d='alpha_FULLorbgyro_lrz38_iyfus240_lz80_nt10_en8000iy102jx200_noTRAN.nc'
#file_cql3d='alpha_FULLorbgyro_lrz38_en8MeV_iy102jx200_TRAN_nt40.nc'

# alphas + Helicon wave   (enorm=14MeV)  :
#file_cql3d='alpha_FULLorbgyro_lrz38_nt20_en14MeV_iy102jx200_TRAN_helicon_pwrscale1.nc'
#file_cql3d='alpha_FOWmimzow_lrz38_nt20_en14MeV_iy102jx200_helicon1.nc'
#file_cql3d='alpha_ZOW_lrz38_nt20_en14MeV_iy102jx200_helicon1.nc'
# Same but no Helicon
#file_cql3d='alpha_FULLorbgyro_lrz38_nt20_en14MeV_iy102jx200_TRAN_noRF.nc'

#file_cql3d='lh_iter5_e-alpha_vhhql.3.6.2.nc'
#file_cql3d='lh_iter5_alpha-e.11.1_HYBRID_alpha-source_damping.8.1.nc'
#file_cql3d='lh_iter5_alpha-e.11.1_HYBRID_alpha-source_damping.7.3.nc'

#file_cql3d='alphaZOW_lrz38_iy102jx200_en14MeV_nt20.nc'  # 04-15-2017

# rerun of EAEA-2016 with latest version (170415): nearly same.
#file_cql3d='alphaFULL_lrz38_nt20_en14MeV_iy102jx200_TRAN_helicon_170415.nc'
# very similar (nearly identical to the IAEA-2016 run on 09/17/2016)

# NEW data for ITER: using G_STEADY_EC1F_ITER_LR_01000.TXT
#file_cql3d='alphaFOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF.nc'  # 500MHz HHFW
#(run is done on 04/16/2017; MPI walltime=6155sec)
# Improved (05/01/2017):
#file_cql3d='alphaFOW_lrz38_iy102jx200_en14MeV_nt40_nmax4_RF_ineg.nc'
#file_cql3d='alphaFOWmimZOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF.nc'
#file_cql3d='alphaZOW_lrz38_iy102jx400_en60MeV_nt40_nmax4_RF_ineg.nc'

# EC1f 500 MHZ not-shifted:
#file_cql3d='alphaFOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF_ineg.nc'
#file_cql3d='alphaFOW_lrz38_iy102jx200_en14MeV_nt40_nmax4_RF_ineg.nc'
#file_cql3d='ngen2_nmax6_HHFW500_FULLnZOW_jx200_eni14MeV_ene400keV_nt40_0731.nc'
# 10/18/2017 in User\GENRAY_wk\ITER_HHFW_alphas\EC_60s_1f\cql3d_ngen2_500MHz_FULL_and_ZOW
#file_cql3d='ngen2_lrz38_HH500_30MW_FULL_ZOW_bootcalc.nc'
#file_cql3d='ngen2_lrz38_noRF_FULL_ZOW_bootcalc.nc'

# EC1f 500 MHZ shifted:
#file_cql3d='alphaFOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF500shift1m_ineg.nc'

# EC1F scenario, 800MHZ Helicon  05/01/2017:
#file_cql3d='alphaFOW_lrz38_iy102jx200_en14MeV_nt40_nmax4_RF800_ineg.nc'
#file_cql3d='alphaFOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF800_ineg.nc'
# 800 MHZ, shifted:
#file_cql3d='alphaFOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF800shift1m_ineg.nc'
#--- NBI with 800 MHz not shifted:  nt40
#file_cql3d='nbi17MW_FULL_jx200_en14MeV_nt40_nmax5_rf10MW_RF800_ineg.nc' 

# ITER LH03 scenario (G_HYBRID_LH03_ITER_LR_00400.TXT):
#file_cql3d='LH03_alphaFOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF500_ineg.nc'
##file_cql3d='LH03_alphaFOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF500shift1m_ineg.nc'
#file_cql3d='LH03_alphaFOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF800_ineg.nc'
#file_cql3d='LH03_alphaFOW_lrz38_iy102jx200_en14MeV_nt20_nmax4_RF800shift1m_ineg.nc'
#--- NBI with 800 MHz not shifted:   nt40
#file_cql3d='LH03_nbi33MW_FULL_jx200_en14MeV_nt40_nmax5_rf10MW_RF800_ineg.nc'


#file_cql3d='tdep_ko_amp-far.5.1.1c.nc' #2018-09-17, for IAEA-2018
#file_cql3d='tdep_ko_amp-far.4.3.1_yup1.nc'

# For CQL3D_svn/tmp_PPPL_LH/
#file_cql3d='cql3d_YuP1.nc'
#file_cql3d='cql3d.nc'

# \MCGO_wk\runs\CMOD_cql3d_runs\CMOD_1110217027_YuP_enorm500_pwrscale1_nstop10\
# Same as 
# \CQL3D_wk\CMOD_npa\1110217027\
#   160428_CMOD_CNPA_1110217027_128x128_4MODE_jx1000_aorsa-ZOW-dc-ZOW-cql3d.1\cql3d.2\
# only enorm=500, and du0u0 is diff? 
#[2019-02-27]
#file_cql3d='cmod_1080408021.2.nc' # use imx=1/sqrt(10) because emax was 5000
#file_cql3d='CMOD_1110217027_YuP_enorm500_pwrscale1_nstop10_2_aorsa2016_du0u0181019_132107.nc'

# For RF_Plasmas_2019\CQL3D_CMOD\
# For t=1.5ms
#file_cql3d='CMOD1110217027enorm500_pwrscale1_nstop30_aorsa2016_du0u0181019_1321_conserv.nc'
# For t=1.0ms
#file_cql3d='CMOD1110217027enorm500_pwrscale1_nstop20_aorsa2016_du0u0181019_1321_conserv.nc'

# For \NSTX_NBI_FOW_CQL3D\
#file_cql3d='fow171127_NSTX128742_FREYA5_frgyrodis_enorm100nstop60__fowFull.nc'


# For DIII-D with Ar pellet
# Good 10keV run, /lrz101_AMPF_pellet_Te10keV_tau01ms_a/ (10/11/2019)
# or /lrz101_AMPF_pellet_Te10keV_tau01ms_a/
#file_cql3d='tdep_ko_amp-far_lrz101_Pellet_Te10keV_dt0050ms_tau01.nc'
#file_cql3d='tdep_ko_amp-far_lrz101_Pellet_Te2keV_dt0050ms_tau005.nc'
# Also faster tau1=0.05ms with Te0=2.5keV case
#file_cql3d='tdep_ko_amp-far_lrz101_Pellet_Te2keV_dt0050ms_tau005.nc'
# Small fraction is converted (<30kA)  Not taken for APS
# But see lrz101_AMPF_pellet_Te2keV_tau005ms_V200: Vpell->200m/s:
# 95kA of RE


# For CMOD with W flake:
#file_cql3d='cmod1101201020_AF_LH.7.2.4.9.nc'
#file_cql3d='cmod1101201020_AF_LH.7.2.0.5.nc'
#--- Now With gamaset=-1.0, and target current totcrt=0.425e6A
#file_cql3d='cmod1101201020_AF_LH.10.1.nc' #curr_fit;
#file_cql3d='cmod1101201020_AF_LH.10.2.0.5.nc' #curr_fit; genray.nc_run.4-4.1
#file_cql3d='cmod1101201020_AF_LH.10.2.4.9.nc' #Wpellet , no AF
#file_cql3d='cmod1101201020_AF_LH.10.2.5.1.nc' #Wflake+AF , from 1160ms, n=900 to 75ms

#--- CQL3D-NIMROD coupling
#file_cql3d='test_read_nimrod_enorm10MeV_v5r.nc' # "r" = R-outboard profiles (not FSA)
#file_cql3d='test_read_nimrod_enorm10MeV_v5r_ext.nc'
# FSA data, 200 slices over 0-6ms
#file_cql3d='test_read_nimrod_enorm10MeV_v5fsa_notran_ext.nc'
#file_cql3d='test_read_nimrod_enorm10MeV_v5fsa_tran_ext.nc'
#file_cql3d='test_read_nimrod_enorm10MeV_v5fsa_tran_ext_KOdis.nc' # knockon='disabled'
#file_cql3d='test_read_nimrod_enorm40MeV_v5fsa_tran_ext.nc'
#file_cql3d='test_read_nimrod_enorm40MeV_v5fsa_notran_ext.nc'

# [2022-01-06] Added 1/gamma^5, and E is evaluated using coll. pressau2
#file_cql3d='cql_nimrod_v5fsa_Epressau2_gammi5_itl_tran000.nc' #two runs: w/wo bscurm2

#2022-03-11 Rerun of APS-2019 pellet cases, using efswtch='method6' (not AMPF):
#file_cql3d='tdep_lrz21_Te2keV_tau02ms_pellet13_method6.nc'
#file_cql3d='tdep_lrz21_Te2keV_tau01ms_method6.nc'
#file_cql3d='tdep_lrz21_Te2keV_tau02ms_method4.nc'
#file_cql3d='tdep_ko_amp-far_lrz101_Pellet_Te10keV_dt0050ms_tau01.nc' #replot
#file_cql3d='tdep_lrz21_Te2keV_tau02ms_method4_efrelax09.nc'
#file_cql3d='tdep_lrz21_Te2keV_tau02ms_method4_efrelax06.nc'

#file_cql3d='lh_iter5_180826.4.nc' # LH case, shifted F, wave growth


#==============================================================================



Coeff = 'f' #'f_local'  # specify which Diff.Coeff to plot: cqlb,cqlc,cqle,cqlf,
#               #  distribution function 'f' or local distn f_local (uses 
#               #  forbshift) for ndeltarho='enabled cql3d case, in which 
#               #  case use .nc file, as opposed to _rf.nc cql3d file path
fow_bndry=0 #set to 0 for FOW-Hybrid f_BA (not local, but solution)
# With fow_bndry=0, only ZOW t-p cone is plotted.     

Emax = 'disabled' # Applicable for data_src='DC_text':
#                 # plots are on a upar,uperp grid, but with 'enabled'
#                 # coeffs above a  constant total maximum energy are
#                 # zero.  If a real number is entered here, it will
#                 # fraction of maximum energy on the grid.
#                 # Else, it is set =1.0 and enorm from cql3d
#                 # or max parallel or perp energy for DC data is used.

DCmn= 0.    # Specify vertical limits for mesh plots of Diff.Coeffs.
DCmx= 1.e16 # If DCmn=0. and DCmx=0., the limits will be set automatically
#DCmx= 0. # If DCmn=0. and DCmx=0., the limits will be set automatically

if Coeff=='f' or Coeff=='f_local':
    DCunits=1.00 # Units (scale) for mesh plots of log10[f()].
else:
    DCunits=1.00 # Units (scale) for plots of Diff.Coeff.


imx=0.125 #0.5 #1/sqrt(10) #0.07 #1.0 #0.75    # limits for u_par/unorm and u_prr/unorm 
#For enorm=40MeV, use imx=0.25
#                   # grid to plot (max: imx=1)
fcut_mn=1.e05 # lower limit in plots of CUTS of distr.func. If =0, then - automatic
fcut_mx=1.e17 # upper limit in plots of CUTS of distr.func. If =0, then - automatic
ismooth=0     # 
#             #  0-> no smoothening of plotted function;   1-> smoothening
itrim= 0  # 1->  Trim isolated peaks in plotted function,
          #  and Fill-in isolated holes in function
imov = 0  # 0-> movie is not saved;  1-> saved into *.avi   # NOT READY
i_R_start=1 # range of flux surfaces to plot [counting from 1, as in FORTRAN]
i_R_stop =23 #5 #21 #32

# For plots of profiles (1-plot/0-noplot)  ------------- : 
ivel_plots=1 # To make plots in vel.space (distr.func. or Diff.Coeffs)
iplot_fcuts=0 # To plot cuts of f at certain pitch angles
iplot_currv=0
iplot_fpar=1 # To plot reduced distr func Fpar(u_par/c) Added[2022-03-19]
iplot_RF=0 #1
iplot_powrfl=0 #1 # For plot of linear damping (data may not be always present)
iplot_powrf=0 #1 #0 # For plot of powrf(lrz_rho,nmodsa) (data may not be always present)
npa=0   # NPA diagnostics plots (vs energy, time, etc)
neutrons=0 #Fusion Neutrons diagnostic plots, for a set of nv_fus view chords
icurr=0 #1 # To plot current profiles
icurr_bscurr=0 # To add plots of model bscurr()
iden=0 #1 #1  # To plot FSA density and temperature (aver.energy) profiles
icons=0 # To plot conservation diagnostics
ienergy=0
# For plots of f_cuts :
imsh_cuts=zeros((4))
imsh_cuts[0]=1 #5 #12 #21  # CUTS of f are done at these i-indices
imsh_cuts[1]=40 #24 #41  # CUTS of f are done at these i-indices
imsh_cuts[2]=53 #66 #80-24 #62  # CUTS of f are done at these i-indices
imsh_cuts[3]=80 #100 #80-12 #82  # CUTS of f are done at these i-indices

#===============================================================================


plot_type='c' #'c' -> contour plot, or 'm' -> mesh plot
isave_eps=0 # To save eps format files (png are saved in any case)
fnt  =20 #24 #20 #28 #18 #12  22    # font size for axis numbers (see 'param=' below) 
# For f() plots the font is adjusted below: fnt=fnt-3
linw = 1.0    # LineWidth for contour plots
Ncont= 25 #20 #50    # Number of contour levels
stride=2      
# For mesh plots: if stride=2, color is assigned to each 2x2 cell of u-grid
# and mesh is shown for each 2nd line; but data is plotted over total u-grid
#
vsign=1.0 # To reverse sign of Vpar, depending on its defenition 
# Set vsign=-1. to match cql3d and COGENT: they have different def. of Vpar


#Specify limits for plots of current profiles (A/cm2):
curr_min=0. #-5.0 #0.0 #-20. #0.0 #-0.5
curr_max=0. #+40.0 #15.0 #+80. #0.0 #8.0 #2.5
powden_mx=0.0 #0.55
# if both limits are set to 0., the limits are set automatically.
enrg_min=0.
enrg_max=0.0 #3200. #4. #3.0 #[kev]
dens_max=0.0 #8e12 # cm^-3
rho_min=0.0
rho_max=1.0


#================== DONE: NOTHING else to specify =============================
# Constants
pi=3.14159265358979
clight= 2.99792458e10   # speed of light [cm/s]
charge= 4.8032e-10      # e-charge [cgs]
e     = 4.8032e-10      # e-charge [cgs]
p_mass= 1.67262158e-24  # proton mass    [gram]
proton= 1.67262158e-24  #    [gramm]
ergtkev=1.6022e-09      # energy(ergs) associated with 1 keV (from cql3d)
clite2=clight*clight

#set fonts and line thicknesses
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
    'font.weight'  : 'regular',
    'format' : '%0.1e'
}

pylab.rcParams.update(params)
#rc.defaults() #to restore defaults

mpl.rcParams['font.size']=fnt+2  # set font size for text in mesh-plots



if data_src=='DC_text':
    #---> READING GRID DATA -------
    # file du0u0_grid is produced by DC fortran code
    if os.path.exists('../du0u0_grid.gz'):
        os.popen('gunzip ../du0u0_grid.gz')
        igzip=1
    else:
        igzip=0
    grid  = open('../du0u0_grid','r') 
    n_uprp= int(grid.readline())
    n_upar= int(grid.readline())
    n_psi = int(grid.readline())   # Number of flux surfaces
    vc_cgs= float(grid.readline()) # Max velocity on the grid
    # vc_cgs = MAX(abs(upar0_min),upar0_max,uprp0_max) ! [cm/s] 
    [upar_min,upar_max]=np.array(grid.readline().split(),float)
    uprp_min=0.e0
    uprp_max=upar_max  # =1.0
    unorm = vc_cgs        # cm/s
    grid.close()
    if igzip==1:
        os.popen('gzip ../du0u0_grid')
    print 'u-grid:',[n_uprp,n_upar]
    unorm = vc_cgs        # cm/s
elif data_src=='cql3d_nc':
    #gunzip, if gzipped, and remember:
    if os.path.exists(file_cql3d+'.gz'):
        os.popen('gunzip file_cql3d')
        igzip=1
    else:
        igzip=0
    #Input netcdf file into a structure:
    if netcdf==2: 
        s_file_cql3d=nc.netcdf_file(file_cql3d,'r')
        if igzip==1:
            os.popen('gzip file_cql3d')

    #------------YuP:
    if netcdf==4: 
        s_file_cql3d= Dataset(file_cql3d, 'r', format='NETCDF4')

    unorm=s_file_cql3d.variables['vnorm'].getValue()  #getValue() for scalar
    unorm=np.asscalar(unorm)
    vnorm=unorm
    unorm2=unorm**2
    unorm3=unorm*unorm2
    unorm4=unorm2*unorm2
    enorm=s_file_cql3d.variables['enorm'].getValue()  #getValue() for scalar
    enorm=np.asscalar(enorm)
    ucmx= imx*unorm/clight # limits for plots
    
    symm=2.  # Factor to account for cql3d diffusion coeffs calculated
    # for half the up-down symmetric torus, and other half
    # assumed to have same diffusion coeff.   Whereas DC
    # obtained full bounce-averaged coeff for both halves.

    print 'The input file, ',file_cql3d,', contains:'
    print '========================================'
    print "The global attributes: ",s_file_cql3d.dimensions.keys()        
    print "File contains variables: ",s_file_cql3d.variables.keys()
    print '========================================'

    # Note: rfpwr array may not exist in *.nc:
    # 'rfpwr(rho,1:mrfn,time)= pwr den from individual modes'
    # In CQL3D, saved from powrf(lrindx(ll),kk=1:mrfn)
    # with added powrft(lrindx(ll))
    #        and sorpwt(lrindx(ll))
    #        and sorpwti(lrindx(ll))
    # so the second index in rfpwr runs 1:mrfn+3 (in CQL3D)
    try:
        try:
            rfpwr=array(s_file_cql3d.variables['rfpwr'])
        except:
            print('No data on rfpwr')
            iplot_RF=0      
        else:
            #rfpwr=array(s_file_cql3d.variables['pwrrf'])
            rfpwr=array(s_file_cql3d.variables['rfpwr'])
            print 'rfpwr:', rfpwr.shape
    finally:
        print '----------------------------------------'  
    
    
    rya=array(s_file_cql3d.variables['rya'])
    print 'rya:', rya.shape
    time=array(s_file_cql3d.variables['time'])
    print 'time:', time.shape
#    time_select=array(s_file_cql3d.variables['time_select'])
    time_select=time # for older versions of CQL3D (and comment the line above)
    #Older versions of CQL3D did not have "time_select" array.
    print 'time_select:', time_select.shape

    dvol= array(s_file_cql3d.variables['dvol'])  # cm^3
    darea=array(s_file_cql3d.variables['darea']) # cm^2
    lrz=s_file_cql3d.variables['lrz'].getValue()   
    lrz=np.asscalar(lrz) 
    if i_R_stop==0:
        i_R_stop=lrz
    iy=s_file_cql3d.variables['iy'].getValue()
    jx=s_file_cql3d.variables['jx'].getValue()        
    itl=array(s_file_cql3d.variables['itl'])
    itu=array(s_file_cql3d.variables['itu'])
    x=s_file_cql3d.variables['x'][:]

    timecode=time
    nstop=len(timecode)-1 # nstop in cqlinput
    ngen= s_file_cql3d.variables['ngen'].getValue()
    ngen= np.asscalar(ngen)
    print 'ngen=',ngen, '  nstop=',nstop
    
    reden=array(s_file_cql3d.variables['density']) # n0(t,lr,k) at midplane
    Nt= np.size(reden,0) # Number of time steps
    print 'reden:', reden.shape, '  Nt=',Nt
    it1=0 #159 # start from this time index
    ite=Nt #90 # Nt
    timeshift=0. #0.152 #sec#to match the plotted time axis with experiment
    # Shift time axis:
    ttt=np.asarray(timecode[it1:ite])+ timeshift #0.152
    t_1st=  ttt[0]
    tlast=  ttt[ite-1]
    print 'start/end time=', t_1st, tlast # 1st and last step


    bnumb=array(s_file_cql3d.variables['bnumb'])
    fmass=array(s_file_cql3d.variables['fmass'])
    print 'ngen=', ngen
    for k in range(0,ngen,1):
        enorm=fmass[k]*clite2*(sqrt(1.+unorm2/clite2)-1.)/ergtkev   
        print 'k, bnumb, fmass, enorm(keV)=', k,bnumb[k],fmass[k],enorm
        
    k=0
    enmx=fmass[k]*clite2*(sqrt(1.+ucmx**2)-1.)/ergtkev
    print 'k, ucmx(=uplot/c), energy at ucmx=', k,ucmx, enmx       
        
    gammac= sqrt(1+(x*unorm)**2/clight**2)  #[jx]
    #if (gammac-1) <= 1.e-6,
    #can use Enrgy_j= 0.5*fmass*(x*unorm)^2/ergtkev  # % keV
    Enrgy_j=zeros((jx,ngen))
    for k in range(0,ngen):
        Enrgy_j[:,k]= (gammac[:]-1)*fmass[k]*clight**2/ergtkev   #[jx,ngen]
        for j in range(0,jx):
            print 'j,x,u/c,gammac,Enrgy_j=',j,x[j],x[j]*unorm/clight,gammac[j],Enrgy_j[j,k]

    energym=array(s_file_cql3d.variables['energym']) #Midplane energy0(t,lr,k)
    print 'energym: shape and max value[keV]', energym.shape, np.max(energym)



    # Note: ns_bndry array may not exist in *.nc:
    try:
        try:
            ns_bndry=array(s_file_cql3d.variables['ns_bndry'])
        except:
            print('No data on ns_bndry')
            fow_bndry=0  # With fow_bndry=0, only ZOW t-p cone is plotted.     
        else:
            v_bndry=array(s_file_cql3d.variables['v_bndry'])
            theta_bndry=array(s_file_cql3d.variables['theta_bndry'])
            #fow_bndry=1
            print 'ns_bndry: ', ns_bndry.shape 
            print 'v_bndry: ', v_bndry.shape 
            print 'theta_bndry: ', theta_bndry.shape 
            # FOW boundaries will be plotted
    finally:
        print '----------------------------------------'  

    # Note: curr array may not exist in *.nc:
    try:
        try:
            curr=array(s_file_cql3d.variables['curr'])  # j_par(time,lr) or (time,k,lr)
        except:
            print('No data on curr')
            i_curr=0
        else:
            i_curr=1
            #curr=array(s_file_cql3d.variables['curr'])   # j_par(time,lr)
            print 'curr:', curr.shape # can be [Nt,ngen,lrz] if ngen>1
    finally:
        print '----------------------------------------'  

    # Note: currv array may not exist in *.nc:
    try:
        try:
            currv=array(s_file_cql3d.variables['currv'])  # j(time,lr,jx) or (time,k,lr,jx)
        except:
            print('No data on currv')
            i_currv=0
        else:
            i_currv=1
            #currv=array(s_file_cql3d.variables['currv'])   # j(time,lr,jx)
            print 'currv:', currv.shape # can be [Nt,ngen,lrz,jx] if ngen>1
    finally:
        print '----------------------------------------'  

    # Note: sigftt array may not exist in *.nc  :
    ifusrate=1
    try:
        try:
            sigftt=array(s_file_cql3d.variables['sigftt'])  #sigftt(nonch,4)
        except:
            print('No data on sigftt')
            ifusrate=0
        else:
            print 'sigftt:',  sigftt.shape
    finally:
        print '----------------------------------------'  


    if Coeff=='urfb':
        krf=s_file_cql3d.variables['krf'].getValue()        
        urfb=array(s_file_cql3d.variables['urfb'])
        print 'urfb:', urfb.shape,'u^2<Duu_QL>|Vpar|tau_b/vnorm^4'
        
    if (Coeff=='f') | (Coeff=='f_local') :

        if neutrons==1:
            nv_fus=  s_file_cql3d.variables['nv_fus'].getValue()
            nv_fus= np.asscalar(nv_fus)
            print 'Number of viewing chords nv_fus=',nv_fus
            # Detector(s) position (the starting points of viewlines):
            z_fus=array(s_file_cql3d.variables['z_fus']) # Z coord (cm)
            x_fus=array(s_file_cql3d.variables['x_fus']) # X=R (Y=0) (cm)
            
            # FUS. Neutron flux from general species (for nv_fus viewlines)
            # Units: Watts/m**2/steradian 
            flux_neutron_f=array(s_file_cql3d.variables['flux_neutron_f'])
            print 'flux_neutron_f.shape=',flux_neutron_f.shape
            flux_fus_min= np.min(flux_neutron_f)
            flux_fus_max= np.max(flux_neutron_f)
            print 'min/max of flux_neutron_f:',flux_fus_min, flux_fus_max
            
            # Flux collected from radial bins (separately)  (4,nv_fus,4*lrz)
            # Units:  neutrons/(sec*cm^2*steradian)
            # (1st index knumb=1:4 is for reaction type;
            #  neutrons can come from reaction 1(D+T) or 4(D+D) only.
            flux_rad_fus_f=array(s_file_cql3d.variables['flux_rad_fus_f'])
            print 'flux_rad_fus_f.shape=',flux_rad_fus_f.shape
            flux_rad_fus_f_min= np.min(flux_rad_fus_f)
            flux_rad_fus_f_max= np.max(flux_rad_fus_f)
            #---------
            print 'min/max of flux_rad_fus_f:',flux_rad_fus_f_min, flux_rad_fus_f_max
            rho_fus=array(s_file_cql3d.variables['rho_fus'])
            print 'rho_fus.shape=',rho_fus.shape
            Nrhobin= np.size(rho_fus,0) # size of array for rho_fus
            print 'max Number of rhobins Nrhobin=', Nrhobin
            # distance to mid of each bin along chord [cm] counting from edge:
            s_fus=array(s_file_cql3d.variables['s_fus']) 
            # width of each bin [cm] along viewing chord (sightline):
            ds_fus=array(s_file_cql3d.variables['ds_fus'])

            #--------------------------------------------------------------------------
            # Determine max rhobin index such that rhobin is not 0
            Nrho=0
            for ivc in range(0,nv_fus,1):
                for ir in range(0,Nrhobin,1):
                    if s_fus[ir,ivc]>0: Nrho=ir
            Nrho=Nrho+1
            print 'Nrho=', Nrho
            
            rrho= np.asarray(rho_fus[0:Nrho,:]) # now monotic 
            print 'rrho.shape=',rrho.shape
            print 'rrho[Nrho-1,:]=',rrho[Nrho-1,:]

            Zmesh = np.tile(z_fus,(Nrho,1))
            print 'Zmesh:', Zmesh.shape
            print 'Zmesh[Nrho-1,:]=',Zmesh[Nrho-1,:]
 
            # Calculate neutron flux from flux_rad_fus_f array. 
            # To get Watts/(m^2 steradian), multiply it by corresponding 
            # energy (14.1MeV for knumb=1; 2.45MeV for knumb=4), 
            # and by conversion factor 1.602e-9
            # which converts MeV to erg, erg/sec to Watts, and 1/cm^2 to 1/m^2
            # Neutrons only come from 
            # reaction#1 (D+T=>alpha(3.5MeV)+n(14.1MeV)) or 
            # reaction#4 (D+D=>he3(.82MeV)+n(2.45MeV))
            # Example as it is done in CQL3D-mirrors:
            # If General species are present:
            #flux_neutron_f(nn)= (flux_fus_f(1,nn)*14.1+flux_fus_f(4,nn)*2.45)*1.602e-9
            # If Maxwellian:
            #flux_neutron_m(nn)= (flux_fus_m(1,nn)*14.1+flux_fus_m(4,nn)*2.45)*1.602e-9

            # Convert to Watts/(m^2 steradian):
            Fluxf_r_z= (flux_rad_fus_f[0:Nrho,:,0]*14.1 \
                       + flux_rad_fus_f[0:Nrho,:,3]*2.45 )*1.602e-9
                        #(rho_bin,chord,type) Last index: 0- for D+T, 3- for D+D
            
            Fluxf_r_z_max=np.max(Fluxf_r_z)
            print 'min/max of Fluxf_r_z:',np.min(Fluxf_r_z), Fluxf_r_z_max
            print 'Fluxf_r_z.shape=',Fluxf_r_z.shape
            
            Fluxf_z= flux_neutron_f*0 # initialize: size=nv_fus
            for ivc in range(0,nv_fus,1):
                for ir in range(0,Nrho,1):
                    Fluxf_z[ivc]=Fluxf_z[ivc]+Fluxf_r_z[ir,ivc]
                #print 'ivc,Fluxf_z[ivc]=',ivc,Fluxf_z[ivc]
                
            fig2=plt.figure()   # fus flux as a func. of (Z,rho_bin)
            # Only make plots for the last time step in the run:
            ax  = Axes3D(fig2,azim=-40,elev=50) # angles in degrees
#BH doesn't have           ax.plot_surface(rrho,Zmesh,Fluxf_r_z,rstride=1,cstride=1,cmap=cm.coolwarm)
            ax.plot_surface(rrho,Zmesh,Fluxf_r_z,rstride=1,cstride=1,cmap=cm.jet)
            ax.set_xlim(-1,1)  # for rho coord, with the sign
            ax.grid(True)
            zdir = (None) # direction for plotting text (title)
            xdir = (None) # direction
            ydir = (None) # direction
            ax.set_ylabel('        $Z$ $of$ $detectors$  $(cm)$',xdir) 
            ax.set_xlabel(r'$\rho$ $along$ $view$ $chord$     ', ydir)
            ax.set_title('$[W/(m^2steradian)]$',fontsize=fnt+6,y=1.03)  
            txt='fusflux_vs_s_Zline_view.png'
            savefig(txt)
            show()
            #---------------
            fig1=plt.figure() # FUS neutron flux as a func. of Z position of detectors
            Flux_Z= flux_neutron_f # [viewchords]
            plt.hold(True)
            plt.grid(True)
            #plt.xlabel('$Z$ $of$ $detectors$  $(m)$',fontsize=26) 
            text(-2,0.02,'$Z$ $of$ $detectors$  $(m)$',fontsize=26)
            #plt.ylabel('$Watts/m^2/steradian$') 
            plt.title(' $Neutron$ $Flux$ $to$ $Detectors$  $[W/(m^2steradian)]$',\
            fontsize=26,y=1.03)  
            plt.plot(z_fus[:]/100,Flux_Z[:],'r',linewidth=linw*4)
            #plt.plot(z_fus[:]/100,Fluxf_z[:],'k--',linewidth=linw)
            txt='FUSneutron_flux_vs_Zviewline.png'
            savefig(txt)
            show()
            
        #stop # end of neutrons==1
#------------------------------------------------------------------------------


        if icurr==1:
            if icurr_bscurr==1:
                # Hirshman-Sauter Model bootstrap current profile
                bscurr_e_gen=array(s_file_cql3d.variables['bscurr_e_gen']) 
                print 'bscurr_e_gen:', bscurr_e_gen.shape
                bscurr_e_maxw=array(s_file_cql3d.variables['bscurr_e_maxw']) 
                print 'bscurr_e_maxw:', bscurr_e_maxw.shape
                bscurr_i_gen=array(s_file_cql3d.variables['bscurr_i_gen']) 
                print 'bscurr_i_gen:', bscurr_i_gen.shape
                bscurr_i_maxw=array(s_file_cql3d.variables['bscurr_i_maxw']) 
                print 'bscurr_i_maxw:', bscurr_i_maxw.shape

            curtor=array(s_file_cql3d.variables['curtor']) # j_tor(time,lr)
            print 'curtor:', curtor.shape
            Nt= np.size(curtor,0) # Number of time steps
            print 'Number of time steps Nt=', Nt
        
            #--------------------------------------------------------------------------
            fig1=plt.figure()   # Current profiles: curtor
            title('     $Current$ $profile$ $change$ $(code)$',y=1.03)
            plt.hold(True)
            plt.grid(True)
            plt.ylabel('$j_{tor}$ $at$ $midplane$'+'  '+'$(A/cm^2)$',fontsize=28)
            if (curr_max-curr_min > 0.):
                plt.ylim([curr_min,curr_max])
                text(1.02,curr_min+0.03*np.amax(curr_max-curr_min),r'$\rho$',fontsize=34)
            else:    
                plt.xlabel(r'$\rho$',fontsize=32)
            J=curtor[0,:]  
            plt.plot(rya,J,'c',linewidth=linw*2) # The 1st t-step: CYAN
            plt.xlim([rho_min,rho_max])
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            for it in range(1,Nt,1):
                J=curtor[it,:]  # j_tor(time,lr)
                plt.plot(rya,J,'g',linewidth=linw)
                #AX=axis([0,1,min(min(cur0_fba),0),max(cur0_fba)])
            plt.plot(rya,J,'r',linewidth=linw*3) # the last t-step: RED
            if isave_eps==1: savefig('profiles_Jtor_midpl_time.eps')
            savefig('profiles_Jtor_midpl_time.png')
            #show() 
        
        #--------------------------------------------------------------------------
        
        for k in range(0,ngen,1):
            fig1=plt.figure()  # Current profiles: curr==<jpar>_FSA from code run
            #NOTE: to convert (A/cm^2) = 10000*(A/m^2) = 10*(kA/m^2)
            # curr() may include a bootstrap current and current from CD_RF 
            print 'k=',k
            if ngen>1: #Note: the style of saving data in netcdfrw2 is diff for ngen>1 
                curr_k=curr[:,k,:]
            else: # ngen=1 (one general species)
                curr_k=curr[:,:]
            print 'shape of curr_k', np.shape(curr_k)
            #stop    
            curr_darea= np.dot(curr_k[Nt-1,:],darea)
            print ' Total current [A] from curr:  k,sum(curr*darea)=',k,curr_darea
            txt="%1.6f" %(curr_darea/1e6) +'$MA$'
            if icurr_bscurr==1:
                title('  $FSA$  $<j_{||}>$  $(kA/m^2)$  $(Black:$ $jhirsh;$ $Red:$ $code)$',fontsize=28,y=1.06)
            else:
                title(' $<j_{||}>_{FSA}$ $(kA/m^2)$   $I=$'+txt,fontsize=24,y=1.03)
            plt.hold(True)
            plt.grid(True)
            #plt.ylabel('$(A/cm^2)$',fontsize=28)
            if (curr_max-curr_min > 0.):
                plt.ylim([curr_min*10,curr_max*10])
                text(1.02,curr_min*10+0.3*np.amax(curr_max-curr_min),r'$\rho$',fontsize=34)
            else:    
                #plt.xlabel(r'$\rho$',fontsize=32)
                text(1.02,0.3*np.max(curr_k),r'$\rho$',fontsize=34)

            if icurr==1:
                J=curr_k[0,:]*10  
                plt.plot(rya,J,'c',linewidth=linw*2) # The 1st t-step: CYAN
                plt.xlim([rho_min,rho_max])
                plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
                for it in range(1,Nt,1):
                    J=curr_k[it,:]*10    # j_par(time,lr)
                    linww=linw  #*(Nt-it)
                    plt.plot(rya,J,'g',linewidth=linww)
                    #AX=axis([0,1,min(min(cur0_fba),0),max(cur0_fba)])
                plt.plot(rya,J,'r',linewidth=linw*3) # the last t-step: RED
                if icurr_bscurr==1:
                    J=bscurr_i_gen[Nt-1 ,:]  # j_par(time,lr)
                    plt.plot(rya,J,'k',linewidth=linw*3)
                if isave_eps==1: savefig('profiles_Jpar_FSA_time.eps')
                savefig('profiles_Jpar_FSA_time_k'+str(k)+'.png')
                savefig('profiles_Jpar_FSA_time_k'+str(k)+'.eps')
                #show() 
        #---- endfor k ------------------------------------
        #stop
        #--------------------------------------------------------------------------
        fig1=plt.figure()  # Current profiles: curr==<jpar>_FSA/Prf from code run
        title('  $FSA$  $<j_{||}>/P$  $(A/m^2/W)$',fontsize=28,y=1.05)
        #NOTE: to convert (A/cm^2)/kW = 10000*(A/m^2)/kW = 10*(A/m^2)/W
        #NOTE: The curr() current may include the bootstrap current, 
        # so it is not a good idea to normalize by Prf.
        plt.hold(True)
        plt.grid(True)
        #plt.ylim([curr_min,curr_max])
        text(1.02,0,r'$\rho$',fontsize=34)

        if icurr==1:
            Prf_inj=10000 # specify in Watt
            JP=curr_k[0,:]*10/Prf_inj # (A/cm^2)/kW = 10000*(A/m^2)/kW = 10*(A/m^2)/W
            plt.plot(rya,JP,'c',linewidth=linw*2) # The 1st t-step: CYAN
            plt.xlim([rho_min,rho_max])
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            for it in range(1,Nt,1):
                JP=curr_k[it,:]*10/Prf_inj    # j_par(time,lr)/P
                linww=linw  #*(Nt-it)
                plt.plot(rya,JP,'g',linewidth=linww)
            plt.plot(rya,JP,'r',linewidth=linw*3) # the last t-step: RED
            if isave_eps==1: savefig('profiles_Jpar_over_Prf_time.eps')
            savefig('profiles_Jpar_over_Prf_time.png')
            #show() 
        #stop
        #--------------------------------------------------------------------------
        if icurr_bscurr==1:
            fig1=plt.figure() #Bootstrap current: Hirshman-Sauter Model. i_general
            title('     $Bootstrap$ $model:$ $ions(FP-ed)$  '+r'$(\rho,time)$',y=1.03)
            plt.hold(True)
            plt.grid(True)
            plt.ylabel('$<j_{||}>$    $(A/cm^2)$',fontsize=28)
            plt.xlim([rho_min,rho_max])
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            if (curr_max-curr_min > 0.):
                plt.ylim([curr_min,curr_max])
                text(1.02,curr_min+0.03*np.amax(curr_max-curr_min),r'$\rho$',fontsize=34)
            else:    
                plt.xlabel(r'$\rho$',fontsize=32)
            if icurr_bscurr==1:
                J=bscurr_i_gen[0,:]  
                plt.plot(rya,J,'c',linewidth=linw*2) # The 1st t-step: CYAN
                for it in range(1,Nt,1):
                    J=bscurr_i_gen[it,:]  # j_par(time,lr)
                    plt.plot(rya,J,'g',linewidth=linw)
                plt.plot(rya,J,'r',linewidth=linw*3) # the last t-step: RED
            if isave_eps==1: savefig('profiles_Jbs_model_igen.eps')
            savefig('profiles_Jbs_model_igen.png')
            #show() 
            #--------------------------------------------------------------------------
            #stop


        #--------------------------------------------------------------------------



        fig1=plt.figure()   # dens profile from code run: reden==n0 at midplane
        k=1 # species for the density plot
        title('   $Midplane$ '+r'$n_0(\rho,t)$'+' $for$ $species$ $k=$'+str(k),y=1.03)
        plt.hold(True)
        plt.grid(True)
        plt.xlabel(r'$\rho$',fontsize=28)
        plt.ylabel('$n_{0}$'+'   '+'$(cm^{-3})$',fontsize=28)
        #axis([0,1,0,max(den_floc)*1.05])
        #axis([0,1,0,2e13]) #axis([0,1,0,3.5e13])
        den=reden[0,:,k-1]  
        plt.plot(rya,den,'c',linewidth=linw*2) # The 1st t-step: CYAN
        plt.xlim([rho_min,rho_max])
        plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
        for it in range(1,Nt,1):
            den=reden[it,:,k-1]  # n0(time,lr)
            plt.plot(rya,den,'g',linewidth=linw)
        plt.plot(rya,den,'r',linewidth=linw*3) # the last t-step: RED
        if isave_eps==1: savefig('profile_reden.eps')
        savefig('profile_reden.png')
        #show() 
        
        #--------------------------------------------------------------------------
        if ienergy==1:
            energym=array(s_file_cql3d.variables['energym']) #Midplane energy0(t,lr,k)
            print 'energym:', energym.shape
            ksp=1 # numbering as in CQL3D
            k=ksp-1 # general species, for now
            if ngen>1: #Note: the style of saving data in netcdfrw2 is diff for ngen>1 
                energym_k=energym[:,k,:]
                print 'shape of energym_k', np.shape(energym_k)
            else: # ngen=1 (one general species)
                energym_k=energym[:,:]
            
            fig1=plt.figure() # Energy (effective T) from code run: at Midplane 
            title('     $Midplane$ '+r'$energy(\rho,t)$'+' $for$ $species$ $k=$'+str(k),y=1.03)
            plt.hold(True)
            plt.grid(True)
            plt.xlabel(r'$\rho$',fontsize=28)
            plt.ylabel('$energy$'+'   '+'$(keV)$',fontsize=28)
            enrgy=energym_k[0,:]  
            plt.plot(rya,enrgy,'c',linewidth=linw*2) # The 1st t-step: CYAN
            plt.xlim([rho_min,rho_max])
            if (enrg_max-enrg_min > 0.):
                plt.ylim([enrg_min,enrg_max])
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            for it in range(1,Nt,1):
                enrgy=energym_k[it,:]  # energym(time,lr)
                ###plt.plot(rya,enrgy,'g',linewidth=linw)
            plt.plot(rya,enrgy,'r',linewidth=linw*3) # the last t-step: RED
            if isave_eps==1: savefig('profile_energy_midpl.eps')
            savefig('profile_energy_midpl.png')
            #show() 
        #--------------------------------------------------------------------------



        if iden==1:
            #CQL3D-FOW: den_FSA=array(s_file_cql3d.variables['den_fsa']) # FSA <n>(time,lr)
            den_FSA=array(s_file_cql3d.variables['density']) # 'density'==reden is n0
            print 'den_FSA:', den_FSA.shape
            k=1 # general species, for now
            if ngen>1: #Note: the style of saving data in netcdfrw2 is diff for ngen>1 
                den_FSA_k=den_FSA[:,k,:]
                print 'shape of den_FSA_k', np.shape(den_FSA_k)
            else: # ngen=1 (one general species)
                den_FSA_k=den_FSA[:,:]
            energy=array(s_file_cql3d.variables['energy']) #FSA <energy>(t,lr,k)
            print 'energy:', energy.shape
            fig1=plt.figure()   # dens profile from code run: den_fsa==<n>
            title('   $FSA$ $density:$'+'  '+'$change$'+'  '+'$in$'+'  '+'$time$',y=1.03)
            plt.hold(True)
            plt.grid(True)
            #plt.xlabel(r'$\rho$',fontsize=28)
            title('   $<n>_{FSA}$'+'   '+'$(cm^{-3})$',fontsize=28,y=1.03)
            den=den_FSA_k[0,:]  # n_FSA(time,lr) for species k
            plt.plot(rya,den,'c',linewidth=linw*2) # The 1st t-step: CYAN       
            plt.xlim([rho_min,rho_max])
            if dens_max>0:
                plt.ylim([0,dens_max])
            else:
                dens_max=np.amax(den)
                plt.ylim([0,dens_max])
            text(1.02,0.03*dens_max,r'$\rho$',fontsize=34)
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            for it in range(1,Nt,1):
                den=den_FSA_k[it,:]  # n_FSA(time,lr)
                plt.plot(rya,den,'g',linewidth=linw)
            plt.plot(rya,den,'r',linewidth=linw*3) # the last t-step: RED
            if isave_eps==1: savefig('profile_nFSA.eps')
            savefig('profile_nFSA.png')
            #show() 
            #--------------------------------------------------------------------------
            #--------------------------------------------------------------------------
            fig1=plt.figure() # Energy (effective T) from code run: <energy>FSA
            k=1 # species for the energy plot
            bnumb=array(s_file_cql3d.variables['bnumb'])
            fmass=array(s_file_cql3d.variables['fmass'])
            ngen= s_file_cql3d.variables['ngen'].getValue()
            print 'ngen=', ngen
            print 'k=',k
            print 'bnumb=',bnumb
            print 'fmass=',fmass
            title('   $FSA$ '+r'$energy(\rho,t)$'+' $for$ $species$ $k=$'+str(k),y=1.03)
            plt.hold(True)
            plt.grid(True)
            ###plt.xlabel(r'$\rho$',fontsize=28)
            plt.ylabel('$<energy>_{FSA}$'+'   '+'$(keV)$',fontsize=28)
            enrgy=energy[0,:,k-1]  
            plt.plot(rya,enrgy,'c',linewidth=linw*2) # The 1st t-step: CYAN
            plt.xlim([rho_min,rho_max])
            if (enrg_max-enrg_min > 0.):
                plt.ylim([enrg_min,enrg_max])
                text(1.02,enrg_min+0.03*np.amax(enrg_max-enrg_min),r'$\rho$',fontsize=34)
            else:    
                plt.xlabel(r'$\rho$',fontsize=32)
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            for it in range(1,Nt,1):
                enrgy=energy[it,:,k-1]  # <energy>(time,lr)
                plt.plot(rya,enrgy,'g',linewidth=linw)
            plt.plot(rya,enrgy,'r',linewidth=linw*3) # the last t-step: RED
            if isave_eps==1: savefig('profile_energyFSA.eps')
            savefig('profile_energyFSA.png')
            #show() 
            
        #--------------------------------------------------------------------------
        
        if icons==1:
            consn=array(s_file_cql3d.variables['consn']) # conservation(time,lr)
            print 'consn:', consn.shape
        
            fig1=plt.figure()   # Particle conservation: consn
            title('   $Particle$ $Conservation: $'+ r'$(\rho,time)$' ,y=1.03)
            plt.hold(True)
            plt.grid(True)
            ###plt.xlabel(r'$\rho$',fontsize=28)
            plt.ylabel('$(ptcln(r,t)-ptcln(r,t0))/ptcln(r,t0)$',fontsize=28)
            den=consn[0,:]  
            plt.plot(rya,den,'c',linewidth=linw*2) # The 1st t-step: CYAN
            plt.xlim([rho_min,rho_max])
            #plt.ylim([-2,22])
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            for it in range(1,Nt,1):
                den=consn[it,:]  # (N-N0)/N0  (time,lr)
                plt.plot(rya,den,'g',linewidth=linw*2)
            plt.plot(rya,den,'r',linewidth=linw*3) # the last t-step: RED
            cons_max=np.amax(den) # at the latest step
            ##text(1.02,0.03*cons_max,r'$\rho$',fontsize=34)
            text(1.04,0,r'$\rho$',fontsize=34)
            if isave_eps==1: savefig('profile_consn.eps')
            savefig('profile_consn.png')
            #show() 
            #stop
        #--------------------------------------------------------------------------

       
        if ifusrate==1:
            DD_neutron_rate= s_file_cql3d.variables['sigftt'][:,3]
            # Open file with experim. data
    #        file_exp='128742_nbi_neutron_rate.dat'
    #        #file_exp='128739_nbi_hhfw_neutron_rate.dat'
    #        f = open(file_exp, 'r')
    #        # Read and ignore header lines
    #        header1 = f.readline()
    #        header2 = f.readline()
    #        # Loop over lines and extract variables of interest
    #        texp=[]
    #        Neutron_rate=[]
    #        for line in f:
    #            line = line.strip()
    #            columns = line.split()
    #            texp.append(float(columns[0]))
    #            Neutron_rate.append(float(columns[1])*1e13)
    #        f.close()
    #        print 'texp:',min(texp),max(texp)
    #        print 'Neutron_rate (experim.):',min(Neutron_rate),max(Neutron_rate)
    #        print file_cql3d
    #        print file_exp
            fig1=plt.figure()
            plt.hold(True)
            plt.grid(True)
            plt.title('    DD Neutron Rate (NBI)', fontsize=20,y=1.03)
            plt.xlabel('Time (secs)', fontsize=18)
            plt.ylabel('Neutron rate (/sec)', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
    #        plt.plot(texp,Neutron_rate,    color='r', lw=2)  # experimental
            plt.plot(ttt, DD_neutron_rate, color='b', lw=2)  # CQL3D results
    #        plt.axis([0.15,0.36, 0.0, 2.0e13])
            plt.savefig('DD_neutron_rate_vs_time_NBI.png')
            plt.show()

        #--------------------------------------------------------------------------
        
        if npa==1:
            #nendim= s_file_cql3d.variables['nendim'].getValue()
            #print 'nendim=',nendim
            nen_npa= s_file_cql3d.variables['nen_npa'].getValue()
            nen_npa= np.asscalar(nen_npa)
            print 'Number of energy bins    nen_npa=',nen_npa
            nv_npa=  s_file_cql3d.variables['nv_npa'].getValue()
            nv_npa= np.asscalar(nv_npa)
            print 'Number of viewing chords nv_npa=',nv_npa
            enmin_npa= s_file_cql3d.variables['enmin_npa'].getValue()
            enmax_npa= s_file_cql3d.variables['enmax_npa'].getValue()
            enmin_npa= np.asscalar(enmin_npa)
            enmax_npa= np.asscalar(enmax_npa)
            print 'enmin_npa,enmax_npa=',enmin_npa,enmax_npa
            eflux_npa=array(s_file_cql3d.variables['eflux_npa'])
            print 'eflux_npa.shape=',eflux_npa.shape
            eflux_npa_min= np.min(eflux_npa)
            eflux_npa_max= np.max(eflux_npa)
            print 'min/max of eflux_npa:',eflux_npa_min, eflux_npa_max
            Nt_npa= np.size(eflux_npa,0) # number of time steps that were saved
            print 'Nt_npa=', Nt_npa
            if Nt_npa==2:
                # Only 1st and last time steps were saved
                itl=0
                ite=2
                ttt= [t_1st,tlast] # 1st and last t-step only
                print 'ttt=',ttt
            
            en_=array(s_file_cql3d.variables['en_'])
            print 'en_.shape=',en_.shape
            #----------- Added [04/24/2015]
            # Flux collected from radial bins (separately)
            eflux_r_npa=array(s_file_cql3d.variables['eflux_r_npa'])
            print 'eflux_r_npa.shape=',eflux_r_npa.shape
            eflux_r_npa_min= np.min(eflux_r_npa)
            eflux_r_npa_max= np.max(eflux_r_npa)
            print 'min/max of eflux_r_npa:',eflux_r_npa_min, eflux_r_npa_max
            rho_npa=array(s_file_cql3d.variables['rho_npa'])
            print 'rho_npa.shape=',rho_npa.shape
            Nrhobin= np.size(rho_npa,0) # size of array for rho_npa
            print 'max Number of rhobins=', Nrhobin
            # distance to mid of each bin along chord [cm] counting from edge:
            s_npa=array(s_file_cql3d.variables['s_npa']) 
            # width of each bin [cm] along viewing chord (sightline):
            ds_npa=array(s_file_cql3d.variables['ds_npa'])


            #--------------------------------------------------------------------------
            for ivc in range(0,nv_npa,1):   # for each viewing chord:
                fig1=plt.figure()   # NPA flux as a func. of E,time
                Flux_t_E= eflux_npa[it1:ite,ivc,:] # [t,viewchord,energy]
                # Cut the lowest values, to avoid 0.:
                eflux_npa_cut= eflux_npa_max*1e-5
                for ie in range(0,nen_npa,1):
                    for it in range(0,ite-it1,1):
                        if Flux_t_E[it,ie]<eflux_npa_cut: 
                            Flux_t_E[it,ie]= eflux_npa_cut
                print np.min(Flux_t_E), np.max(Flux_t_E)
                eee=np.asarray(en_)
                E,T=meshgrid(eee,ttt)
                print 'T.shape=',T.shape, ' Flux_t_E.shape=',Flux_t_E.shape
                ax  = Axes3D(fig1,azim=-60,elev=43) # angles in degrees
#BH doesn't have              ax.plot_surface(E,T,log10(Flux_t_E),rstride=1,cstride=1,cmap=cm.coolwarm)
                ax.plot_surface(E,T,log10(Flux_t_E),rstride=1,cstride=1,cmap=cm.jet)
                #ax.set_zlim(6, 10)
                ax.grid(True)
                zdir = (None) # direction for plotting text (title)
                xdir = (None) # direction
                ydir = (None) # direction
                ax.set_xlabel('$Particle$ $Energy$  $(keV)$           ', xdir) 
                ax.set_ylabel('$time$  $(sec)$', ydir) 
                ax.set_title('   $log_{10}(NPA flux /sec.cm^2.ster.eV)$' , fontsize=fnt+6,y=1.03)  
                #ax.set_zscale('log',basez=10,subsz=[2, 3, 4, 5, 6, 7, 8, 9])
                txt='NPA_flux_vs_E_t_view'+str(ivc)+'.png'
                savefig(txt)
                show()
                #-----------------------------
                #-----------------------------
                fig2=plt.figure()   # NPA flux as a func. of E,rho_bin
                # Only make plots for the last time step in the run:
                # Determine max rhobin index such that rhobin is not 0
                Nrho=0
                for ir in range(0,Nrhobin,1):
                    if rho_npa[ir,ivc]>0: Nrho=ir
                Nrho=Nrho+1
                eee=np.asarray(en_)
                rrho=np.asarray(rho_npa[0:Nrho,ivc]) # non-monotic -> messed up 
                ssnpa=np.asarray(s_npa[0:Nrho,ivc]) # [cm] ascending
                #rrr=np.arange(0,Nrho)
                print 'rrho=',rrho
                print 'ssnpa=',ssnpa
                
                Flux_r_E= eflux_r_npa[0:Nrho,ivc,:] #(rho_bin,chord,enrg)
                Flux_r_E_max=np.max(Flux_r_E)
                print 'min/max of Flux_r_E:',np.min(Flux_r_E), Flux_r_E_max
                # Cut off the values that are too low (NPA flux):
                Flux_r_E_cut= Flux_r_E_max*1e-5
                for ie in range(0,nen_npa,1):
                    for ir in range(0,Nrho,1):
                        if Flux_r_E[ir,ie]<Flux_r_E_cut: 
                            Flux_r_E[ir,ie]=Flux_r_E_cut
                E,S=meshgrid(eee,ssnpa)
                print 'Flux_r_E.shape=',Flux_r_E.shape
                ax  = Axes3D(fig2,azim=-40,elev=50) # angles in degrees
#BH doesn't' have                ax.plot_surface(E,S,log10(Flux_r_E),rstride=1,cstride=1,cmap=cm.coolwarm)
                ax.plot_surface(E,S,log10(Flux_r_E),rstride=1,cstride=1,cmap=cm.jet)
                #ax.set_zlim(6, 10)
                ax.grid(True)
                zdir = (None) # direction for plotting text (title)
                xdir = (None) # direction
                ydir = (None) # direction
                ax.set_xlabel('$Particle$ $Energy$  $(keV)$           ',xdir) 
                #ax.set_ylabel('   $bin$ $count$', ydir) 
                ax.set_ylabel('   $distance$ $along$ $view$ $chord$ $(cm)$', ydir)
                ax.set_title('   $log_{10}(NPA flux /sec.cm^2.ster.eV)$',fontsize=fnt+6,y=1.03)  
                #ax.set_zscale('log',basez=10,subsz=[2, 3, 4, 5, 6, 7, 8, 9])
                txt='NPAflux_vs_E_sline_view'+str(ivc)+'.png'
                savefig(txt)
                show()
            #stop
            #--------------------------------------------------------------------------
            
            #NPA plots: For each viewchord (sightline), plot NPA flux vs E,
            # for all time steps in one figure. 
            # Save one figure for one sightline
            for ivc in range(0,nv_npa,1):  # Loop in view-chord (sightline)
                fig1=plt.figure()   # NPA flux
                title('   $NPA$ $flux$ $for$ $View$ $Chord=$'+str(ivc),fontsize=20,y=1.03)
                plt.hold(True)
                plt.grid(True)
                plt.xlabel('$Particle$ $Energy$  $(keV)$',fontsize=20)
                plt.ylabel('$N.particles/(cm^2-sec-ster-eV)$',fontsize=20)
                plt.yscale('log') 
                #plt.axis([enmin_npa[0],enmax_npa[0],1e3,2e12])
                if remainder(ivc,6)==0: col='b'
                if remainder(ivc,6)==1: col='g'
                if remainder(ivc,6)==2: col='r'
                if remainder(ivc,6)==3: col='c'    
                if remainder(ivc,6)==4: col='m' 
                if remainder(ivc,6)==5: col='k' 
                for it in range(0,ite-it1,1):
                    it10=  it  #it/10
                    rem_it=remainder(it10,6)
                    if rem_it==0: col='c'
                    if rem_it==1: col='m'
                    if rem_it==2: col='g'
                    if rem_it==3: col='r'    
                    if rem_it==4: col='b' 
                    if rem_it==5: col='k' 
                    print 'it,time=', it, ttt[it]
                    Flux= eflux_npa[it,ivc,:] # [t,viewchord,energy]
                    plt.plot(en_,Flux,color=col,linewidth=2)
                    txt="%1.3f" %(ttt[it]) +'$sec$'
                    en_max= np.max(en_)
                    #text(0.8*en_max,2e8*10**(5.5*(-it)/float(Nt)),'$t=$'+txt,color=col,fontsize=10)
                txt='NPAflux_vs_E'+str(ivc)
                savefig(txt+'.png')
                if isave_eps==1: savefig(txt+'.eps')
                show()  
        #stop
#------------------------------------------------------------------------------



        bnumb=array(s_file_cql3d.variables['bnumb'])
        fmass=array(s_file_cql3d.variables['fmass'])
        ngen= s_file_cql3d.variables['ngen'].getValue()
        print 'ngen=', ngen
        print 'bnumb=',bnumb
        print 'fmass=',fmass
        
        powers_int=array(s_file_cql3d.variables['powers_int']) # p(time,k,type)
        print "powers_int:",powers_int.shape
        for k in range(0,ngen,1):
            #print 'powers_int at time=t_end:',k,powers_int[Nt-1,k,:]
            print 'powers_int for RF at time=t_end:',k,powers_int[Nt-1,k,5-1]
        
        powers=array(s_file_cql3d.variables['powers']) # powers(time,k,type,lr)
        print "powers:",powers.shape
        #for k in range(0,ngen,1):
            #print 'powers at time=t_end:',k,powers[Nt-1,k,:,:]
            
        #stop

        k=ngen
        powden_RF=  powers[:,k-1,5-1, :] # type#5 is for RF power 
        powden_RF_Nt= powden_RF[Nt-1,0,:] # last time step, 1st species
        print "powden_RF shape:", powden_RF.shape, powden_RF_Nt.shape
        print "rya shape:", rya.shape
        powden_e =  powers[:,k-1,1-1, :] # type#1 is for coll.power to maxw. e 
        powden_i =  powers[:,k-1,2-1, :] # type#2 is for coll.power to maxw. i 
        powden_tot= powers[Nt-1,k-1,13-1,:] # type#13 is for total power 
        powden_NBI= powers[Nt-1,k-1,6-1, :] # type#6 is for NBI power 
        powden_loss=powers[Nt-1,k-1,7-1, :] # type#7 is for loss power 
        
        # fuspwrv array may not exist in *.nc  (means no Fus.rate was enabled):
        try:
            try:
                fuspwrv=array(s_file_cql3d.variables['fuspwrv'])  
            except:
                print('No data on fuspwrv (no Fus.rate was calc-ed)')
                i_fus=0
            else:
                fuspwrv=array(s_file_cql3d.variables['fuspwrv'])
                print "fuspwrv:",fuspwrv.shape
                fuspwrvt=array(s_file_cql3d.variables['fuspwrvt'])
                print "fuspwrvt:",fuspwrvt.shape, fuspwrvt[:]
                i_fus=1
        finally:
            print '----------------------------------------'  
        
        
        
        #--------------------------------------------------------------------------
        fig7=plt.figure()   # Power profiles
        it_stride=6 # to skip some time slices
        
        powers_int=array(s_file_cql3d.variables['powers_int'])
        #'Vol int of FSA powers, respectively, to gen species k vs t' [Watt]
        print "powers_int shape:", powers_int.shape
        powers_int_RF= powers_int[:,k-1,5-1] # type#5 is for RF power 
        #print 'powers_int_RF [W], time slices:', powers_int_RF
        
        #plt.subplot(221) #-------------------------
        #title('$Power$ $den$  $(W/cm^3)$') 
        #plt.hold(True)
        #plt.grid(True)
        #plt.ylabel('$p_{loss}$')
        #plt.plot(rya,np.asarray(powden_loss),'r',linewidth=linw)
        #axis([0.,1.0,0.,1.05*np.amax(powden_loss)+0.001])
        
        #plt.subplot(222) #-------------------------
        plt.hold(True)
        plt.grid(True)
        #plt.title('   $p_{NBI}$  $(W/cm^3)$',fontsize=28,y=1.03)
        #plt.ylabel('$(W/cm^3)$')
        #plt.xlabel(r'$\rho$',fontsize=28)
        powden_mx= 1.05*np.amax(powden_RF)+0.001
        powden_mn=      np.amin(powden_RF)-0.001
        powden_mn= min(0.,powden_mn)
        plt.xlim([rho_min,rho_max])
        if powden_mx>0:
            plt.ylim([powden_mn,powden_mx])
        else:
            powden_mx=np.amax(powden_NBI)+0.001
            plt.ylim([powden_mn,powden_mx])
            text(1.05,0.03*np.amax(powden_NBI),r'$\rho$',fontsize=34)
        if (np.amax(powden_RF)>0.1) & (np.amax(powden_NBI)>0.1):
            # Both RF and NBI
            plt.plot(rya,np.asarray(powden_RF_Nt),'r',linewidth=linw*3)
            plt.title('   $FSA$  $p_{RF}$ $(red)$ $and$ $p_{source}$ $(blue)$   $(W/cm^3)$',fontsize=28,y=1.03)        
        elif np.amax(powden_RF)>0.1:
            plt.title('   $FSA$  $p_{RF}$  $(W/cm^3)$',fontsize=28,y=1.03)
            text(1.05,0.03*powden_mx,r'$\rho$',fontsize=34)
            dy=2*powden_mx/(Nt)
            it_count=0 # for text
            for it in range(0,Nt,it_stride):
                it_count=it_count+1 # for text
                powden_RF_it=powden_RF[it,0,:] #each t step, 1st species (k-1=0)
                plt.plot(rya,np.asarray(powden_RF_it),'g',linewidth=linw)
                power_W= np.asscalar(powers_int_RF[it])
                time_ms= time[it]*1e3
                if it>0:
                    txt1=r"$P_{rf}[W]=$"+r"$%1.4e$" %(power_W)
                    txt2= r" $t[ms]=$"+r"$%1.3f$" %(time_ms)
                    ypos_txt= powden_mx-dy*it_count
                    plt.text(0.3, ypos_txt, txt1 , va='center',fontsize=fnt-4)
                    plt.text(0.7, ypos_txt, txt2 , va='center',fontsize=fnt-4) 
            #Red - For the last time slice:
            plt.plot(rya,np.asarray(powden_RF_Nt),'r',linewidth=linw*3)
        elif np.amax(powden_NBI)>0.1:
#            plt.plot(rya,np.asarray(powden_NBI),'b',linewidth=linw*3)
            plt.title('   $FSA$  $p_{source}$  $(W/cm^3)$',fontsize=28,y=1.03)
#        for i in range(0,1,1):
#            plt.plot(rya,np.asarray(powrfl[Nt-1,i,:]),'k',linewidth=linw)
        #axis([0.,1.,0., np.amax(powden_RF_Nt+powden_NBI)+0.001])
        #axis([0.,1., 0.,0.35])
        savefig('profiles_powers.png')
        if isave_eps==1: savefig('profiles_powers.eps')
        show() 
        #stop

        #-----------------------------------------------------------------------
        fig7=plt.figure()   # Power profiles: Coll. transfer to e
        powden_e =-powers[:,k-1,1-1, :] # type#1 is for coll.power to maxw. e 
        powden_i =-powers[:,k-1,2-1, :] # type#2 is for coll.power to maxw. i 
        # CAREFUL! THESE POWERS ARE NEGATIVE (RF heating of general species)!
        # WE REVERSED THEM TO MAKE POSITIVE.
        print 'powden_e shape',np.shape(powden_e)
        powden_e_Nt=powden_e[Nt-1,0,:] # last time step, 1st species
        powden_i_Nt=powden_i[Nt-1,0,:] # last time step, 1st species
        powden_e_Nt=np.asarray(powden_e_Nt)
        powden_i_Nt=np.asarray(powden_i_Nt)
        print 'powden_e_Nt shape',np.shape(powden_e_Nt), np.shape(rya)
        #'Vol int of FSA powers, respectively, to gen species k vs t' [Watt]
        powers_int_e=-powers_int[:,0,1-1] # type#1 is for coll.power to maxw. e 
        powers_int_i=-powers_int[:,0,2-1] # type#2 is for coll.power to maxw. i 
        plt.hold(True)
        plt.grid(True)
        powden_mx= 1.05*np.amax(powden_e_Nt)+0.001
        text(1.05,0.03*np.amax(powden_mx),r'$\rho$',fontsize=34)
        print 'powden_mx_e[Watt/cm^3]=',powden_mx
        dy=powden_mx/(Nt+40)
        plt.xlim([rho_min,rho_max])
        plt.ylim([0., powden_mx])
        if np.amax(powden_mx)>0.01:
            plt.title('   $FSA$  $p_{e}$  $(W/cm^3)$',fontsize=28,y=1.03)
            it_count=0 # for text
            for it in range(0,Nt,it_stride):
                it_count=it+1
                powden_e_it=powden_e[it,0,:] #each t step, 1st species (k-1=0)
                plt.plot(rya,np.asarray(powden_e_it),'g',linewidth=linw)
                power_W= np.asscalar(powers_int_e[it])
                time_ms= time[it]*1e3
                print 'time[ms]=',time_ms,' power_W=',power_W
                if it>0:
                    txt1=r"$P_e[W]=$"+r"$%1.4e$" %(power_W)
                    txt2= r" $t[ms]=$"+r"$%1.3f$" %(time_ms)
                    ypos_txt= powden_mx-dy-dy*it_count
                    plt.text(0.3, ypos_txt, txt1 , va='center',fontsize=fnt-4)
                    plt.text(0.7, ypos_txt, txt2 , va='center',fontsize=fnt-4) 
            #Red - For the last time slice:
            plt.plot(rya,np.asarray(powden_e_Nt),'r',linewidth=linw*3)       
        savefig('profiles_powers_Coll_to_e.png')
        if isave_eps==1: savefig('profiles_powers_Coll_to_e.eps')
        show()

        #-----------------------------------------------------------------------
        fig7=plt.figure()   # Power profiles: Coll. transfer to i
        plt.hold(True)
        plt.grid(True)
        powden_mx= 1.05*np.amax(powden_i_Nt)+0.001
        text(1.05,0.03*np.amax(powden_mx),r'$\rho$',fontsize=34)
        print 'powden_mx_i[Watt/cm^3]=',powden_mx
        dy=powden_mx/(Nt+40)
        plt.xlim([rho_min,rho_max])
        plt.ylim([0., powden_mx])
        if np.amax(powden_mx)>0.01:
            plt.title('   $FSA$  $p_{i}$  $(W/cm^3)$',fontsize=28,y=1.03)
            it_count=0 # for text
            for it in range(0,Nt,it_stride):
                it_count=it+1
                powden_i_it=powden_i[it,0,:] #each t step, 1st species (k-1=0)
                plt.plot(rya,np.asarray(powden_i_it),'g',linewidth=linw)
                power_W= np.asscalar(powers_int_i[it])
                time_ms= time[it]*1e3
                print 'time[ms]=',time_ms,' power_W=',power_W
                if it>0:
                    txt1=r"$P_i[W]=$"+r"$%1.4e$" %(power_W)
                    txt2= r" $t[ms]=$"+r"$%1.3f$" %(time_ms)
                    ypos_txt= powden_mx-dy-dy*it_count
                    plt.text(0.3, ypos_txt, txt1 , va='center',fontsize=fnt-4)
                    plt.text(0.7, ypos_txt, txt2 , va='center',fontsize=fnt-4) 
            #Red - For the last time slice:
            plt.plot(rya,np.asarray(powden_i_Nt),'r',linewidth=linw*3)       
        savefig('profiles_powers_Coll_to_i.png')
        if isave_eps==1: savefig('profiles_powers_Coll_to_i.eps')
        show()
      


        #--------------------------------------------------------------------------
        if iplot_RF>0:
            powrf_max=0.
            powrfl_max=0.
            if iplot_powrfl:
                powrfl=array(s_file_cql3d.variables['powrfl']) # pow_lin(time,type,lr)
                nmodsa=np.size(powrfl,1) # Number of modes (harmonic*wave types)
                # Presently ALL nmodsa values are saved into 'powrf','powrfl','powrfc'
                # NEEDS ADJUSTMENT !
                # See 'rfpwr' - only mrfn modes are saved (from nharm1 to nharm1+nharms-1 )
                print "powrfl:", powrfl.shape   #,powrfl[Nt-1,0,:],
                print "nmodsa=",nmodsa
                #stop
                fig7=plt.figure()   # Power profiles:  Linear damping (powrfl)
                plt.hold(True)
                plt.grid(True)
                plt.xlim([rho_min,rho_max])
                powrfl_max=np.amax(powrfl)*1e3 # converted to kW/m^3
                if powrfl_max>0:
                    plt.ylim([0,powrfl_max*1.05])
                text(1.02,0.03*powrfl_max,r'$\rho$',fontsize=34)
                plt.title('   $FSA$  $p_{lin.damping,e}$  $(kW/m^3)$',fontsize=28,y=1.04)
                krf=0
                Plin_it1=powrfl[1,krf,:]*1e3  # initialize
                Plin_Nt= powrfl[1,krf,:]*1e3  # initialize
                for krf in range(0,nmodsa,1):    # pow_lin(time,nmodsa,lr)
                    powrfl_krf_peak=np.amax(powrfl[:,krf,:])
                    print "krf, RF_Lin.damping(krf)_max [W/cm^3]",krf+1,powrfl_krf_peak
                    if powrfl_krf_peak>1e-10:
                        #Plin=powrfl[0,krf,:]*1e3  # it=0 DO NOT USE - it is zero
                        #plt.plot(rya,Plin,'c',linewidth=linww*4)
                        Plin_it1=powrfl[1,krf,:]*1e3  # it=1
                        for it in range(1,Nt,1):
                            Plin=powrfl[it,krf,:]*1e3     
                            # P_RF_lin(time,lr), converted to kW/m^3
                            linww=linw  #*(Nt-it)
                            plt.plot(rya,Plin,'g',linewidth=linww)
                        Plin_Nt=Plin # the last t-step
                        plt.plot(rya,Plin,'r',linewidth=linw*3) # the last t-step: RED
                        #plt.plot(rya,Plin_it1-Plin_Nt,'b',linewidth=linw*3) # DIFFERENCE
                savefig('profiles_powrfl_RF_lin.png')
                if isave_eps==1: savefig('profiles_powrfl_RF_lin.eps')
                show() 

            #.....................................................
            if iplot_powrf:       
                powrf=array(s_file_cql3d.variables['powrf']) # powrf(time,krf,lr)
                # Presently ALL nmodsa values are saved into 'powrf','powrfl','powrfc'
                # NEEDS ADJUSTMENT !
                # See 'rfpwr' - only mrfn modes are saved (from nharm1 to nharm1+nharms-1 )
                nmodsa=np.size(powrf,1) # Number of modes (harmonic*wave types)
                powrfc=array(s_file_cql3d.variables['powrfc']) # powrfc(time,krf,lr)
                print "powrf:",  powrf.shape 
                print "powrfc", powrfc.shape
                fig7=plt.figure()   # Power profiles:  (powrf)
                subplot(2,1,1) #---------
                plt.title('  $p_{rf,ql}(red)$ $and$ $p_{rf,coll}(blue)$  $(W/cm^3)$',fontsize=28,y=1.07)    
                plt.hold(True)
                plt.grid(True)
                plt.minorticks_on() # To add minor ticks
                plt.tick_params(which='both',  width=1)
                plt.tick_params(which='major', length=7)
                plt.tick_params(which='minor', length=4, color='k')
                plt.xlim([rho_min,rho_max])
                plt.plot([0.,1.],[0.,0.],'k')
                #plt.title(' $FSA$  $p_{rf,ql}$ $and$ $p_{rf,coll}$  $(W/cm^3)$',fontsize=28,y=1.04)
                # SUM over krf modes, initialize:
                powrf_sumkrf_Nt=  powrf[Nt-1,0,:]*0 # function of rho, at last t-step.
                powrfc_sumkrf_Nt= powrfc[Nt-1,0,:]*0 # function of rho, at last t-step.
                for krf in range(0,nmodsa-1,1):    # powrf(time,nmodsa,lr)
                    powrf_krf_peak= np.amax(powrf[Nt-1,krf,:])
                    powrf_sumkrf_Nt= powrf_sumkrf_Nt+ powrf[Nt-1,krf,:]
                    powrfc_sumkrf_Nt= powrfc_sumkrf_Nt+ powrfc[Nt-1,krf,:]
                    #print "krf, RF.damping(krf)_max [W/cm^3]",krf+1,powrf_krf_peak
                    if powrf_krf_peak>1e-20:
                        Pgen_it1=powrf[1,krf,:] #*1e3: # it=1
                        for it in range(1,Nt,1) :
                            Prf=powrf[it,krf,:] #*1e3  #to convert to kW/m^3
                            Pgen=Prf # Prf(time,mode,lr), 
                            Prfc=powrfc[it,krf,:] #*1e3  #to convert to kW/m^3
                            linww=linw  #*(Nt-it)
                            #plt.plot(rya,Prf,'g',linewidth=linww)
                        Pgen_Nt=Pgen # the last t-step
                        plt.plot(rya,Prf,'r',linewidth=linw*2) # the last t-step: RED
                # Plot the sum over krf modes, at last t-step:
                plt.plot(rya,powrf_sumkrf_Nt,'r-o',linewidth=linw*2) # 
                powrf_max=np.amax(powrf_sumkrf_Nt)  #*1e3 # converted to kW/m^3
                powrf_min=np.amin(powrf_sumkrf_Nt)  #*1e3
                powrf_min=min(0.,powrf_min)
                text(1.02,0.03*(powrf_max-powrf_min),r'$\rho$',fontsize=34)
                if powrf_max-powrf_min > 0:
                    plt.ylim([powrf_min,powrf_max*1.05])
                #plt.ylim([-0.0199,0.04])
                Prf_tot= np.dot(powrf_sumkrf_Nt,dvol) # Watt
                
                subplot(2,1,2) #--------- 
                #plt.title('  $p_{rf,ql}(red)$ $and$ $p_{rf,coll}(blue)$  $(W/cm^3)$',fontsize=28,y=1.06)    
                plt.hold(True)
                plt.grid(True)
                plt.minorticks_on() # To add minor ticks
                plt.tick_params(which='both',  width=1)
                plt.tick_params(which='major', length=7)
                plt.tick_params(which='minor', length=4, color='k')
                plt.xlim([rho_min,rho_max])
                plt.plot([0.,1.],[0.,0.],'k')
                plt.plot(rya,powrfc_sumkrf_Nt,'b-o',linewidth=linw*2) # 
                powrfc_max=np.amax(powrfc_sumkrf_Nt)  #*1e3 # converted to kW/m^3
                powrfc_min=np.amin(powrfc_sumkrf_Nt)  #*1e3
                powrfc_min=min(0.,powrfc_min)
                text(1.02,0.03*(powrfc_max-powrfc_min),r'$\rho$',fontsize=34)
                if powrfc_max-powrfc_min > 0:
                    plt.ylim([powrfc_min,powrfc_max*1.05])
                plt.ylim([0.,1.2])
                Prfc_tot=np.dot(powrfc_sumkrf_Nt,dvol) # Watt
                print 'Prf_tot,Prfc_tot[W]=',Prf_tot,Prfc_tot
                # Adjust the last point in rho - it contains accumulated P from 
                # rays outside of rho=1 .
                print powrfc_sumkrf_Nt[lrz-1] # The last point on rya grid
                powrfc_sum= powrfc_sumkrf_Nt # Copy from complete array
                powrfc_sum[lrz-1]= powrfc_sum[lrz-2] # "project" value from previous pt
                print 'After adjustment: Prfc at rho~1 point:',powrfc_sum[lrz-1]
                Prfc_tot_inside=np.dot(powrfc_sum,dvol) # Watt
                print 'Prfc_tot_inside(rho<1)=',Prfc_tot_inside ,' W'
                savefig('profiles_powrf.png')
                if isave_eps==1: savefig('profiles_powrf.eps')
                show() 
                #stop
            
            
                fig7=plt.figure()   # Power profiles:  
                # Linear damping          (powrfl_t0 - powrfl_tend)*dvol       and 
                # General-species damping (powrf_tend - powrf_t0)*dvol
                # 1).  (Plin_it1-Plin_Nt)*dvol
                if iplot_powrfl:
                    Plin_dvol= (Plin_it1-Plin_Nt)*(dvol*1e-6) # 1e-6 is cm^3->m^3 convert
                    for ir in range(0,lrz,1):
                        print ir,' dPlin*dvol(kW)=',1e-6*(Plin_it1[ir]-Plin_Nt[ir])*dvol[ir]
                    print 'sum(dPlin*dvol) (kW) ==', np.sum(Plin_dvol)
                # 2). (Pgen_Nt-Pgen_it1)*dvol == Pgen_Nt*dvol
                Pgen_dvol= (1e-3*powrf_sumkrf_Nt)*dvol #(kW/cm^3)*(cm^3) == kW
                for ir in range(0,lrz,1):
                    print ir,' Pgen*dvol(kW)=',Pgen_dvol[ir]
                print 'sum(Pgen*dvol) (kW) ==', np.sum(Pgen_dvol)
                plt.hold(True)
                plt.grid(True)
                plt.xlim([rho_min,rho_max])
                pgen_dvol_mx=np.max(Pgen_dvol)
                pgen_dvol_mn=np.min(Pgen_dvol)
                txt_pos= pgen_dvol_mn+0.03*(pgen_dvol_mx-pgen_dvol_mn)
                text(1.02,txt_pos,r'$\rho$',fontsize=34)
                plt.xlabel(r'$\rho$',fontsize=28)
                plt.title('$Red:p_{rf}dvol,$ $Blue:(p_{lin,e,t0}-p_{lin,e,tend})dvol$  $(kW)$',fontsize=24,y=1.03)
                if iplot_powrfl: 
                    plt.plot(rya,Plin_dvol,'b-o',linewidth=linw*3)
                plt.plot(rya,Pgen_dvol,'r-o',linewidth=linw*3)
                savefig('profiles_powrf_dvol.png')
                show()
                #stop
            

            #--------------------------------------------------------------------------
            # rfpwr array contains:   --------------------------------------
            #  powrf(*,1:mrfn,*) for all time steps and radial points,
            #  followed by total powrft,
            #  followed by Summed rf+nbi pwr den: sorpwt,
            #  followed by Radially integrated: sorpwti.
            #  So, the size of rfpwr is mrfn+3 ;  Subtract 3:
            mrfn=np.size(rfpwr,1)-3   # (,1) means second argument position
            #
            #plt.xlabel(r'$\rho$',fontsize=30)
            #plt.ylabel('$<P_{RF+NBI}>$',fontsize=22)
            #for it in range(Nt-1,Nt,1):  # use (2,Nt,1) to skip first 2 steps
                #Pwr=rfpwr[it,3,:] # 0:2 -> RF from harmonics 1,2,3; 
                # 3-> RF+NBI; 4-> RF+NBI integrated in rho
                #plt.plot(rya,Pwr,'r',linewidth=linw*2)
                #plt.plot(rya,Pwr,'k',linewidth=linw*2)  
            print "rfpwr:", rfpwr.shape, '   mrfn=', mrfn
            #'rfpwr' - only mrfn modes are saved (from nharm1 to nharm1+nharms-1 )
            
            if ngen==1: # one gen.species - sum all krf modes
                rfpwr_sumkrf= rfpwr[:,0,:]*0 # function of time and rho
                for krf in range(0,mrfn,1):    # rfpwr(time,mrfn+3,lr)
                    rfpwr_krf_peak= np.amax(rfpwr[:,krf,:])
                    rfpwr_sumkrf= rfpwr_sumkrf+ rfpwr[:,krf,:]
                    print "krf, rfpwr.damping(krf)_max [W/cm^3]",krf+1,rfpwr_krf_peak
                fig1=plt.figure()   # Power profiles
                plt.title('   $FSA$  $p_{rf}$ $(kW/m^3)$ $for$ $k=1$',fontsize=28,y=1.03)
                plt.hold(True)
                plt.grid(True)
                for it in range(1,Nt,1):
                    Psumkrf=rfpwr_sumkrf[it,:]*1e3 # W/cm^3 to kW/m^3  
                    linww=linw  #*(Nt-it)
                    plt.plot(rya,Psumkrf,'g',linewidth=linww)
                plt.plot(rya,Psumkrf,'r',linewidth=linw*3) # the last t-step: RED
            
                print "rfpwr_sumkrf_max [W/cm^3] over rho",np.amax(rfpwr_sumkrf)
                print "np.amax(rfpwr[Nt-1,mrfn-1,:])=", np.amax(rfpwr[Nt-1,mrfn-1,:])
                rfpwr_max=np.amax(rfpwr_sumkrf)*1e3 # converted to kW/m^3
                rfpwr_min=np.amin(rfpwr_sumkrf)*1e3 # converted to kW/m^3
                rfpwr_min=min(0.,rfpwr_min)
                text(1.02,0.03*rfpwr_max,r'$\rho$',fontsize=34)
                if rfpwr_max>0:
                    plt.ylim([rfpwr_min,rfpwr_max*1.05])
                #plt.ylim((0.,0.8))
                savefig('profiles_rfpwr_k0_allkrf.png')
                show() 
                    
            if ngen==2: 
                # TWO gen.species - sum all krf modes except last, for ions,
                # and remaining - is for electrons
                rfpwr_sumkrf_k0= rfpwr[:,0,:]*0 # function of time and rho
                rfpwr_sumkrf_k1= rfpwr[:,0,:]*0 # function of time and rho
                for krf in range(0,mrfn-1,1):    # rfpwr(time,mrfn+3,lr)
                    # For tandem i+e:
                    #sum-up in range (0,mrfn-1,1) for ions;
                    #sumup  in range (mrfn-1,mrfn,1) for electrons (one value)
                    rfpwr_krf_peak= np.amax(rfpwr[:,krf,:])
                    rfpwr_sumkrf_k0= rfpwr_sumkrf_k0+ rfpwr[:,krf,:]
                    print "krf, rfpwr.damping(krf)_max [W/cm^3]",krf+1,rfpwr_krf_peak
                krf=mrfn-1 # the last krf in code: presumably for electrons
                rfpwr_sumkrf_k1= rfpwr[:,krf,:]
                # Make TWO plots in this case (ngen=2)
                fig1=plt.figure()   # Power profiles
                plt.title('   $FSA$  $p_{rf}$ $(kW/m^3)$ $for$ $ions$',fontsize=28,y=1.03)
                plt.hold(True)
                plt.grid(True)
                for it in range(1,Nt,1):
                    Psumkrf=rfpwr_sumkrf_k0[it,:]*1e3 # W/cm^3 to kW/m^3  
                    linww=linw  #*(Nt-it)
                    plt.plot(rya,Psumkrf,'g',linewidth=linww)
                plt.plot(rya,Psumkrf,'r',linewidth=linw*3) # the last t-step: RED            
                rfpwr_max=np.amax(rfpwr_sumkrf_k0)*1e3 # converted to kW/m^3
                text(1.02,0.03*rfpwr_max,r'$\rho$',fontsize=34)
                if rfpwr_max>0:
                    plt.ylim([0,rfpwr_max*1.05])
                #plt.ylim((0.,0.8))
                savefig('profiles_rfpwr_k0_allkrf.png')
                show() 
                fig1=plt.figure()   # Power profiles
                plt.title('   $FSA$  $p_{rf}$ $(kW/m^3)$ $for$ $e$',fontsize=28,y=1.03)
                plt.hold(True)
                plt.grid(True)
                for it in range(1,Nt,1):
                    Psumkrf=rfpwr_sumkrf_k1[it,:]*1e3 # W/cm^3 to kW/m^3  
                    linww=linw  #*(Nt-it)
                    plt.plot(rya,Psumkrf,'g',linewidth=linww)
                plt.plot(rya,Psumkrf,'r',linewidth=linw*3) # the last t-step: RED            
                rfpwr_max=np.amax(rfpwr_sumkrf_k1)*1e3 # converted to kW/m^3
                text(1.02,0.03*rfpwr_max,r'$\rho$',fontsize=34)
                if rfpwr_max>0:
                    plt.ylim([0,rfpwr_max*1.05])
                #plt.ylim((0.,0.8))
                savefig('profiles_rfpwr_k1_allkrf.png')
                show() 
            #stop
            
            #--------------------------------------------------------------------------
            fig1=plt.figure()   # Power profiles as a func of krf
            plt.title('$P_{rf}$ $(kW)$ $vs$ $harmonic$ $number$',fontsize=28,y=1.025)
            plt.hold(True)
            plt.grid(True)
            nharm1=1 #13 #9 #17 #9 for 500MHz/ITER # not in mnemonic.nc (it is in *rf.nc) Get it from cqlinput
            harmonic= np.arange(0,mrfn,1)+nharm1
            Prf_krf= harmonic*0
            print 'harmonic=', np.shape(harmonic), harmonic
            for krf in range(0,mrfn,1):    
                Prf_krf[krf]= np.dot(rfpwr[Nt-1,krf,:],dvol[:])
            print np.shape(Prf_krf), np.sum(Prf_krf)*1e-3, ' kW'
            plt.plot(harmonic, Prf_krf*1e-3, '-o', linewidth=linw)
            plt.xlim((nharm1-1,nharm1+mrfn))
            plt.xlabel(' harmonic number ')
            savefig('profiles_rfpwr_vs_mode.png')
            show() 

            #stop

#--------------------------------------------------------------------------
            fig7=plt.figure() # Power profiles:  General species damping (powden_RF)
            plt.hold(True)
            plt.grid(True)
            plt.xlim([rho_min,rho_max])
            powdenRF_max=np.amax(powden_RF)*1e3 # converted to kW/m^3
            powdenRF_min=np.amin(powden_RF)*1e3 # converted to kW/m^3
            powdenRF_min=min(0.,powdenRF_min)
            print 'np.shape(powden_RF)', np.shape(powden_RF)
            if powdenRF_max>0:
                plt.ylim([powdenRF_min,powdenRF_max*1.05])
            text(1.02,0.03*powdenRF_max,r'$\rho$',fontsize=34)
            plt.title('   $FSA$  $p_{damping}$  $(kW/m^3)$',fontsize=28,y=1.04)
            # powden_RF(time,lr)
            #print "krf, RF_Lin.damping(krf)_max [W/cm^3]",krf+1,powrfl_krf_peak
            if powdenRF_max>1e-10:
                for it in range(1,Nt,1):
                    Pgen=powden_RF[it,0,:]*1e3  # 0 is for species. NEED TO FIX 
                    # converted to kW/m^3
                    linww=linw  #*(Nt-it)
                    plt.plot(rya,Pgen,'g',linewidth=linww)
                plt.plot(rya,Pgen,'r',linewidth=linw*3) # the last t-step: RED
            savefig('profiles_power_RF_gen.png')
            if isave_eps==1: savefig('profiles_power_RF_gen.eps')
            show() 
            #stop
        

        #--------------------------------------------------------------------------
        fig7=plt.figure()   # Fusion Power profile
        #plt.subplot(224) 
        plt.hold(True)
        plt.grid(True)
        text(1.04,0,r'$\rho$',fontsize=34)
        plt.title('     $Fusion$ $power$  $(W/cm^3)$',fontsize=28,y=1.03)
        if i_fus==1:
            plot_max=1.05*np.amax(fuspwrv[:,:]) # W/cm^3 
            for knumb in range(0,4,1):  # Reaction type
                if knumb==0: 
                    col='r'  # D+T   -> alpha + n(14.1)
                    txt= '$D+T$'+r'$\rightarrow$'+r'$\alpha$' +'$+n$      P='+\
                    "%.2f" %(fuspwrvt[knumb]/1e6) +'$MW$'
                if knumb==1: 
                    col='g'  # D+He3 -> alpha + p
                    txt= '$D+He^3$'+r'$\rightarrow$'+r'$\alpha$'+'$+p$    P='+\
                    "%.2f" %(fuspwrvt[knumb]/1e6) +'$MW$'
                if knumb==2: 
                    col='b'  # D+D   -> T + p
                    txt= '$D+D$'+r'$\rightarrow$'+'$T$'+'$+p$        P='+\
                    "%.2f" %(fuspwrvt[knumb]/1e6) +'$MW$'
                if knumb==3: 
                    col='k'  # D+D   -> He3 + n(2.45)
                    txt= '$D+D$'+r'$\rightarrow$'+'$He^3$'+'$+n$    P='+\
                    "%.2f" %(fuspwrvt[knumb]/1e6) +'$MW$'
                Fus_pwr=np.asarray(fuspwrv[knumb,:]) # W/cm^3
                Fus_pwr_max= np.amax(Fus_pwr)
                if Fus_pwr_max>1e-3*plot_max:
                    plt.plot(rya,Fus_pwr,color=col,linewidth=linw*2)
                    text(.1,(0.8-0.1*knumb)*plot_max,txt,fontsize=24,color=col)
                    
            axis([0.,1., 0.,plot_max])
            #axis([0.,1., 0.,0.35])
        savefig('profiles_FUSpower.png')
        if isave_eps==1: savefig('profiles_FUSpower.eps')
        show() 

#        stop
        
        #gammac= sqrt(1+(x*unorm)**2/clight**2)  
        #if (gammac-1) <= 1.e-6,
        #Enrgy_j= 0.5*fmass*(x*unorm)^2/ergtkev  # % keV
        #Enrgy_j= (gammac-1)*fmass[k-1]*clight**2/ergtkev   
        #for j in range(0,jx,1):
        #    print 'j,Enrgy_j=',j,Enrgy_j[j]
            
        jmn1=0
        jmx1=min(10,jx) # min/max for the range of j, to calc. density
        jmn2=min(10,jx)
        jmx2=min(20,jx) # min/max for the range of j, to calc. density
        jmn3=min(20,jx)
        jmx3=min(30,jx) # min/max for the range of j, to calc. density
        jmn4=min(30,jx)
        jmx4=min(120,jx) # min/max for the range of j, to calc. density

        dx=s_file_cql3d.variables['dx']
        cur0_fba=np.zeros([lrz])
        cur0_sum=np.zeros([lrz])
        curFSA_sum=np.zeros([lrz])
        cur0_floc=np.zeros([lrz])
        den_fba=np.zeros([lrz])
        den_fba1=np.zeros([lrz])
        den_fba2=np.zeros([lrz])
        den_fba3=np.zeros([lrz])
        den_fba4=np.zeros([lrz])
        den_floc=np.zeros([lrz])
        den_floc1=np.zeros([lrz])
        den_floc2=np.zeros([lrz])
        den_floc3=np.zeros([lrz])
        den_floc4=np.zeros([lrz])
        f=s_file_cql3d.variables['f']
        print 'ngen=',ngen
        print "f.shape=",f.shape, ' time_select.shape=',time_select.shape
        nselect=len(time_select)  #  len(f[:,0,0,0])
        print 'nselect=',nselect
        
        curr_v=zeros((jx,lrz)) # to form specific current density 
        energy_v=zeros((jx,lrz)) # to form specific energy
        if iden==1:
            #for it in range(0,nselect,1):
            #    print 'time_select, min/max of f=',time_select[it],np.min(f[it,:,:,:]),np.max(f[it,:,:,:])
        
            ksp=1
            k=ksp-1
            for i_R in range(1,lrz+1,1):
                lr=i_R-1
                if ngen>1:
                    fba= s_file_cql3d.variables['f'][k,lr,:,:]
                else:  # only one gen.species
                    fba= s_file_cql3d.variables['f'][lr,:,:]
                if Coeff=='f_local':
                    floc=s_file_cql3d.variables['forbshift'][lr,:,:]
                else:
                    floc=fba    
                    
                cynt2=s_file_cql3d.variables['cynt2'][lr,:] # 2pi*sin(theta)dtheta
                cint2=s_file_cql3d.variables['cint2'][:] # x^2*dx
                y=s_file_cql3d.variables['y'][lr,:]
                tauv=s_file_cql3d.variables['tau'][lr,:]
                coss=cos(y)  # cos(theta)
                cynt2coss=cynt2*coss # element-by-element [i]
                cint2x=cint2*x/gammac # element-by-element [j]
                cynt2cosstau=abs(cynt2coss)*tauv
                #print cynt2.shape,coss.shape,cynt2coss.shape
                #print cint2.shape,x.shape,cint2x.shape
                cur0_fba[lr]= np.dot(cint2x,np.dot(fba, cynt2coss)) # scalar
                #if Coeff=='f_local':
                cur0_floc[lr]=np.dot(cint2x,np.dot(floc,cynt2coss))
                cur0_floc[lr]=cur0_floc[lr]*bnumb[k]*charge*unorm/3.e9  
                jrng=range(jmn1,jmx1,1)
                den_floc1[lr]=np.dot(cint2[jrng],np.dot(floc,cynt2)[jrng])
                jrng=range(jmn2,jmx2,1)
                den_floc2[lr]=np.dot(cint2[jrng],np.dot(floc,cynt2)[jrng])
                jrng=range(jmn3,jmx3,1)
                den_floc3[lr]=np.dot(cint2[jrng],np.dot(floc,cynt2)[jrng])
                jrng=range(jmn4,jmx4,1)
                den_floc4[lr]=np.dot(cint2[jrng],np.dot(floc,cynt2)[jrng])
                den_floc[lr]=np.dot(cint2,np.dot(floc,cynt2))
                print 'rya,cur0_floc=',rya[lr],cur0_floc[lr]
                    
                for j in range(0,jx):
                    factor=1. #(clight/unorm)*gammac[j]/x[j]
                    curr_v[j,lr]=factor*(cint2x[j]/dx[j])*np.dot(floc,cynt2coss)[j]
                    #specific current density at the midplane
                    energy_v[j,lr]= factor*(cint2[j]/dx[j])*np.dot(floc,cynt2)[j]*Enrgy_j[j,k]
                    #specific energy [keV/cm3]  at the midplane
                    
                curr_v[:,lr]= curr_v[:,lr]*bnumb[k]*charge*unorm/3.e9 #[A/cm2]
                cur0_fba[lr]= cur0_fba[lr]*bnumb[k]*charge*unorm/3.e9   
                if lr>lrz-1:
                    cur0_sum[lr]=  cur0_sum[lr-1] + cur0_fba[lr]*darea[lr] # cum.current [A] 
                    curFSA_sum[lr]=curFSA_sum[lr-1]+ curr_k[Nt-1,lr]*darea[lr] # cum.current [A] 
                else:  # lr=0:
                    cur0_sum[lr]=   cur0_fba[lr]*darea[lr]
                    curFSA_sum[lr]= curr_k[Nt-1,lr]*darea[lr]
               
                den_fba[lr]= np.dot(cint2,np.dot(fba,cynt2))
                jrng=range(jmn1,jmx1,1)
                den_fba1[lr]= np.dot(cint2[jrng],np.dot(fba,cynt2)[jrng])
                jrng=range(jmn2,jmx2,1)
                den_fba2[lr]= np.dot(cint2[jrng],np.dot(fba,cynt2)[jrng])
                jrng=range(jmn3,jmx3,1)
                den_fba3[lr]= np.dot(cint2[jrng],np.dot(fba,cynt2)[jrng])
                jrng=range(jmn4,jmx4,1)
                den_fba4[lr]= np.dot(cint2[jrng],np.dot(fba,cynt2)[jrng])
                #print i_R_start, lr
                print 'rya[lr],cur0_fba[A/cm2],cur0_sum[A],den_floc[cm-3],reden[Nt-1,lr,k]=',\
                rya[lr],cur0_fba[lr],cur0_sum[lr],den_floc[lr],reden[Nt-1,lr,k]
                if ngen>1: #Note: the style of saving data in netcdfrw2 is diff for ngen>1 
                    curr_k=curr[:,k,:]  # [Nt,ngen,lrz]
                else: # ngen=1 (one general species)
                    curr_k=curr[:,:]
                # From next two lines, curr_k~~np.dot(dx,curr_v[:,lr]) (with some accuracy)
                print 'rya[lr],curr[lr]         =',rya[lr], curr_k[Nt-1,lr]
                print 'rya[lr],np.dot(curr_v,dx)=',rya[lr], np.dot(dx,curr_v[:,lr])
                # These two lines give same printout:
                print 'rya[lr],np.dot(energy_v,dx)/den_floc[lr]=',rya[lr],np.dot(dx,energy_v[:,lr])/den_floc[lr]
                print 'rya[lr],energym[Nt-1,lr] (keV)=',rya[lr],energym[Nt-1,lr]
            

        
            #--------------------------------------------------------------------------
            fig1=plt.figure()   # Current density profile based on f_BA here
            title('  $Current$'+'  '+'$profile$'+'  '+'$from$'+'  '+'$f_{soln}$',y=1.03)
            plt.hold(True)
            plt.grid(True)
            plt.xlabel(r'$\rho$',fontsize=28)
            plt.ylabel('$j_{||}$ $at$ $midplane$'+'   '+'$(A/cm^2)$',fontsize=28)
            plt.plot(rya,cur0_fba,'r',linewidth=linw)
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            plt.xlim([rho_min,rho_max])
            if (curr_max-curr_min > 0.):
                plt.ylim([curr_min,curr_max])
            #axis([0,1,min(min(cur0_fba),0),max(cur0_fba)])
            #axis([0,1,-2,16])
            if isave_eps==1: savefig('profile_Jpar_midpl_fba.eps')
            savefig('profile_Jpar_midpl_fba.png')
            #show() 
            #--------------------------------------------------------------------------
            fig1=plt.figure()   # Cum.Current based on f_BA 
            title('  $Cum.Current.$  $Red:j_{midplane};$ $Blue:<j>_{FSA}$',y=1.03)
            plt.hold(True)
            plt.grid(True)
            plt.xlabel(r'$\rho$',fontsize=28)
            plt.ylabel('$Cum.Current$ $I_{||}$'+'  '+'$(kA)$',fontsize=28)
            if (curr_max-curr_min > 0.):
                plt.ylim([curr_min,curr_max])
            plt.plot(rya,cur0_sum/1e3,'r',linewidth=linw) #midplane, from cur0_fba
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            plt.plot(rya,curFSA_sum/1e3,'b',linewidth=linw) #FSA, based on curr
            plt.xlim([rho_min,rho_max])
            #axis([0,1,min(min(cur0_sum),0)/1e3,1.05*max(cur0_sum)/1e3])
            #axis([0,1,-2,10])
            if isave_eps==1: savefig('profile_Ipar_cum.eps')
            savefig('profile_Ipar_cum.png')
            #show() 
            #--------------------------------------------------------------------------
            fig1=plt.figure()   # Current profile based on f_local here
            title('  $Current$'+'  '+'$profile$'+'  '+'$from$'+'  '+'$f_{local}$',y=1.03)
            plt.hold(True)
            plt.grid(True)
            plt.xlabel(r'$\rho$',fontsize=28)
            plt.ylabel('$j_{||}$ $at$ $midplane$'+'   '+'$(A/cm^2)$',fontsize=28)
            if (curr_max-curr_min > 0.):
                plt.ylim([curr_min,curr_max])      
            plt.plot(rya,cur0_floc,'r',linewidth=linw)
            plt.plot([rho_min,rho_max],[0,0],'k',linewidth=linw)  # zero level
            plt.xlim([rho_min,rho_max])
            #axis([0,1,min(min(cur0_floc),0),max(cur0_floc)])
            #axis([0,1,-2,10])
            if isave_eps==1: savefig('profile_Jpar_midpl_floc.eps')
            savefig('profile_Jpar_midpl_floc.png')
            #show() 
        #--------------------------------------------------------------------------
        if iden==1:
            fig1=plt.figure()   # dens profile based on f_BA here
            title('  $Density$'+'  '+'$profile$'+'  '+'$from$'+'  '+'$f_{soln}$',y=1.03)
            plt.hold(True)
            plt.grid(True)
            plt.xlabel(r'$\rho$',fontsize=28)
            plt.ylabel('$n$ $at$ $midplane$'+'   '+'$(cm^{-3})$',fontsize=28)
            plt.plot(rya,den_fba,'k',linewidth=linw)
            plt.plot(rya,den_fba1,'b',linewidth=linw*2)
            plt.plot(rya,den_fba2,'g',linewidth=linw*2)
            plt.plot(rya,den_fba3,'c',linewidth=linw*2)
            plt.plot(rya,den_fba4,'m',linewidth=linw*2)
            plt.xlim([rho_min,rho_max])
            #axis([0,1,0,max(den_fba)*1.05])
            #axis([0,1,0,2e13]) #axis([0,1,0,3.5e13])
            txt="$E$"+"$=$%3.1f" %(Enrgy_j[jmn1,k]) +"$-$"+\
            "%3.1f" %(Enrgy_j[jmx1,k]) + "$keV$"
            text(rya[i_R_start+3],den_fba1[i_R_start+3],txt,fontsize=18,color='b')        
            txt="$E$"+"$=$%3.1f" %(Enrgy_j[jmn2,k]) +"$-$"+\
            "%3.1f" %(Enrgy_j[jmx2,k]) + "$keV$"
            text(rya[i_R_start+6],den_fba2[i_R_start+6],txt,fontsize=18,color='g')        
            txt="$E$"+"$=$%3.1f" %(Enrgy_j[jmn3,k]) +"$-$"+\
            "%3.1f" %(Enrgy_j[jmx3,k]) + "$keV$"
            text(rya[i_R_start+9],den_fba3[i_R_start+9],txt,fontsize=18,color='c')        
            txt="$E$"+"$=$%3.1f" %(Enrgy_j[jmn4,k]) +"$-$"+\
            "%3.1f" %(Enrgy_j[jmx4,k]) + "$keV$"
            text(rya[i_R_start+9],den_fba4[i_R_start+9],txt,fontsize=18,color='m')        
            #axis([0,1,0,7])
            #axis([0,1,0,7])
            if isave_eps==1: savefig('profile_Nmidpl_fba.eps')
            savefig('profile_Nmidpl_fba.png')
            #show() 
            
            #--------------------------------------------------------------------------
            fig1=plt.figure()   # dens profile based on f_loc here
            title('  $Density$'+'  '+'$profile$'+'  '+'$from$'+'  '+'$f_{local}$',y=1.03)
            plt.hold(True)
            plt.grid(True)
            plt.xlabel(r'$\rho$',fontsize=28)
            plt.ylabel('$n$ $at$ $midplane$'+'   '+'$(cm^{-3})$',fontsize=28)
            plt.plot(rya,den_floc,'k',linewidth=linw)
            plt.plot(rya,den_floc1,'b',linewidth=linw*2)
            plt.plot(rya,den_floc2,'g',linewidth=linw*2)
            plt.plot(rya,den_floc3,'c',linewidth=linw*2)
            plt.plot(rya,den_floc4,'m',linewidth=linw*2)
            plt.xlim([rho_min,rho_max])
            #axis([0,1,0,max(den_floc)*1.05])
            #axis([0,1,0,2e13]) #axis([0,1,0,3.5e13])
            txt="$E$"+"$=$%3.1f" %(Enrgy_j[jmn1,k]) +"$-$"+\
            "%3.1f" %(Enrgy_j[jmx1,k]) + "$keV$"
            text(rya[i_R_start+3],den_floc1[i_R_start+3],txt,fontsize=18,color='b')        
            txt="$E$"+"$=$%3.1f" %(Enrgy_j[jmn2,k]) +"$-$"+\
            "%3.1f" %(Enrgy_j[jmx2,k]) + "$keV$"
            text(rya[i_R_start+6],den_floc2[i_R_start+6],txt,fontsize=18,color='g')        
            if isave_eps==1: savefig('profile_Nmidpl_floc.eps')
            savefig('profile_Nmidpl_floc.png')
            #show() 
        
    

#--------------------------------------------------------------------------
if ivel_plots==0:
    stop

#==============================================================================
#==============================================================================
#=========== 2D(vel.space) PLOTS of distr.func. or Diff.Coeffs ================



#--------------
if Coeff=='cqlb':
    D='Duu' # for name in file (saving plots)
    DC=r"$D_{uu}$"  # for title in plots
    stop=1  # Determines how much data to read from du0u0

#--------------
if Coeff=='cqlc':
    D='Dup' 
    DC=r"$D_{up}$"  # for title in plots
    stop=2   

#--------------
if Coeff=='cqle':
    D='Dpu_sin' 
    DC=r"$D_{pu}sin(\theta)$"  # for title in plots
    stop=3   

#--------------    
if Coeff=='cqlf':
    D='Dpp_sin' 
    DC=r"$D_{pp}sin(\theta)$"  # for title in plots
    stop=4

#--------------
if Coeff=='f':
    D='f' # for name in file (saving plots)
    DC=r"$log_{10}(f)$"  # for title in plots

#--------------
if Coeff=='f_local':
    D='f_local' # for name in file (saving plots)
    DC=r"$log_{10}(f)$"  # for title in plots

#--------------

if Coeff=='urfb':
    D='urfb_krf'+str(krf)  # for name in file (saving plots)
    DC=r"$D_{uu,QL}$"  # for title in plots

#--------------
        


#----------- Function to check s is number --------------------
#Checks if s is a number, from WWW (if not, could be string):
#NOT USED.
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
#----------- Function to check s is number --------------------


#----------- Function to read DC data --------------------------
def read_vector(flnm,ndim,nums_per_line):
    """
    Reads delimited items from an open file, flnm.
    Returns them in a vector of input length ndim.
    nums_per_line is (assumed constant) number of
    items per line (except last line may be shorter).
    """      
    a=np.ones(ndim)
    nlines=ndim/nums_per_line
    if nlines*nums_per_line != ndim: nlines=nlines+1
    #    print nlines
    for i in range(nlines):
        ibegin=i*nums_per_line
        iend=min(ibegin+nums_per_line,ndim)
        #print ibegin,iend
        a[ibegin:iend]=np.array(flnm.readline().split(),float)
    return a
#----------- Function to read data --------------------------




#elapsed_time = time.time() - e0
#cpu_time = time.clock() - c0
#
#print 'cpu time since start (sec.) =',  cpu_time
print 'STARTING LOOP in i_R (flux surfaces index)'
print '=========================================='

if plot_type == 'c':
    text_type='_contour'
elif plot_type == 'm':
    text_type='_mesh'

#set fonts and line thicknesses
fnt=fnt-6 #+2 # 20
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
    'font.weight'  : 'regular',
    'format' : '%0.1e'
}

pylab.rcParams.update(params)

#===========================================================================

#-------------------------------------------------------------------------
#------------ Fpar(upar/c) distr.func at all available radial surfaces
if iplot_fpar==1:
    xl=s_file_cql3d.variables['xl'][:]  # u_par/unorm, size=jfl
    fl=s_file_cql3d.variables['fl']  #[lr,:] is given radius
    jfl=len(xl)
    lrz=len(fl[:,0])
    print('shape of fl',shape(fl) )
    #print 'lrz=',lrz
    #print 'jfl=',jfl
    fl_mx=np.max(fl)
    print('min/max of fl at all saved lr', np.min(fl), fl_mx)
    fig1= plt.figure() #------------ Fpar(upar/c) distr.func
    ax = plt.subplot(1, 1, 1)
    plt.hold(True)
    plt.grid(True)
    plt.xlim((-ucmx,ucmx))   
    fl_mn= fl_mx/1.e7  #Set lower limit; In data, it can be 0.0 
    ir=0
    for lr in range(0,lrz,1):
        fl_lr=fl[lr,:]
        fl_lr_mx= np.max(fl_lr)
        fl_lr_mn= fl_lr_mx/1.e7
        if fl_lr_mx>0.0:
            for j in range(0,jfl,1):
                fl_= fl_lr[j]
                fl_lr[j]=max(fl_,fl_lr_mn)
            if remainder(ir,6)==0: col='b'
            if remainder(ir,6)==1: col='g'
            if remainder(ir,6)==2: col='r'
            if remainder(ir,6)==3: col='c'    
            if remainder(ir,6)==4: col='m' 
            if remainder(ir,6)==5: col='k' 
            ir=ir+1              
            #print 'lr=',lr,'  ir=',ir            
            plt.yscale('log') # log10 scale !Comment if you want a linear scale
            plt.plot(xl[:]*(unorm/clight),fl_lr[:],color=col,linewidth=linw)
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k') 
    plt.title("$F_{||}$  $(normalized$ $units)$",y=1.03)    
    xlabel(r"$u_{||}/c$",fontsize=fnt+6)        
    # Save as png plot:
    savefig('Fpar_vs_upar'+'.png')
    # Save as eps plot:
    if isave_eps==1: 
        savefig('Fpar_vs_upar'+'.eps')
    #-------------------- FIGURE IS SAVED -------------------------
    #show() # comment it, to avoid pressing "Enter" after each plot


# Loop in flux surface index i_R  starts here; 
# scanning i_R= i_R_start:i_R_stop
#
for i_R in range(i_R_start,i_R_stop+1,1):
#for i_R in range(2,3,1):
    lr=i_R-1
    #----------------------------
    if i_R<10:
        i_R_index = '00'+str(i_R)
    elif i_R<100:
        i_R_index = '0'+str(i_R) 
    else:
        i_R_index = str(i_R)
    #----------------------------
    text_trim=''
    if data_src=='DC_text':  

    # Form the filename to read:
        file_DC= '../du0u0_r'+str(i_R_index)  # name of file to read.
        # files du0u0_r### are produced by DC fortran code.
        if os.path.exists(file_DC+'.gz'):
            os.popen('gunzip '+file_DC+'.gz')
            igzip=1
        else:
            igzip=0
    #------------
        du0u0=open(file_DC,'r') 
    #------------
        nums_per_line=6  # format of data in the file: 6 columns (for Diff.Coeffs)
        rho_a=float(du0u0.readline())    #read_vector(du0u0,1,1)
        uprp=read_vector(du0u0,n_uprp,nums_per_line) # normalized to  0.0...+1.0
        upar=read_vector(du0u0,n_upar,nums_per_line) # normalized to -1.0...+1.0
        # Note: indices of uprp are 0 to n_uprp-1
        # In Python, indices start from 0
        #
        print 'i_R=',i_R,  '   rho_a=',rho_a, '   u/c max =',ucmx,\
        ' umx[cm/s]=',ucmx*clight
        #
        # Depending on which coefficient is selected, the data is read in a loop
        # until the section of data is reached for this specific coefficient
        icount=1
        while icount<= stop:
            rdc_cql=read_vector(du0u0,n_uprp*n_upar,nums_per_line)
            rdc_cql.shape=(n_upar,n_uprp)
            icount=icount+1
        #------------
        du0u0.close()
        if igzip==1: os.popen('gzip '+file_DC)
        #------------

        DDD= transpose(rdc_cql) # rows and columns in source file are reversed    
        # Spike removal according to "Running Median Filters and a General
        # Despiker", John R. Evans, Bull. Seismological Soc. Amer. (1982).
        #
        # Smoothing based on averaging over 9 neighboring points.
        
        #---> Trim the peaks.
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        trim1=5  # Trimming factor for diagonal Dii.     Recommended: 4-5
        med_size=3  #Number of points on either side of the mean.
        #  Recommended: 2-3.  Recommend 4 for Dpp from DC
        med_window=2*med_size+1
        # If a peak exceeds trim1*median(2*med_size+1 neighbors), trim it.
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        if itrim>0:
            text_trim='_trim' # part of filename to be saved
            
            med_out=np.zeros(n_upar)
                
            for j in range(0,n_uprp,1):
                for i in range(0,n_upar,1):
                    iimn= max(0,   i-med_size)       #limit at low i
                    iimn =min(iimn,n_upar-med_window)  #limit at high i
                    iimx= min(n_upar,i+med_size+1)   #limit at high i
                    iimx= max(iimx,med_window)         #limit at low i
#                    med_out[i]=np.median(DDD[iimn:iimx,j])  #np.median gives
                                                            #median value
#                    if abs(DDD[i,j])>trim1*abs(med_out[i]): DDD[i,j]=med_out[i]
#                    if abs(DDD[i,j])>DCmx: DDD[i,j]=np.sign(DDD[i,j])*DCmx
                    med_out[i]=np.median(DDD[j,iimn:iimx])  #np.median gives
                                                            #median value
                    if abs(DDD[j,i])>trim1*abs(med_out[i]): DDD[j,i]=med_out[i]
                    if abs(DDD[j,i])>DCmx: DDD[j,i]=np.sign(DDD[j,i])*DCmx

    if data_src=='cql3d_nc':
        #unorm=s_file_cql3d.variables['vnorm']
        #x=s_file_cql3d.variables['x']
        y=s_file_cql3d.variables['y'][lr,:]
        itl=s_file_cql3d.variables['itl'][lr]
        itu=s_file_cql3d.variables['itu'][lr]
        pchl = y[itl-1] # % pitch angle for t-p bndry
        pchu = y[itu-1] # % pitch angle for t-p bndry
        rho_a=rya[lr]
                
        print 'i_R=',i_R,  '   rho_a=',rho_a, '   u/c max =',ucmx

        #--------------
        if Coeff=='cqlb':
            DDD=s_file_cql3d.variables['rdcb'][lr,:,:]
            DDD=unorm4*symm*DDD

        #--------------
        if Coeff=='cqlc':
            DDD=s_file_cql3d.variables['rdcc'][lr,:,:]
            DDD=unorm3*symm*DDD

        #--------------
        if Coeff=='cqle':
            DDD=s_file_cql3d.variables['rdce'][lr,:,:]
            DDD=unorm3*symm*DDD

        #--------------    
        if Coeff=='cqlf':
            DDD=s_file_cql3d.variables['rdcf'][lr,:,:]
            DDD=unorm2*symm*DDD

        #--------------
        if Coeff=='f':
            if ngen>1: # more than one gen.species
                DDD=s_file_cql3d.variables['f'][k-1,lr,:,:]
            else:  # one gen.species
                DDD=s_file_cql3d.variables['f'][lr,:,:]
            print 'MAX/MIN of f: ', np.max(DDD),np.min(DDD)
            fmin= 1.0e-100  # 
            for j in range(jx):
                for i in range(iy):
                    if DDD[j,i] <= 0:  DDD[j,i]=fmin
            DDD=np.nan_to_num(np.log10(DDD))
            DDDmax=np.max(DDD)
            #DDDmin=DDDmax-20.

        #--------------
        if Coeff=='f_local':
            DDD=s_file_cql3d.variables['forbshift'][lr,:,:] #1st-FOW or Hybrid-FOW
            DDD=np.nan_to_num(np.log10(DDD))
            DDDmax=np.max(DDD)
            #DDDmin=DDDmax-20.
            #for j in range(jx):
            #    for i in range(iy):
            #        if DDD[j,i] < DDDmin:  DDD[j,i]=DDDmin
        #--------------
        if Coeff=='urfb':
            DDD=s_file_cql3d.variables['urfb'][lr,:,:]
            DDD= (unorm4*symm*DDD)**0.5

            
        #--------------
        DDD= transpose(DDD) # rows and columns in source array are reversed 
        (nrow,nclm)=shape(DDD)   # nrow=iy  nclm=jx
        #print nrow,nclm
        
        if vsign==-1.0:
            tmp=DDD
            iygrid=np.arange(0,iy,1)
            iyrevs=np.arange(iy-1,-1,-1)
            DDD[iygrid,:]=tmp[iyrevs,:]
  
        # Spike removal according to "Running Median Filters and a General
        # Despiker", John R. Evans, Bull. Seismological Soc. Amer. (1982).
        #
        # Smoothing based on averaging over 9 neighboring points.
        
        #---> Trim the peaks.
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        trim1=2#3  # Trimming factor for diagonal Dii.     Recommended: 4-5
        med_size=5  #Number of points on either side of the mean. 
        #  Recommended: 3.  Recommend 5 for Dpp
        med_window=2*med_size+1
        # If a peak exceeds trim1*median(2*med_size+1 neighbors), trim it.
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #---> Fill-in isolated holes
        hole=0.9  # If local point of f is less than hole*mean(f), it's filled
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if itrim>0:
            text_trim='_trim' # part of filename to be saved
            
            med_out=np.zeros(iy)
            
            for j in range(0,jx,1):
                for i in range(0,iy,1):
                    iimn= max(0,   i-med_size)       #limit at low i
                    iimn =min(iimn,iy-med_window)    #limit at high i
                    iimx= min(iy,i+med_size+1)       #limit at high i
                    iimx= max(iimx,med_window)       #limit at low i
                    med_out[i]=np.median(DDD[iimn:iimx,j])  #np.median gives
                                                            #median value
                    if abs(DDD[i,j])>trim1*abs(med_out[i]): DDD[i,j]=med_out[i]
                    #if abs(DDD[i,j])>DCmx: DDD[i,j]=np.sign(DDD[i,j])*DCmx
                    if abs(DDD[i,j])<hole*abs(med_out[i]): DDD[i,j]=med_out[i]        

    imsh_cuts_size=np.size(imsh_cuts)
    print 'imsh_cuts_size=',imsh_cuts_size
    inext=0
    for ii in range(0,iy,1):
        if inext<imsh_cuts_size:
            if (ii+1)==imsh_cuts[inext]:
                if i_R==i_R_start:
                    print 'i_R,ii=', i_R,ii, ' size of y=', np.size(y)
                    print '--->CUTS of f are done at y[ii]*180/pi=',y[ii]*180/pi
                # Here, in Python, indices of y[] start with 0
                inext=inext+1 # advance

    #stop
    #========= FILTERING OUT THE SMALL-SCALE NOISE in D
    #
    text_smooth = '_nosmooth' # part of filename to be saved
    #
    #---> Smoothening
    if ismooth==1:
    # This construct doesn't work in Python:
    #    DDD[icprp,icpar]=(DDD[ilprp,ilpar]+DDD[ilprp,icpar]+DDD[ilprp,irpar]
    #    +                 DDD[icprp,ilpar]+DDD[icprp,icpar]+DDD[icprp,irpar]
    #    +                 DDD[irprp,ilpar]+DDD[irprp,icpar]+DDD[irprp,irpar])/9.
    # But this works (not convenient)
        DDD[1:nrow-2,:]= (1./3.)* \
            (DDD[0:nrow-3,:]
            +DDD[1:nrow-2,:]
            +DDD[2:nrow-1,:])

#        DDD[1:nrow-2,1:nclm-2]= (1./9.)* \
#            (DDD[0:nrow-3,0:nclm-3]+DDD[0:nrow-3,1:nclm-2]+DDD[0:nrow-3,2:nclm-1]
#            +DDD[1:nrow-2,0:nclm-3]+DDD[1:nrow-2,1:nclm-2]+DDD[1:nrow-2,2:nclm-1]
#            +DDD[2:nrow-1,0:nclm-3]+DDD[2:nrow-1,1:nclm-2]+DDD[2:nrow-1,2:nclm-1])
#        # top edge of the u-grid:
#        DDD[nrow-1,1:nclm-2]= (1./6.)* \
#            (DDD[nrow-2,0:nclm-3]+DDD[nrow-2,1:nclm-2]+DDD[nrow-2,2:nclm-1]
#            +DDD[nrow-1,0:nclm-3]+DDD[nrow-1,1:nclm-2]+DDD[nrow-1,2:nclm-1])
        text_smooth = '_smooth'
    #---< Smoothening done


    #
    fig1= plt.figure() #---------------- FIGURE of Diff.Coeff. or Distr.func
    # Text for title of the plot:
    #txt=r"$|u_{||}| \tau_b u^2$"+str(DC)+\
    #'  ['+str(DCunits)+' cgs]  for '+ r"$\rho($"+str(i_R)+r"$)=$%1.5f" %(rho_a)

# Text for title of the plot:
    if Coeff=='f' or Coeff=='f_local':
        #txt=str(DC)+\
        #'   $for$  '+ r"$\rho($"+str(i_R)+r"$)=$%1.3f" %(rho_a)
        txt=str(DC)+\
        '   $for$  '+ r"$\rho$"+r"$=$%1.3f" %(rho_a)

    else: # Diff Coeffs
        txt=r"$|u_{||}| \tau_b u^2$"+str(DC)+\
            '  ['+str(DCunits)+' cgs]  for '+ \
            r"$\rho($"+str(i_R)+r"$)=$%1.4f" %(rho_a)
            
    if Coeff=='urfb':
        txt=r"$sqrt[|u_{||}| \tau_b u^2$"+str(DC)+\
            '$]$   for '+ \
            r"$\rho($"+str(i_R)+r"$)=$%1.4f" %(rho_a)

    # u-grid; units u/c

    if data_src=='DC_text':
        X,Y = np.meshgrid(upar*imx*unorm/clight, uprp*imx*unorm/clight)
    elif data_src=='cql3d_nc':
        #In matlab:
        #mesh( x(1:nv)*cos(pch)*vnorm/clight, x(1:nv)*sin(pch)*vnorm/clight, flog);
        #where  pch is an array pch(1:npitch).
        #The multiplication of x(1:nv)*cos(pch) creates a matrix nv*npitch
        cosy=np.asmatrix(cos(y))  # make a matrix (1,iy) {not same as vector!!!}
        cosy=cosy.transpose()     # transpose to (iy,1) shape
        siny=np.asmatrix(sin(y))  
        siny=siny.transpose()    
        xx=np.asmatrix(x)         # make a matrix (1,jx)
        #print cosy.shape,xx.shape
        X=np.dot(cosy,xx)*(unorm/clight)  # (iy,jx) matrix
        Y=np.dot(siny,xx)*(unorm/clight)
        #print X.shape
                        
#For 3D plots, check http://matplotlib.sourceforge.net/mpl_toolkits/mplot3d/tutorial.html

    #import pdb; pdb.set_trace()    # To turn on debugging
    if plot_type=='m':  #-> MESH plot
        #Check help(Axes3D), includes plot_surface:
        if Coeff=='f' or Coeff=='f_local':
            ax  = Axes3D(fig1,azim=+65,elev=35) # angles in degrees
        else:
            #ax  = Axes3D(fig1,azim=-90,elev=35) # angles in degrees
            ax  = Axes3D(fig1,azim=-80,elev=25)


        #X = np.arange(-ucmx, ucmx, ucmx/500)
        #Y = np.arange(0, ucmx, ucmx/200)
        #X, Y = np.meshgrid(X, Y)
        #R = np.sqrt(X**2 + Y**2)
        #DDD = np.sin(R)
        print shape(X), shape(Y), shape(DDD)
        
        #ax.plot_surface(X,Y,DDD/DCunits,rstride=10,cstride=50,cmap=cm.jet)
        ax.plot_surface(X,Y,X)


        if DCmn==0. and DCmx==0.: 
            ax.get_zlim3d()
            DCmin = np.min(DDD/DCunits)
            DCmax = np.max(DDD/DCunits)
            #print amax(X), amax(Y), ucmx, shape(DDD), DCmin, DCmax
        else:
            DCmin = DCmn
            DCmax = DCmx
            ax.set_zlim3d(DCmin/DCunits, DCmax/DCunits)
            #print amax(X), amax(Y), ucmx, shape(DDD), DCmin, DCmax
            
        zdir = (None) # direction for plotting text (title)
        xdir = (None) # direction
        ydir = (None) # direction
        #ax.set_xlabel(r"$u_{||}/c$", xdir) 
        #ax.set_ylabel(r"$u_{\perp}/c$", ydir) 
        #ax.set_zlabel(str(D), zdir)
        text(-ucmx*1.1, ucmx*1.1, DCmax/DCunits, txt, zdir)
        text(0.0,-ucmx*.37, DCmin/DCunits,r"$u_{||}/c$", xdir) #upar label
        text(-ucmx*1.6,ucmx*.5, DCmin/DCunits,r"$u_{\perp}/c$",ydir) #uprp
        #title(txt, fontsize=fnt+2)   # Does not work. WHY???
        plt.show()

        
    if plot_type=='c':  #-> CONTOUR plot 
        ax = plt.subplot(1, 1, 1)
        #ax.set_aspect(1.0)
        if DCmn==0. and DCmx==0.: 
            DCmin = np.min(DDD)
            DCmax = np.max(DDD)
        else:
            DCmin = DCmn
            DCmax = DCmx
        if Coeff=='f' or Coeff=='f_local':
            DCmin=6.9 #8 #6 #6 #4 #-4 #-50 #5  # log10(f) lowest level
            DCmax=16.5 #18 #16 #15 #17 # log10(f) highest level
        if DCmin==DCmax:
            DCmin=0
            DCmax=1
        levels=np.arange(DCmin/DCunits,DCmax/DCunits,(DCmax-DCmin)/DCunits/(Ncont-1))
        #print levels
        CS=plt.contour(X,Y,DDD/DCunits,levels,linewidths=linw,cmap=plt.cm.jet)
        ##CS=plt.contour(X,Y,DDD/DCunits,levels,linewidths=linw,cmap=plt.cm.brg)
        #CB=plt.colorbar(orientation='vertical', shrink=0.5) # colorbar
        l,b,w,h = plt.gca().get_position().bounds
        if Coeff=='f' or Coeff=='f_local' :            
            CB=plt.colorbar(orientation='vertical',shrink=0.46,\
            #ticks=[-2,0,2,4,6,8,10,12,14,16,18,20],format='%1.0f') #colorbar) #YuP updt
            ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],format='%1.0f') 
            ll,bb,ww,hh = CB.ax.get_position().bounds
            #CB.ax.set_position([l+w*0.04, bb, ww, hh]) # Left edge + a bit   
            CB.ax.set_position([l+w*0.7, bb, ww, hh]) # Near right edge   
            CB.lines.set_linewidth(5)            
        else:            
            CB=plt.colorbar(orientation='vertical',shrink=0.4,format='%0.1e') #YuP updt
            ll,bb,ww,hh = CB.ax.get_position().bounds
            CB.ax.set_position([l+w*0.85, bb, ww, hh]) 
            CB.lines.set_linewidth(5)
        #--> trapped-pass bndry:
        if fow_bndry==0:  # ZOW or Hybrid-FOW-solution
            plt.plot(vsign*x*cos(pchl)*ucmx,x*sin(pchl)*ucmx,'m--',linewidth=linw*2)
            plt.plot(vsign*x*cos(pchu)*ucmx,x*sin(pchu)*ucmx,'m--',linewidth=linw*2)
            #print 'ZOW bndry'
        else:     # FOW bndry
            # But add ZOW as well, for reference:
            plt.plot(vsign*x*cos(pchl)*ucmx,x*sin(pchl)*ucmx,'g--',linewidth=linw*2)
            plt.plot(vsign*x*cos(pchu)*ucmx,x*sin(pchu)*ucmx,'g--',linewidth=linw*2)
            # And now - FOW 
            for ib in range(1,8,1):  # 4 t-p bndries + 2 loss-cone bndries + stagnation
                npp=ns_bndry[lr,ib-1]
               # print ib
               # print ns_bndry
                #print 'ib, boundary length:', ib,npp
                if npp>0:
                    vc_bndry=    v_bndry[lr,0:npp,ib-1]/clight
                    th_bndry=theta_bndry[lr,0:npp,ib-1]                  
                    col='k'; linw2=linw*1.5
                    if ib==7: 
                        col='m';linw2=2.5*linw  # stagnation line
                        #[June2015 Tempor.corr. for stagn.line]
                        #th_bndry=3.14-theta_bndry[lr,0:npp,ib-1]
                    if ib==2: col='r' # t-p bndry #2, uppermost pitch angle(itu)
                    if ib==3: col='r' # t-p bndry #3, lower pitch angle (itl)
                    if ib==1: 
                        col='b' # t-p bndry #1, near stagnation region
                        for iii in range(0,npp):
                            #print th_bndry
                            if th_bndry[iii]>3.14: th_bndry[iii]=1.5708 #pi/2
                    if ib==4: 
                        col='b' # t-p bndry #4, near stagnation region
                        for iii in range(0,npp):
                            #print th_bndry
                            if th_bndry[iii]>3.14: th_bndry[iii]=1.5708 #pi/2
#                    if ib==6:  # arc-type bndry on counter-passing side
#                        plt.plot(vsign*vc_bndry[5:npp]*cos(th_bndry[5:npp]),\
#                                    vc_bndry[5:npp]*sin(th_bndry[5:npp]),\
#                                     '-',color=col,linewidth=linw2)
                    if ib!=6:
                        plt.plot(vsign*vc_bndry[2:npp]*cos(th_bndry[2:npp]),\
                                   vc_bndry[2:npp]*sin(th_bndry[2:npp]),\
                                    '-',color=col,linewidth=linw2)
                
        # upper boundary line:
        plot(vsign*ucmx*cos(y), ucmx*sin(y), color='k')

        #ax.axis([-ucmx,ucmx,0.0,ucmx])  # YuP: doesn't work for me
        ax.set_aspect(1.0)
        plt.axis([-ucmx,ucmx,0.,ucmx]) # But this works
        if ucmx<0.015:
            plt.xticks([-0.010, -0.005, 0.0, 0.005, 0.010])   
            plt.yticks([0.0, 0.005, 0.010])
        elif ucmx<=0.04:
            plt.xticks([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])   
            plt.yticks([0.0, 0.01, 0.02, 0.03])
        elif ucmx<=0.07:
            plt.xticks([-0.06, -0.03, 0.0, 0.03, 0.06])   
            plt.yticks([0.0, 0.03, 0.06])
        elif ucmx<=0.15:
            plt.xticks([ -0.08, -0.04, 0.0, 0.04, 0.08])   
            plt.yticks([0.0, 0.04, 0.08])
        elif ucmx<=12: #possible ECH case: u/c can be >1 (relativis p/m)
            plt.xticks([-10,-8,-6,-4,-2,0,2,4,6,8,10]) # 1000keV for electrons: u/c=2.8
            plt.yticks([0,2,4,6,8,10])
        plt.minorticks_on() # To add minor ticks
        plt.tick_params(which='both',  width=1)
        plt.tick_params(which='major', length=7)
        plt.tick_params(which='minor', length=4, color='k')
        plt.grid(True) # grid lines
        #plt.axis('equal')
        xlabel(r"$u_{||}/c$",fontsize=fnt+6)
        #ylabel(r"$u_{\perp}/c$",fontsize=fnt)
        title(txt,fontsize=fnt+4,y=1.03)   # Here works just fine
        #plt.text(-0.025, 0.085, '$log_{10}(f)$',fontsize=fnt+2)
        #plt.text(0.08, 0.085, '(b)',fontsize=fnt+2)
    # Save as png plot:
    savefig(str(D)+'_'+data_src+'_r'+str(i_R_index)+text_type+text_trim+text_smooth+'.png')
    # Save as eps plot:
    if isave_eps==1: 
        savefig(str(D)+'_'+data_src+'_r'+str(i_R_index)+text_type+text_trim+text_smooth+'.eps')
    #-------------------- FIGURE IS SAVED -------------------------
    #show()    # comment it, to avoid pressing "Enter" after each plot
    print 'MAX/MIN of plotted function: ',np.max(DDD),np.min(DDD)
    #
    #
    
    ksp=1
    k=ksp-1
    
    if iplot_fcuts==1:
        fig1= plt.figure() #---------------- Cuts of Distr.func at few pitch angles
        ax = plt.subplot(1, 1, 1)
        plt.hold(True)
        plt.grid(True)
        plt.xlim((0.,ucmx))   
        inext=0
        for ii in range(0,iy,1):
            if inext<imsh_cuts_size:
                if (ii+1)==imsh_cuts[inext]:
                    if Coeff=='f':
                        if ngen>1: # more than one gen.species
                            fcut=s_file_cql3d.variables['f'][k,lr,:,ii]
                        else:  # one gen.species
                            fcut=s_file_cql3d.variables['f'][lr,:,ii]
                        for j in range(jx):                       
                            if fcut[j] <= 0:  fcut[j]=1.0e-100
                    if remainder(inext,4)==0: col='b'
                    if remainder(inext,4)==1: col='m'
                    if remainder(inext,4)==2: col='r'
                    if remainder(inext,4)==3: col='g'
                    #print 'ii, y[ii]*180/pi, color=',ii,y[ii]*180/pi,col                    
                    #print 'ii, MAX/MIN of fcut: ', ii,np.max(fcut),np.min(fcut)
                    #print ucmx, np.max(xx), np.size(xx), np.size(fcut)
                    plt.yscale('log') # log 10 scale !!! Comment it, if you want a linear scale
                    plt.plot(x[:]*(unorm/clight),fcut[:],color=col,linewidth=linw)
                    #plt.plot(x[:]*(unorm/clight),fcut[:],'.',color=col,) # j grid points
                    #plt.ylim((1e7,1e20))  # for log10 scale
                    if fcut_mx==0:
                        fcut_mx= np.max(fcut)
                    if fcut_mn==0:
                        fcut_mn= fcut_mx/1e15
                    plt.ylim((fcut_mn,fcut_mx))  # for log10 scale                
                    inext=inext+1 # advance
        plt.minorticks_on() # To add minor ticks
        plt.tick_params(which='both',  width=1)
        plt.tick_params(which='major', length=7)
        plt.tick_params(which='minor', length=4, color='k')            
        xlabel(r"$u/c$",fontsize=fnt+6)        
        # Save as png plot:
        savefig('CUTS_'+str(D)+'_'+data_src+'_r'+str(i_R_index)+'.png')
        # Save as eps plot:
        if isave_eps==1: 
            savefig('CUTS_'+str(D)+'_'+data_src+'_r'+str(i_R_index)+'.eps')
        #-------------------- FIGURE IS SAVED -------------------------
        #show()    # comment it, to avoid pressing "Enter" after each plot


    Enrgy_MeV=  Enrgy_j[:,k]/1e3 # Horiz axis, to MeV
    currv_ka=    curr_v[:,lr]/1e3  # Vert.axis, A/cm2 to kA/cm2
    energy_mev=energy_v[:,lr]/1e3  # Vert.axis, keV/cm3 to MeV/cm3

    if iplot_currv==1:
        fig1= plt.figure() 
        ax = plt.subplot(2, 1, 2)
        # curr_v, such that np.dot(dx,curr_v[:,ir]) gives A/cm^2
        plt.hold(True)
        plt.grid(True)
        plt.xlim((0.,ucmx))   
        plt.plot(x*unorm/clight,currv_ka,linewidth=linw*2)
        xlabel(r"$u/c$",fontsize=fnt+6)    
        ylabel(r"$j(u)$ $(kA/cm^2)$",fontsize=fnt+6)
        ax = plt.subplot(2, 1, 1)
        # energy_v, such that np.dot(dx,energy_v[:,ir]) gives keV/cm^3
        # and np.dot(dx,energy_v[:,ir])/reden  gives [keV] same as energym[]
        plt.hold(True)
        plt.grid(True)
        plt.yscale('log') # log 10 scale !!! Comment it, if you want a linear scale
        ylim_max= np.max(energy_mev)*2
        ylim_min= ylim_max/1e7
        plt.ylim((ylim_min,ylim_max))
        plt.xlim((0.,ucmx))   
        plt.plot(x*unorm/clight,energy_mev,linewidth=linw*2)
        #xlabel(r"$u/c$",fontsize=fnt+6)    
        ylabel(r"$E(u)$ $(MeV/cm^3)$",fontsize=fnt+6)
        # Save as png plot:
        savefig('curr_v_p'+str(D)+'_'+data_src+'_r'+str(i_R_index)+'.png')
        # Save as eps plot:
        if isave_eps==1: 
            savefig('curr_v_p'+str(D)+'_'+data_src+'_r'+str(i_R_index)+'.eps')
        #-------------------- FIGURE IS SAVED -------------------------
        #show()    # comment it, to avoid pressing "Enter" after each plot

        fig1= plt.figure() 
        xlim_max= np.max(Enrgy_MeV) #  imx**2 # MeV
        xlim_min= 1e-4 #Enrgy_MeV[43] # MeV 
        # curr_v, such that np.dot(dx,curr_v[:,ir]) gives A/cm^2
        ax = plt.subplot(2, 1, 2)
        plt.hold(True)
        plt.grid(True)
        plt.xscale('log') # log 10 scale !!! Comment it, if you want a linear scale
        #plt.yscale('log') # log 10 scale !!! Comment it, if you want a linear scale
        ylim_max= np.max(currv_ka)*1.05
        ylim_min= np.min(currv_ka) #ylim_max/1e7
        plt.xlim((xlim_min, xlim_max))   
        plt.ylim((ylim_min, ylim_max))
        plt.plot(Enrgy_MeV,currv_ka,linewidth=linw*2)
        xlabel(r"$E$ $(MeV)$",fontsize=fnt+6)    
        ylabel(r"$j(u)$ $(kA/cm^2)$",fontsize=fnt+6)
        ax = plt.subplot(2, 1, 1)
        # energy_v, such that np.dot(dx,energy_v[:,ir]) gives keV/cm^3
        # and np.dot(dx,energy_v[:,ir])/reden  gives [keV] same as energym[]
        plt.hold(True)
        plt.grid(True)
        plt.xscale('log') # log 10 scale !!! Comment it, if you want a linear scale
        plt.yscale('log') # log 10 scale !!! Comment it, if you want a linear scale
        ylim_max= np.max(energy_mev)*2
        ylim_min= ylim_max/1e7
        plt.xlim((xlim_min, xlim_max))   
        plt.ylim((ylim_min, ylim_max))
        plt.plot(Enrgy_MeV,energy_mev,linewidth=linw*2)
        #xlabel(r"$E$ $(MeV)$",fontsize=fnt+6)    
        ylabel(r"$E(u)$ $(MeV/cm^3)$",fontsize=fnt+6)
        # Save as png plot:
        savefig('curr_v_E'+str(D)+'_'+data_src+'_r'+str(i_R_index)+'.png')
        # Save as eps plot:
        if isave_eps==1: 
            savefig('curr_v_E'+str(D)+'_'+data_src+'_r'+str(i_R_index)+'.eps')
        #-------------------- FIGURE IS SAVED -------------------------
        #show()    # comment it, to avoid pressing "Enter" after each plot


#    elapsed_time = time.time() - e0
#    cpu_time = time.clock() - c0
#    print 'elapsed and cpu time since start (sec.) =', elapsed_time, cpu_time
print '------------------------------'
#===============================================================================
# END of Loop in flux surface index i_R
print 'unorm [cm/s] =' , unorm
print file_cql3d

#gzip all du0u0 files in ../
#os.popen('gzip ../du0u0*')
