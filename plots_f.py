#ipython
#YuP and BH, to May 5, 2011

#Mesh or contour plots of diffusion coeffs at each minor radius,
#  either in DC multi-radius text format, or cql3d netcdf rf coefficient
#  format.

# Also - f4d distributions.


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

file_cql3d='nstx_130608.00352.0_Sept2023_iexcit5_f4d.nc'
#--------------------------------------------------------
# See below:   i_Z=  #Do plots for one value of Z


Coeff = 'f'     # specify which Diff.Coeff to plot: cqlb,cqlc,cqle,cqlf
#               # OR distribution function 'f', in which case use .nc
#               # file, as opposed to _rf.nc cql3d file path

Emax = 'disabled' # Applicable for data_src='DC_text':
#                 # plots are on a upar,uperp grid, but with 'enabled'
#                 # coeffs above a  constant total maximum energy are
#                 # zero.  If a real number is entered here, it will
#                 # fraction of maximum energy on the grid.
#                 # Else, it is set =1.0 and enorm from cql3d
#                 # or max parallel or perp energy for DC data is used.

plot_type='c' #'c' -> contour plot, or 'm' -> mesh plot
fnt  = 14     # font size for axis numbers (see 'param=' below) 
linw = 1.0    # LineWidth for contour plots
Ncont= 100    # Number of contour levels
stride=2      
# For mesh plots: if stride=2, color is assigned to each 2x2 cell of u-grid
# and mesh is shown for each 2nd line; but data is plotted over total u-grid
#
i_R_start=1 # range of flux surfaces to plot [Beginning at 1]
i_R_stop =33 #33

i_Z=11  #Do plots for one value of Z
# Suggested: Half of nz_f4d

if i_Z<10:
    i_Z_index = '00'+str(i_Z)
elif i_Z<100:
    i_Z_index = '0'+str(i_Z) 
else:
    i_Z_index = str(i_Z)

DCmn= 0.    # Specify vertical limits for mesh plots of Diff.Coeffs.
#DCmx= 3.e41 # If DCmn=0. and DCmx=0., the limits will be set automatically
DCmx= 0. # If DCmn=0. and DCmx=0., the limits will be set automatically
#DCmn=1.
#DCmx=20.
DCunits=1. # Units (scale) for mesh plots of Diff.Coeff.

imx = 1.0 # 0.75    # limits for u_par/unorm and u_per/unorm 
#                   # grid to plot (max: imx=1)
clight= 2.99792458e10 # speed of light [cm/s]



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


#------------------------------------------------------ READ DATA -----

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


R=array(s_file_cql3d.variables['f4dr'])
print 'shape of R=f4dr :', R.shape
Z=array(s_file_cql3d.variables['f4dz'])
print 'shape of Z=f4dr :', Z.shape
print 'grid Z(cm)=', Z

x=array(s_file_cql3d.variables['f4dv'])
y=array(s_file_cql3d.variables['f4dt'])

f=array(s_file_cql3d.variables['f4d'])
print 'shape of f=f4d :', shape(f)

jx=s_file_cql3d.variables['nv_f4d'].getValue()
iy=s_file_cql3d.variables['nt_f4d'].getValue()
unorm=s_file_cql3d.variables['vnorm'].getValue()
unorm2=unorm**2
unorm3=unorm*unorm2
unorm4=unorm2*unorm2
ucmx  = imx*unorm/clight # limits for plots

txt='Title'
#if Coeff=='f':
D='f' # for name in file (saving plots)
DC=r"$log_{10}(f)$"  # for title in plots
#elapsed_time = time.time() - e0
#cpu_time = time.clock() - c0


#print 'elapsed and cpu time since start (sec.) =', elapsed_time, cpu_time
print 'STARTING LOOP in i_R (flux surfaces index)'
print '=========================================='

if plot_type == 'c':
    text_type='_contour'
elif plot_type == 'm':
    text_type='_mesh'

#i_R=i_R_start
#print 'i_R=',i_R,  '   R=',R[i_R-1]


#===========================================================================
# Loop in flux surface index i_R  starts here; 
# scanning i_R= i_R_start:i_R_stop
#
for i_R in range(i_R_start,i_R_stop+1,1):

    #----------------------------
    if i_R<10:
        i_R_index = '00'+str(i_R)
    elif i_R<100:
        i_R_index = '0'+str(i_R) 
    else:
        i_R_index = str(i_R)
    #----------------------------
    text_trim=''

    RR=R[i_R-1]
    ZZ=Z[i_Z-1]
    print 'i_R=',i_R, ' RR(cm)=',RR, ' i_Z=', i_Z,' ZZ(cm)=',ZZ

#if Coeff=='f':
#DDD=s_file_cql3d.variables['f4d'][:,:,i_Z,i_R]
    DDD=f[:,:,i_Z-1,i_R-1]   #Alternatively
#    DDD[iy-1,jx-1]=DDD[iy-2,jx-1]
    DDD=np.nan_to_num(np.log10(DDD))
    DDDmax=np.max(DDD)
    DDDmin=DDDmax-20.
    for j in range(jx):
        for i in range(iy):
            #        if DDD[j,i] < DDDmin:  DDD[j,i]=DDDmin
            if DDD[i,j] < DDDmin:  DDD[i,j]=DDDmin

#    DDD= transpose(DDD) # rows and columns in source array are reversed
#    Not sure why transpose is require [BH]
            

#Some printing, if required
#for i in range(iy):
#    DDD[i,:]
#Text for title of the plot:
    txt=str(DC)+\
            " $at$" +\
            " $R=$"+"%1.1f" %(RR) +'$cm$' +\
            " $Z=$"+"%1.1f" %(ZZ) +'$cm$'
    #print txt
    

    fig1= plt.figure()

    X=np.zeros([iy,jx])
    Y=np.zeros([iy,jx])
    for j in range(jx):
        for i in range(iy):
            X[i,j]=x[j]*cos(y[i])*imx*unorm/clight
            Y[i,j]=x[j]*sin(y[i])*imx*unorm/clight
                
    ax = plt.subplot(1, 1, 1)
    ax.set_aspect(1.0)
    if DCmn==0. and DCmx==0.: 
        DCmin = np.min(DDD)
        DCmax = np.max(DDD)
    else:
        DCmin = DCmn
        DCmax = DCmx
        
    print 'DCmin, DCmax=',DCmin,DCmax

    levels=np.arange(DCmin/DCunits,DCmax/DCunits,(DCmax-DCmin)/DCunits/(Ncont-1))
    CS=plt.contour(X,Y,DDD/DCunits,levels,linewidths=linw,cmap=plt.cm.jet)
    CB=plt.colorbar(orientation='vertical', shrink=0.5, format='%1.0f') #YuP updt
# This makes the original colorbar look a bit out of place,
# so let's improve its position.
    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([l+w*1.1, bb, ww, hh])        
    plt.grid(True) # grid lines
    plt.minorticks_on() # To add minor ticks
    plt.tick_params(which='both',  width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='k')
    ax.set_aspect(1.0)
    ax.axis([-ucmx,ucmx,0.0,ucmx])
#        plt.plot([0,X_tp_l],[0,Ymax],'r--')
#        plt.plot([0,-X_tp_l],[0,Ymax],'r--')
    xlabel(r"$u_{||}/c$") #, fontsize=fnt)
    ylabel(r"$u_{\perp}/c$") #, fontsize=fnt)
    title(txt,fontsize=fnt+2,y=1.05)   # Here works just fine
    
#    savefig(str(D)+'_'+data_src+'_r'+str(i_R_index)+text_type+text_trim+'.png')
    savefig('f4d_R'+str(i_R_index)+'_Z'+str(i_Z_index)+'.png')

#!gwenview Fig_33.16.png &
    #-------------------- FIGURE IS SAVED -------------------------
    #show()    
    print 'MAX value dep var is ',np.max(DDD)
    #elapsed_time = time.time() - e0
    #cpu_time = time.clock() - c0
    #print 'elapsed and cpu time since start (sec.) =', elapsed_time, cpu_time
    print '------------------------------------------------------------------'
#===============================================================================
# END of Loop in flux surface index i_R


