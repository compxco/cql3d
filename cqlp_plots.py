#ipython
#YuP 2017-2021

# Plots for data saved into mnemonic.nc files 

from numpy import *
from mpl_toolkits.mplot3d import Axes3D

from pylab import *
from matplotlib import rc 
from matplotlib.pyplot import cm,figure,axes,plot,xlabel,ylabel,title,savefig,show

import os
import math

import netCDF4
print 'netCDF4:', netCDF4.__version__

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



e0 = time.time()  # elapsed time since the epoch
c0 = time.clock() # total cpu time spent in the script so far

#-----------------------------------------------
# NetCDF issues: machine-dependent
# Try netcdf=4 to envoke netCDF4,
# Or try netcdf=2 to work with older netCDF.
netcdf=4
#-----------------------------------------------
if netcdf==4: from netCDF4 import Dataset 
#-----------------------------------------------

# Constants
pi=3.14159265358979
clight= 2.99792458e10   # speed of light [cm/s]
charge= 4.8032e-10      # e-charge [cgs]
e_cgs = 4.8032e-10      # e-charge [cgs]
e_mass= 9.1095e-28      #    [gramm]
e_si  = 1.6022e-19      # coulombs
p_mass= 1.67262158e-24  # proton mass    [gram]
proton= 1.67262158e-24  #    [gramm]
ergtkev=1.6022e-09


#-------------------------------------------------------------------------
file_name='cqlp_fixedMU_E000_noCollis_Dsrc.nc'  # 2022-02-09 tests with source

fnt  = 15 # font size for axis numbers (see 'param=' below) 
linw = 1.0  # LineWidth for contour plots
Ncont= 30   # Number of contour levels
#-------------------------------------------------------------------------


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

mpl.rcParams.update(params)
#rc.defaults() #to restore defaults

mpl.rcParams['font.size']=fnt+2  # set font size for text in mesh-plots


#------------ Open *.nc file
if netcdf==4: 
    s_file_name= Dataset(file_name, 'r', format='NETCDF4')


print 'The input file, ',file_name,', contains:'
print '========================================'
print "The global attributes: ",s_file_name.dimensions.keys()        
print "File contains variables: ",s_file_name.variables.keys()
print '========================================'


x=array(s_file_name.variables['x'])
xmin=np.min(x)
xmax=np.max(x)
print 'x:     shape/min/max', x.shape, xmin,xmax
jx= x[:].size  #  u/vnorm grid size
print 'jx=',jx

dx=array(s_file_name.variables['dx'])
dxmin=np.min(dx)
dxmax=np.max(dx)
cint2= dx*x*x
print 'cint2:     shape/min/max', cint2.shape

y=array(s_file_name.variables['y'])
ymin=np.min(y)
ymax=np.max(y)
print 'y:     shape/min/max', y.shape, ymin,ymax # 
iy= y[0,:].size  # pitch angle grid size
print 'iy=',iy

cynt2=array(s_file_name.variables['cynt2'])
print 'cynt2:     shape', cynt2.shape

z=array(s_file_name.variables['z'])
z=np.transpose(z)
zmin=np.min(z)
zmax=np.max(z)
print 'z:     shape/min/max', z.shape, zmin,zmax
kz= z[:].size #  Note that z[1:lz] in CQLP
lz= kz
print 'kz=',kz

iy_=array(s_file_name.variables['iy_'])
print 'iy_:     shape', iy_.shape
itl=array(s_file_name.variables['itl'])
itu=array(s_file_name.variables['itu'])
print 'itl:     shape', itl.shape 

time=array(s_file_name.variables['time'])
timecode=time
nstop=len(timecode)-1 # nstop in cqlinput
nt=nstop

ngen= s_file_name.variables['ngen'].getValue()
ngen= np.asscalar(ngen)
print 'ngen=',ngen, '  nstop=',nstop


nstride= np.floor(nt/200) # np.floor(nt/40) #Plot only 40 time steps (for clarity).
nstride= int(max(nstride,1))  # To make sure it is >0
nwstride= 1 #nstride*2 #larger stride for wire plots, for clarity
# If there are many time steps, and data for all time steps is plotted
# the 'wiremesh' plots look bad - as a solid-filled color.
# For clarity, it is recommended to plot data for only 20 steps.
# That is why we set the skipping factor, nstride.
print 'time step counter  nt=', nt
print 'timecode[1] -timecode[0]    [ms] =', timecode[1]-timecode[0]
print 'timecode[nt]-timecode[nt-1] [ms] =', timecode[nt]-timecode[nt-1]


#  f3d(iy,jx,0:lz-1) distribution function (at last time step)
f3d= array(s_file_name.variables['f'])  # 
f3d_min=np.min(f3d)
f3d_max=np.max(f3d)
print 'f3d:        shape/min/max', f3d.shape,  f3d_min,f3d_max

#Form Integral of f3d over u (norm-ed)
f_z_y=zeros((lz,iy))
wk=zeros((jx))
for k in range(0,lz,1): 
    for i in range(0,iy,1): 
        #print wk.shape, f3d[k,:,i].shape
        wk[:]=f3d[k,:,i]
        f_z_y[k,i]= np.dot(wk[:],cint2[:]) # sum in j index
print 'shape of f_z_y', np.shape(f_z_y) 




#-------- theta vs z  grid points for each i level 
#--- Also add plots of f_z_y(z,theta)= SUM(f3d() x^2 dx)
#    which is Integral of f3d over u (norm-ed)
#    f_z_y[0:kz-1,0:iy-1]
F= np.transpose(f_z_y) # [0:iy-1,0:kz-1]
fig0=plt.figure()
plot_name= 'theta_grid'
ax=plt.subplot(111)
#ax.set_aspect(1.0)    
plt.hold(True)
plt.minorticks_on() # To add minor ticks
plt.tick_params(which='both',  width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4, color='k')    
plt.title('$pitch-angle$ $grid$',fontsize=fnt+2,y=1.01)
plt.ylabel('$theta$ $(rad)$',fontsize=fnt+2)
plt.xlabel('$z$ $(cm)$',fontsize=fnt+2)
plt.grid(True)
plt.ylim((-0.05*pi, 1.05*pi))
zmax=np.max(z)
plt.xlim((-0.05*zmax, 1.05*zmax))
izturn= kz*ones((iy))
for iz in range(0,kz,1):
    for i in range(iy_[iz],iy,1):
        y[iz,i]=pi #extend non-physical points to pi (those are zeros by default)

for iz in range(0,kz/2,1):
    if (iy_[iz]/2)*2 != iy_[iz] :
        print 'WARN iy_[iz]=',iy_[iz]
    iyh= iy_[iz]/2 -1
    izturn[iyh]=iz
    izturn[iyh+1]=iz
    print 'iz,iy_[iz],iyh=',iz,iy_[iz],iyh
    
for i in range(0,iy,1):  
    if remainder(i,4)==0: col='k'
    if remainder(i,4)==1: col='b'
    if remainder(i,4)==2: col='r'
    if remainder(i,4)==3: col='g'    
    if remainder(i,4)==4: col='m' 
    if remainder(i,4)==5: col='c' 
    izz=izturn[i]
    #plt.plot(z[0:izz], y[0:izz,i],color=col,linewidth=linw)
    plt.plot(z[:], y[:,i],'.',color=col) # dots at grid nodes
#for iz in range(0,kz,1):
#    for i in range(0,iy_[iz],1):
#        if remainder(i,4)==0: col='k'
#        if remainder(i,4)==1: col='b'
#        if remainder(i,4)==2: col='r'
#        if remainder(i,4)==3: col='g'    
#        if remainder(i,4)==4: col='m' 
#        if remainder(i,4)==5: col='c' 
#        plot(z[iz], y[iz,i],'.',color=col) # dots at grid nodes
#Z,Y = np.meshgrid(y[0,:],z[1:kz+1])  # 2D grids [rad,cm]
Z=0*F
Y=0*F
print 'shape of Z, Y, F', np.shape(Z), np.shape(Y), np.shape(F)
for i in range(0,iy,1):
    for iz in range(0,kz,1):
        #print i,Z[i,0],Z[i,kz-1]
        Z[i,iz]= z[iz]
        Y[i,iz]= y[iz,i]
print 'shape of Z, Y, F', np.shape(Z), np.shape(Y), np.shape(F)
f_z_y_mn= np.min(F)
f_z_y_mx= np.max(F)
levels=np.arange(f_z_y_mn,f_z_y_mx,(f_z_y_mx-f_z_y_mn)/(Ncont-1))
CS=plt.contour(Z[0:iy,:],Y[0:iy,:],F[0:iy,:],levels,linewidths=linw*2,cmap=plt.cm.jet)
CB=plt.colorbar(orientation='vertical',shrink=1.0,format='%1.3e')  

savefig(plot_name+'.png')
plt.show() 
#stop
