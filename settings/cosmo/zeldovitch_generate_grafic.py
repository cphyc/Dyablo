import sys
import os
import numpy as np
from scipy import integrate

def writeInGraficFormat(data, filename, nx, ny, nz, dx, x0, y0, z0, astart, om, ov, H0) :
    f=open(filename,'wb')
    f.write(np.int32([44, nx, ny, nz]).tobytes()) # byte size of all header elements
    f.write(np.float32([dx, x0, y0, z0, astart, om, ov, H0]).tobytes())
    f.write(np.int32([44]).tobytes())

    i=0
    while(i<nz):
        x = data[i,:,:].flatten() # fill first dimension for velz (not sure why)
        f.write(np.int32([len(x)*4]).tobytes()) # here we store the size of n flotas
        f.write(np.float32(x).tobytes())
        f.write(np.int32([len(x)*4]).tobytes())
        i=i+1
    f.close()

def ddplus(a):
    global om
    global ov
    if (a==0.0):
        return 0
        
    eta = np.sqrt(om/a+ov*a*a+1.0-om-ov)
    return 2.5/(eta*eta*eta)

#=======================================================

def dplus(a,omegam,omegav):
    eta = np.sqrt(omegam/a+omegav*a*a+1.0-omegam-omegav)
    return eta/a*integrate.romberg(ddplus, 0., a)

#=======================================================

def dladt(a,omegam,omegav):
    eta = np.sqrt(omegam/a+omegav*a*a+1.0-omegam-omegav)
    return a*eta

#=======================================================

def fomega(a,omegam,omegav) :                                                    
    if (omegam==1.0 and omegav==0.0):
        return 1.0
    
    omegak = 1.0-omegam-omegav
    eta = np.sqrt(omegam/a+omegav*a*a+omegak)
    return (2.5/dplus(a,omegam,omegav)-1.5*omegam/a-omegak)/(eta*eta)

# Cosmology
H0 = 70.0 #km/s/Mpc
omegab = 0.049
omegam = 0.3
omegav = 1.0 - omegam
om = omegam
ov = omegav
rhobar = omegab/omegam

# Pancake physics
zstart = 50 # redshift start
zcross = 5.0 # redshift of the collapse
LboxMpc = 10.0 # box size in Mpc (and not in Mpc/h)
across = 1.0/(1.0 + zcross)
astart = 1.0/(1.0 + zstart)

# Box parameters
n1 = 32 # number of cells
dx = LboxMpc/n1
k = 2*np.pi/n1
nx=ny=nz=n1

A = 1./(dplus(across,omegam,omegav)*k)
Dp = dplus(astart,omegam,omegav)
vfact = fomega(astart,omegam,omegav)*H0*dladt(astart,omegam,omegav)/astart

q = np.linspace(0,n1,16384)
x = q+dplus(astart,omegam,omegav)*A*np.sin(k*q)

xg0 = np.linspace(0,n1,n1+1)
xgrid0 = (xg0[1:]+xg0[:-1])*0.5
xgrid = xgrid0#np.roll(xgrid0,10)
zzg,yyg,xxg = np.meshgrid(xgrid,xgrid,xgrid,indexing='ij')
qxgrid = np.interp(xxg,x,q)
rho = 1/(1+A*Dp*k*np.cos(k*qxgrid)) - 1.0

vel = A*Dp*np.sin(k*qxgrid)*dx*vfact #velocity km/s
velnull = vel*0.

dir = "./zeldo_5/"
os.makedirs( dir, exist_ok=True )
writeInGraficFormat(rho,     dir + 'ic_deltab', nx,ny,nz,dx, 0.,0.,0., astart, om, ov, H0)
writeInGraficFormat(vel,     dir + 'ic_velbx',  nx,ny,nz,dx, 0.,0.,0., astart, om, ov, H0)
writeInGraficFormat(velnull, dir + 'ic_velby',  nx,ny,nz,dx, 0.,0.,0., astart, om, ov, H0)
writeInGraficFormat(velnull, dir + 'ic_velbz',  nx,ny,nz,dx, 0.,0.,0., astart, om, ov, H0)