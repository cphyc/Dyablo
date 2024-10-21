import pyablo
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import newton



xmf_filename = sys.argv[1] #"test_sod_2D_main.xmf"
target_precision = float(sys.argv[2])
png_filename = sys.argv[3]

print("Validate Sod")
print(f'XMF filename : {xmf_filename}')
print(f'Target Precision : {target_precision}')
print(f'PNG output filename : {png_filename}')

class sod_analytical:
  def __init__(self, rL, uL, pL, rR, uR, pR, xi, T, gg ):
    # rL, uL, pL, rR, uR, pR : Initial conditions of the Reimann problem 
    # xi: POSITION where the interface sits.
    # T: the desired solution time
    # gg: adiabatic constant 1.4=7/5 for a 3D diatomic gas    
    self.rL = rL
    self.uL = uL
    self.pL = pL
    self.uR = uR
    self.pR = pR
    self.xi = xi
    self.T = T
    # compute speed of sound
    cL = np.sqrt(gg*pL/rL); 
    cR = np.sqrt(gg*pR/rR);
    # compute P
    # Function to find the roots of!
    def f(P, pL, pR, cL, cR, gg):
      a = (gg-1)*(cR/cL)*(P-1) 
      b = np.sqrt( 2*gg*(2*gg + (gg+1)*(P-1) ) )
      return P - pL/pR*( 1 - a/b )**(2.*gg/(gg-1.))
    self.P = newton(f, 0.5, args=(pL, pR, cL, cR, gg), tol=1e-12);

  def value( self, x ):  
    rL = self.rL
    uL = self.uL
    pL = self.pL
    uR = self.uR
    pR = self.pR
    xi = self.xi
    T  = self.T     
    P  = self.P     
    
    # compute speed of sound
    cL = np.sqrt(gg*pL/rL); 
    cR = np.sqrt(gg*pR/rR);

    # compute region positions right to left
    # region R
    c_shock = uR + cR*np.sqrt( (gg-1+P*(gg+1)) / (2*gg) )
    x_shock = xi + T * c_shock
    if( x >= x_shock ):
        return (rR, uR, pR, 5)

    # region 2
    alpha = (gg+1)/(gg-1)
    c_contact = uL + 2*cL/(gg-1)*( 1-(P*pR/pL)**((gg-1.)/2/gg) )
    x_contact = xi + T * c_contact

    if( x >= x_contact ):
        r = (1 + alpha*P)/(alpha+P)*rR
        u = c_contact
        p = P * pR
        return ( r, u, p, 4 )

    # region 3
    r3 = rL*(P*pR/pL)**(1/gg);
    p3 = P*pR;
    c_fanright = c_contact - np.sqrt(gg*p3/r3)
    x_fanright = xi + c_fanright*T

    if( x >= x_fanright ):
        r = r3
        u = c_contact
        p = P * pR
        return ( r, u, p, 3 )

    # region 4
    c_fanleft = -cL
    x_fanleft = xi + c_fanleft*T
    if( x >= x_fanleft ):
        u = 2 / (gg+1) * (cL + (x-xi)/T )
        r = rL*(1 - (gg-1)/2.*u/cL)**(2/(gg-1))
        p = pL*(1 - (gg-1)/2.*u/cL)**(2*gg/(gg-1))
        return ( r, u, p, 2 )

    return (rL,uL,pL,1)

# Physics
gg=1.66666667  # gamma = C_v / C_p = 7/5 for ideal gas -> cv/cp or cp/cv ?
rL, uL, pL =  1.0,  0.0, 1; 
rR, uR, pR = 0.125, 0.0, .1;
xi = 0.5
T = 0.25

# Open output file
reader = pyablo.XdmfReader()
series = reader.readTimeSeries(xmf_filename)
snap = reader.readSnapshot(series[-1])

NCells = snap.getNCells()
ids = np.arange( 0, NCells )

# Load fields
px = np.array(snap.getCellsCenter( ids ))[:,0]
rho = np.asarray( snap.readAllFloat('rho') )
rho_u = np.asarray( snap.readAllFloat('rho_vx') )
e_tot = np.asarray( snap.readAllFloat('e_tot') )

## Sorting is only needed for plot
sort_order = np.argsort( px )
px = px[sort_order]
rho = rho[sort_order]
rho_u = rho_u[sort_order]
e_tot = e_tot[sort_order]

u = rho_u/rho
Ek = 0.5 * rho_u**2.0 / rho
prs = (e_tot - Ek) * (gg-1.0)

formula = sod_analytical(rL, uL, pL, rR, uR, pR, xi, T, gg)

analytical = np.asarray([ formula.value( x ) for x in px ])

expected_rho = analytical[:,0]
expected_u = analytical[:,1]
expected_prs = analytical[:,2]

# Computing L1 norm
L1_rho = np.linalg.norm(rho - expected_rho, ord=1) / NCells
L1_u   = np.linalg.norm(u - expected_u, ord=1)   / NCells
L1_prs = np.linalg.norm(prs - expected_prs, ord=1) / NCells

print( f'Density; L1 error = {L1_rho:.4}' )
print( f'Velocity; L1 error = {L1_u:.4}' )
print( f'Pressure; L1 error = {L1_prs:.4}' )

fig, ax = plt.subplots(3, 1, figsize=(7, 7))
ax[0].plot(px, expected_rho, '--k')
ax[0].plot(px, rho, '-r')
ax[0].set_title(f'Density; L1 error = {L1_rho:.4}')
ax[0].set_xlabel('x')
ax[0].set_ylabel(r'$\rho$')
ax[1].plot(px, expected_u, '--k')
ax[1].plot(px, u, '-r')
ax[1].set_title(f'Velocity; L1 error = {L1_u:.4}')
ax[1].set_xlabel('x')
ax[1].set_ylabel(r'$v_x$')
ax[2].plot(px, expected_prs, '--k')
ax[2].plot(px, prs, '-r')
ax[2].set_title(f'Pressure; L1 error = {L1_prs:.4}')
ax[0].set_xlabel('x')
ax[0].set_ylabel(r'$p$')
plt.suptitle(f'SOD Shock tube')
plt.tight_layout()
plt.savefig(f'{png_filename}')
print( f'Exported {png_filename}' )

if( L1_rho > target_precision ):
  print( f'Precision target not met {L1_rho:.4} > {target_precision:.4}'  )
  exit(1)