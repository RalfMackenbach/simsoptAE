#!/usr/bin/env python
import sys
sys.path.append('/Users/ralfmackenbach/Documents/GitHub/simsoptAE/target_functions/')  
import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve
from simsopt import make_optimizable
from sympy import *
from rm_targets import *
from mpi4py import MPI

mpi = MpiPartition()
mpi.write()
vmec = Vmec(filename="input.tok",mpi=mpi,verbose=False)
surf = vmec.boundary



###################################### AE VOLUME ######################################
# set setting for AE volume calculation
n_alpha     = 1     # tokamak so this is sufficient
s_last      = 1.0
n_int       = 10 
t_int       = 'quad'
n_turns     = 4
t_turns     = 'pol'
omnigenous  = True  # read as 'is the device omnigenous? - tokamak, so True'
lam_res     = 1000  
delta_theta = 1e-12
gpts        = 128
del_sing    = 0.0
log_scale   = False
# finally, make profiles
s = Symbol('s')
dens  = (1.0-1.0*s)**1.0
temp  = (1.0-1.0*s)**4.0
omn  = -dens.diff(s)/dens
omt  = -temp.diff(s)/temp
n_f     = lambdify(s,   dens,   'numpy')
T_f     = lambdify(s,   temp,   'numpy')
omn_f   = lambdify(s,   omn,    'numpy')
omt_f   = lambdify(s,   omt,    'numpy')


# calculate AE across volume
ae_vol =ae_volume_target(vmec,n_alpha,s_last,n_int,t_int,n_turns,t_turns,n_f,T_f,omn_f,omt_f,omnigenous,lam_res,delta_theta,gpts,del_sing,log_scale)
print('Total AE of device is', ae_vol)
###################################### AE VOLUME ######################################




###################################### AE SURFACE ######################################
# set settings for AE surface calculation
n_alpha     = 1     # tokamak so this is sufficient
s_val       = 0.9
n_turns     = 4
t_turns     = 'pol'
omnigenous  = True  # read as 'is the device omnigenous? - tokamak, so True'
lam_res     = 1000  
delta_theta = 1e-12
gpts        = 128
del_sing    = 0.0
log_scale   = False
omn         = 4.0
omt         = 0.0

# calculate AE of surface
ae_surf = ae_surface_target(vmec,n_alpha,s_val,n_turns,t_turns,omn,omt,omnigenous,lam_res,delta_theta,gpts,del_sing,log_scale)
print('Total AE of surface is', ae_surf)
###################################### AE SURFACE ######################################