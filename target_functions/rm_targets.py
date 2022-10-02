import sys  
sys.path.append('/Users/ralfmackenbach/Documents/GitHub/simsoptAE/AE_routines')  
import  numpy           as  np
import  AE_routines     as  ae
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import fsolve
from scipy.special import ellipe
from scipy.interpolate import InterpolatedUnivariateSpline
from simsopt.mhd.boozer import Boozer
import mpi4py
from scipy import integrate
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")






# returns the fraction of the total energy that is available ∫ (A/V) dV / ∫ p dV.
def ae_volume_target(vmec,n_alpha,s_last,n_int,t_int,n_turns,t_turns,n_f,T_f,omn_f,omt_f,omnigenous,lam_res,delta_theta,gridpoints,del_sing,logscale):
    """
    Given some array of s and phi_center, calculate the available energy of device approximated by nodes on these arrays.
    Please note that the routine will calculate the average AE over all phi's for a fixed s, and then integrates over s.
    As an example if; s = [0.4,0.5] and phi_centre=[0.0,pi], the routine will calculate 4 AEs: 2 at s=0.4 and 2 at s=0.5.
    vmec        -   the vmec object
    s_last      -   A/E will be estimated as the integral from s = [0,s_last]. s_last <= 1
    n_alpha     -   Number of alpha taking into account for the flux surface average.
    n_int       -   Number of points along s taking into account for the integration method
    t_int       -   Type of integration along s. Either 'trapz' or 'quad'.
    n_turns     -   Number of turns of the fieldline. Either poloidal or toroidal, depending on t_turns
    t_turns     -   Type of turns for n_turns. Either 'pol' or 'tor'
    n_f         -   Number density as a function of radius (e.g. lambda s: ( 1 - s ))
    T_f         -   Electron temperature as a function of radius (e.g. lambda s: (1 - s) )
    omn_f       -   Logarithmic density derivative function (i.e. d ln(n) / d s)
    omn_f       -   Logarithmic electron temperature derivative function (i.e. d ln(T) / d s)
    omnigenous  -   can be set True is one want to enforce that radial drifts are set zero in calculation of AE
    lam_res     -   number of lambda point in calculation of AE (trapz)
    delta_theta -   padding for periodic boundary condition. Set very small.
    gridpoints  -   number of theta_pest point per fieldline. Recommended ~128 per well.
    del_sing    -   padding around possible singular points. Tends to not change the results - keep zero.
    logscale    -   return ln(AE+1) instead of AE. Useful for a coldstart, where AEs can become extremely large.
    verbose     -   print intermediate steps.
    """
    # run vmec
    vmec.run()
    surf = vmec.boundary
    nfp = surf.nfp

    # make phi_arr 
    alpha_arr = np.linspace(0,2*np.pi,n_alpha,endpoint=False)

    # make dVds function
    s_half_grid_arr = vmec.s_half_grid
    dVds_arr = 4 * np.pi * np.pi * np.abs(vmec.wout.gmnc[0, 1:])
    dVds = InterpolatedUnivariateSpline(s_half_grid_arr, dVds_arr, ext='extrapolate')

    # create function returning integrand 
    def ae_integrand(vmec,s,alpha,n_turns,t_turns,n_f,T_f,omn_f,omt_f,dVds):
        if t_turns == 'pol':
            fieldline = vmec_fieldlines(vmec,s,alpha,theta1d=np.linspace(-n_turns*np.pi,n_turns*np.pi,gridpoints))
        if t_turns == 'tor':
            fieldline = vmec_fieldlines(vmec,s,alpha,phi1d=np.linspace(-n_turns*np.pi,n_turns*np.pi,gridpoints))
        iota        = (fieldline.iota).flatten()[0]
        rho_star_pol  = np.sqrt(T_f(s))/iota # assuming fixed B_ref, m, q, and L_ref
        p         = 1.5 * n_f(s) * T_f(s)
        omn_x1      = omn_f(s)*2*np.sqrt(s)
        omt_x1      = omt_f(s)*2*np.sqrt(s)
        ae_val      = p * rho_star_pol**2.0 * dVds(s) * ae_per_V_per_nT(fieldline,omn_x1,omt_x1,omnigenous,lam_res,delta_theta,del_sing,verbose=False)
        return ae_val
    
    # make array containing ae's for flux surface average.
    ae_alpha_arr = np.empty_like(alpha_arr)
    
    # per alpha, do the integral over s
    for idx, alpha_val in enumerate(alpha_arr):
        f = lambda x: ae_integrand(vmec,x,alpha_val,n_turns,t_turns,n_f,T_f,omn_f,omt_f,dVds)
        if t_int == 'quad':
            ae_total, ae_err = integrate.quadrature(f, 0.0, s_last, maxiter=n_int, vec_func=False)
        if t_int == 'trapz':
            s_arr = np.linspace(0.0,s_last,n_int+1,endpoint=False)[1::]
            f_arr = np.zeros_like(s_arr)
            for idx2, value in np.ndenumerate(s_arr):
                f_arr[idx2] = f(value)
            ae_total = integrate.trapezoid(f_arr, s_arr)
        ae_alpha_arr[idx]  = ae_total

    # average over alpha
    ae_tot = 2 * np.pi * np.average(ae_alpha_arr)
    
    # calculate total energy
    f = lambda x: 3/2 * n_f(x) * T_f(x) * dVds(x)
    if t_int == 'quad':
        E_tot, E_err = integrate.quadrature(f, 0.0, s_last, maxiter=n_int, vec_func=False)
    if t_int == 'trapz': 
        s_arr = np.linspace(0.0,s_last,n_int+1,endpoint=False)[1::]
        E_tot = integrate.trapezoid(f(s_arr), s_arr)
    
    # divide ae by total thermal energy
    ae_tot = ae_tot / E_tot

    # check if logscale is true
    if logscale == True:
        ae_tot = np.log(1+ae_tot)
        print("log(AE+1) = ", ae_tot)
    elif logscale == False:
        print("AE = ", ae_tot)

    return ae_tot



# returns the available energy on a surface.
def ae_surface_target(vmec,n_alpha,s_val,n_turns,t_turns,omn,omt,omnigenous,lam_res,delta_theta,gridpoints,del_sing,logscale):
    """
    Given some array of s and phi_center, calculate the available energy of device approximated by nodes on these arrays.
    Please note that the routine will calculate the average AE over all phi's for a fixed s, and then integrates over s.
    As an example if; s = [0.4,0.5] and phi_centre=[0.0,pi], the routine will calculate 4 AEs: 2 at s=0.4 and 2 at s=0.5.
    vmec        -   the vmec object
    n_alpha     -   Number of alpha taking into account for the flux surface average.
    s_val       -   Normalized flux at which the ae is calculated
    n_turns     -   Number of turns of the fieldline. Either poloidal or toroidal, depending on t_turns
    t_turns     -   Type of turns for n_turns. Either 'pol' or 'tor'
    omn         -   Logarithmic density derivative      (i.e. d ln(n) / d s)
    omt         -   Logarithmic temperature derivative  (i.e. d ln(T) / d s)
    omnigenous  -   can be set True is one want to enforce that radial drifts are set zero in calculation of AE
    lam_res     -   number of lambda point in calculation of AE (trapz)
    delta_theta -   padding for periodic boundary condition. Set very small.
    gridpoints  -   number of theta_pest point per fieldline. Recommended ~128 per well.
    del_sing    -   padding around possible singular points. Tends to not change the results - keep zero.
    logscale    -   return ln(AE+1) instead of AE. Useful for a coldstart, where AEs can become extremely large.
    verbose     -   print intermediate steps.
    """
    # run vmec
    vmec.run()
    surf = vmec.boundary
    nfp = surf.nfp

    # make alpha 
    alpha_arr = np.linspace(0,2*np.pi,n_alpha,endpoint=False)

    # create function returning integrand 
    def ae_integrand(vmec,s,alpha,n_turns,t_turns,omn,omt):
        if t_turns == 'pol':
            fieldline = vmec_fieldlines(vmec,s,alpha,theta1d=np.linspace(-n_turns*np.pi,n_turns*np.pi,gridpoints))
        if t_turns == 'tor':
            fieldline = vmec_fieldlines(vmec,s,alpha,phi1d=np.linspace(-n_turns*np.pi,n_turns*np.pi,gridpoints))
        iota        = (fieldline.iota).flatten()[0]
        ae_val      = 1.0/iota**2.0 * ae_per_V_per_nT(fieldline,omn,omt,omnigenous,lam_res,delta_theta,del_sing,verbose=False)
        return ae_val
    
    ae_alpha_arr = np.empty_like(alpha_arr)
    
    for idx, alpha_val in enumerate(alpha_arr):
        ae_alpha_arr[idx]  = ae_integrand(vmec,s_val,alpha_val,n_turns,t_turns,omn,omt)

    ae_tot = 2 * np.pi * np.average(ae_alpha_arr)

    print('AE @ s=', s_val, ' is ', ae_tot)

    return ae_tot



# function which returns AE per unit volume per n0*T0 of a fieldline
def ae_per_V_per_nT(fieldline,omn,omt,omnigenous,lam_res,delta_theta,del_sing,verbose):
    """
    Returns the available energy given a vmec fieldlines object. Specifically, it returns the available energy per unit volume, per 3/2nT.
    Assumes correlation length (lengthscale over which energy is available) of 1.0. To set to poloidal gyroradius, normalize by 1/iota
    fl          -   fieldline object
    omn         -   normalized density gradient (a/L_n)
    omt         -   normalized electron temperature gradient
    omnigenous  -   boolean. Set True to enforce no radial drifts in calculation of AE
    lam_res     -   number of lamba values to integrate over (using trapz)
    delta_theta -   periodic boundary condition "padding". Set to a very small value
    del_sing    -   at local maximum integrand can become singular, this removes local maxima from 
                    lamda arrays. Tends to have no effect - keep 0.
    verbose     -   prints the AE of the fieldlines.
    """
    # import data to be used in the calculation
    s_val       = (fieldline.s).flatten()[0]
    iota        = (fieldline.iota).flatten()[0]
    alpha_val   = (fieldline.alpha).flatten()[0]
    modB        = (fieldline.modB).flatten()
    jac         = (fieldline.sqrt_g_vmec).flatten()
    theta_v     = (fieldline.theta_vmec).flatten()
    theta_p     = (fieldline.theta_pest).flatten()
    grad_B_X    = (fieldline.grad_B_X).flatten()
    grad_B_Y    = (fieldline.grad_B_Y).flatten()
    grad_B_Z    = (fieldline.grad_B_Z).flatten()
    grad_psi_X  = (fieldline.grad_psi_X).flatten()
    grad_psi_Y  = (fieldline.grad_psi_Y).flatten()
    grad_psi_Z  = (fieldline.grad_psi_Z).flatten()
    grad_phi_X  = (fieldline.grad_phi_X).flatten()
    grad_phi_Y  = (fieldline.grad_phi_Y).flatten()
    grad_phi_Z  = (fieldline.grad_phi_Z).flatten()
    grad_alpha_X= (fieldline.grad_alpha_X).flatten()
    grad_alpha_Y= (fieldline.grad_alpha_Y).flatten()
    grad_alpha_Z= (fieldline.grad_alpha_Z).flatten()
    B_sup_phi   = (fieldline.B_sup_phi).flatten()
    d_B_d_s     = (fieldline.d_B_d_s).flatten()
    B_ref       = (fieldline.B_reference)
    L_ref       = (fieldline.L_reference)
    phi_edge    = (fieldline.edge_toroidal_flux_over_2pi)

    # we construct dBdpsi
    # Realize that gradB = dBdpsi * gradpsi + dBdalpha * gradalpha + dBdphi * gradphi
    # Take inner product of above expression with (gradalpha cross gradphi)
    # dBdpsi = gradB inner ( gradalpha cross grad phi ) / B_sup_phi
    gradphi         = np.stack([grad_phi_X,grad_phi_Y,grad_phi_Z],axis=1)
    gradpsi         = np.stack([grad_psi_X,grad_psi_Y,grad_psi_Z],axis=1)
    gradalpha       = np.stack([grad_alpha_X,grad_alpha_Y,grad_alpha_Z],axis=1)
    gradB           = np.stack([grad_B_X,grad_B_Y,grad_B_Z],axis=1)
    gradalphaXgradphi       = np.cross(gradalpha, gradphi)
    gradBingradphiXgradpsi  = np.einsum('ij, ij->i',gradalphaXgradphi,gradB)
    jac             = 1/B_sup_phi
    d_B_d_psi       = gradBingradphiXgradpsi / B_sup_phi

    # now we construct dBdalpha
    # Realize that gradB = dBdpsi * gradpsi + dBdalpha * gradalpha + dBdphi * gradphi
    # Take inner product of above expression with (gradphi cross gradpsi)
    # dBdalpha = gradB inner ( gradphi cross grad psi ) / B_sup_phi
    gradphiXgradpsi = np.cross(gradphi, gradpsi)
    gradBingradphiXgradpsi  = np.einsum('ij, ij->i',gradphiXgradpsi,gradB)
    d_B_d_alpha     = gradBingradphiXgradpsi / B_sup_phi

    # now convert to dimless quantities as in GIST
    b_norm      = modB/B_ref
    dbdx1       = d_B_d_psi * 2 * np.sqrt(s_val) / B_ref * phi_edge
    dbdx2       = d_B_d_alpha / ( np.sqrt(s_val) * B_ref )
    sqrt_g      = np.abs(jac/(L_ref**3 * iota / 2)*phi_edge)

    # fig, axs = plt.subplots(2,2)
    # axs[0,0].plot(theta_p,b_norm)
    # axs[0,1].plot(theta_p, dbdx1)
    # axs[1,0].plot(theta_p, dbdx2)
    # axs[1,1].plot(theta_p,b_norm*sqrt_g)
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()


    # set scalars
    dlnTdx = -omt
    dlnndx = -omn
    # iota should be put as prefactor!
    Delta_x1= 1.0
    Delta_x2= 1.0
    L_tot  = np.abs(np.trapz(b_norm*sqrt_g/iota,theta_p))
    rec_B_ave = np.abs(np.trapz(sqrt_g/iota,theta_p) / L_tot)

    # set numerical parameters
    ae_val = ae.ae_total(1.0,dlnTdx,dlnndx,Delta_x1,Delta_x2,b_norm,dbdx1,dbdx2,sqrt_g,theta_p,lam_res,delta_theta,del_sing,L_tot,omnigenous)
    if verbose==True:
        print("AE @ (s:{},alpha:{}) = {}".format(round(s_val,2),round(alpha_val,2),ae_val, nsmall = 2))

    if ae_val < 0:
        print('AE_val is: ', ae_val)
        print('AE is negative, something is wrong!!!')
    # return ae per unit volume per 3/2 n0*T0
    return ae_val/(rec_B_ave*6*np.sqrt(np.pi))



# penalize mirror ratio
def mirror_ratio_target(vmec,t=0.21):
    """
    Adopted from A. Goodman. Calculates Mirror near axis. Penalizes iff the target value is exceeded.
    vmec        -   the vmec object
    t           -   target value
    """
    vmec.run()
    xm_nyq = vmec.wout.xm_nyq
    xn_nyq = vmec.wout.xn_nyq
    bmnc = vmec.wout.bmnc.T
    bmns = 0*bmnc
    nfp = vmec.wout.nfp

    Ntheta = 100
    Nphi = 100
    thetas = np.linspace(0,2*np.pi,Ntheta)
    phis = np.linspace(0,2*np.pi/nfp,Nphi)
    phis2D,thetas2D=np.meshgrid(phis,thetas)
    b = np.zeros([Ntheta,Nphi])
    for imode in range(len(xn_nyq)):
        angles = xm_nyq[imode]*thetas2D - xn_nyq[imode]*phis2D
        b += bmnc[1,imode]*np.cos(angles) + bmns[1,imode]*np.sin(angles)
    Bmax = np.max(b)
    Bmin = np.min(b)
    m = (Bmax-Bmin)/(Bmax+Bmin)
    print("Mirror = ",m)
    pen = np.max([0,1-t/m])
    return pen


# volume target
def volume_target(vmec,minor_r=0.125):
    """
    Targets the volume. Can set minor_r to desired value. Penalizes iff volume is lower than desired.
    """
    vmec.run()
    vmec_volume = vmec.volume()
    appr_volume = 2*np.pi*(np.pi*minor_r**2.0)
    print("V_vmec/V_want = ",vmec_volume/appr_volume)
    pen = np.max([0,1-vmec_volume/appr_volume])
    return pen


# elongation target
def elongation_target(vmec,t=2.0):
    """
    Elongation target. Penalizes iff elongation is bigger than target.
    """
    vmec.run()
    surf = vmec.boundary
    vmec_area = surf.area()
    wout = vmec.wout
    a_minor = wout.Aminor_p
    r_major = wout.Rmajor_p
    # find best elliptic fit 
    kappa_list= fsolve(lambda x:  np.sqrt(x) * ellipe(1-1/x**2) + ellipe(1 - x**2)/np.sqrt(x) - vmec_area/(4 * np.pi * a_minor * r_major), [1.1])
    kappa_fit = kappa_list[0]
    if kappa_fit < 1:
        kappa_fit = 1/kappa_fit
    print("Elongation = ",kappa_fit)
    pen = np.max([0,kappa_fit-t])
    return pen


# aspect ratio target
def aspect_target(vmec,t=2.0):
    """
    Aspect ratio target. Penalizes iff aspect ratio is higher than target.
    """
    vmec.run()
    aspect_ratio = vmec.aspect()
    print('Aspect ratio = ', aspect_ratio)
    pen = np.max([0,aspect_ratio-t])
    return pen