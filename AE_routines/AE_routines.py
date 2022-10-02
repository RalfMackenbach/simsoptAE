# This file contains all subroutines used in the calculation of the Available
# Energy of trapped particles. Based on fortran implementation
import  numpy           as      np
from    numba           import  jit
from    scipy.signal    import  find_peaks
import  scipy.integrate as      integrate
from    scipy           import  special
import  quadpy




@jit
def zero_cross_idx(y_arr):
    """
    Returns all indices left of zeros of y. Based on "Numerical Recipes in
    Fortran 90" (1996), chapter B9, p.1184. Also returns number of crossings.

    Takes as input:
    y_arr   -   array of which zero points are to be determined

    Returns:
    zero_idx, num_crossings
    """

    l_arr  = y_arr[0:-1]
    r_arr  = y_arr[1:]

    # check zero crossings, returns TRUE left of the crossing. Also, count number
    # of crossings
    mask      = l_arr*r_arr < 0
    # Return indices where mask is true, so left of crossings
    zero_idx = np.asarray(np.nonzero(mask))
    return zero_idx[0], np.sum(mask)


@jit
def inner_bounce_trapz(h_i,h_j,f_i,f_j,theta_i,theta_j):
    """
    Estimation of integral of h/sqrt(f) dtheta, where we assume h and f to be
    well approximated by their linear interpolation between theta_i and theta_j.
    Only estimate the INNER part (i.e. not accounting for the edges where f
    vanshises). These are handled seperately.
    """
    y = np.sum((-2*(np.sqrt(f_j)*(2*h_i + h_j) + np.sqrt(f_i)*(h_i + 2*h_j))*
          (theta_i - theta_j))/(3.*(f_i + f_j + 2*np.sqrt(f_i*f_j))))
    return y


@jit
def left_bounce_trapz(h_l,h_0,f_l,f_0,theta_l,theta_0):
    """
    Estimation of integral of edge integral h/sqrt(f) dtheta, from theta_l where
    f = 0, to the first theta node to its right theta_0
    """
    y = 2*(2*h_l + h_0)*(theta_0 - theta_l)/(3.*np.sqrt(f_0))
    return y


@jit
def right_bounce_trapz(h_n,h_r,f_n,f_r,theta_n,theta_r):
    """
    Estimation of integral of edge integral h/sqrt(f) dtheta, from theta_n to theta_r
    where f = 0
    """
    y = 2*(h_n + 2 * h_r)*(-theta_n + theta_r)/(3.*np.sqrt(f_n))
    return y


@jit
def bounce_wells(theta_arr,b_arr,lam_val):
    """
    ! This routine calculates the bounce points and turns them into bounce wells.
    ∫h(ϑ)/sqrt(1-lam*b(ϑ))·dϑ
    theta_arr   -   Array containing theta nodes
    h_arr       -   Array containing h values
    b_arr       -   Array containing normalized magnetic field values
    lam         -   Lambda value at which we wish to calculate quantities
    Returns two arrays and one scalar
    bounce_idx, bounce_arr, num_wells
    Each row of bounce_idx contains the two bounce indices left of the bounce points
    Each row of bounce_arr contains the two bounce points (in theta)
    Number of wells
    """
    # define function of which we need zero crossings, and retrieve crossings
    zero_arr = 1.0 - lam_val * b_arr
    zero_idx, num_cross = zero_cross_idx(zero_arr)
    # Check if even number of wells
    if (np.mod(num_cross,2)!=0):
        print('ERROR: odd number of well crossings, please adjust lambda resolution')
    # Calculate number of wells
    num_wells = int(num_cross/2)


    # Check if the first crossing is end of well
    if  ( b_arr[zero_idx[0]+1] - b_arr[zero_idx[0]] > 0 ):
        first_well_end = 1
    else:
        first_well_end = 0


    # First let's fill up the bounce_idx array
    # If well crosses periodicity we must shift the indices
    if (first_well_end == 1):
        zero_idx = np.roll(zero_idx,-1)

    # make array holding bounce well information
    bounce_idx = np.empty([num_wells,2],np.float_)
    bounce_arr = np.empty([num_wells,2],np.float_)

    # Fill up bounce array
    for  do_idx in range(0,num_wells):
      l_idx                     = zero_idx[2*do_idx]
      r_idx                     = zero_idx[2*do_idx+1]
      bounce_idx[do_idx,  0]    = l_idx
      bounce_idx[do_idx,  1]    = r_idx
      bounce_arr[do_idx,  0]    = (-(zero_arr[l_idx+1]*theta_arr[l_idx]) +
                                  zero_arr[l_idx]*theta_arr[l_idx+1])/(zero_arr[l_idx] -
                                  zero_arr[l_idx+1])
      bounce_arr[do_idx,  1]    = (-(zero_arr[r_idx+1]*theta_arr[r_idx]) +
                                  zero_arr[r_idx]*theta_arr[r_idx+1])/(zero_arr[r_idx] -
                                  zero_arr[r_idx+1])

    return bounce_idx, bounce_arr, num_wells


@jit
def bounce_average(theta_arr,h_arr,b_arr,lam):
    """
    Does the bounce averaging operation, i.e. calculates
    ∫h(ϑ)/sqrt(1-lam*b(ϑ))·dϑ
    theta_arr   -   Array containing theta nodes
    h_arr       -   Array containing h values
    b_arr       -   Array containing normalized magnetic field values
    lam         -   Lambda value at which we wish to calculate quantities
    """
    # Find the bounce wells
    bounce_idx, bounce_arr, num_wells = bounce_wells(theta_arr,b_arr,lam)
    bounce_ave = np.empty(num_wells,np.float_)
    f_arr   = 1 - lam*b_arr

    l_idx   = bounce_idx[:,0]
    r_idx   = bounce_idx[:,1]
    l_cross = bounce_arr[:,0]
    r_cross = bounce_arr[:,1]

    # check if well crosses periodicity boundary
    for  do_idx in range(0,num_wells):
        l = int(l_idx[do_idx])
        r = int(r_idx[do_idx])
        if (l_idx[do_idx]>r_idx[do_idx]):
            # Split up inner int into two parts
            # first left-to-end
            h_i     = h_arr[(l + 1):-1]
            h_j     = h_arr[(l + 2):]
            f_i     = f_arr[(l + 1):-1]
            f_j     = f_arr[(l + 2):]
            theta_i = theta_arr[(l + 1):-1]
            theta_j = theta_arr[(l + 2):]
            y_l = inner_bounce_trapz(h_i,h_j,f_i,f_j,theta_i,theta_j)
            # then start-to-right
            h_i     = h_arr[(0):(r)]
            h_j     = h_arr[(1):(r+1)]
            f_i     = f_arr[(0):(r)]
            f_j     = f_arr[(1):(r+1)]
            theta_i = theta_arr[(0):(r)]
            theta_j = theta_arr[(1):(r+1)]
            y_r = inner_bounce_trapz(h_i,h_j,f_i,f_j,theta_i,theta_j)
            inner = y_l + y_r
    # otherwise business as usual
        else:
            h_i     = h_arr[(l + 1):(r)]
            h_j     = h_arr[(l + 2):(r+1)]
            f_i     = f_arr[(l + 1):(r)]
            f_j     = f_arr[(l + 2):(r+1)]
            theta_i = theta_arr[(l + 1):(r)]
            theta_j = theta_arr[(l + 2):(r+1)]
            inner = inner_bounce_trapz(h_i,h_j,f_i,f_j,theta_i,theta_j)


        # Now do the edge integrals
        h_l = h_arr[l] + (l_cross[do_idx] -
              theta_arr[l])/(theta_arr[l+1] -
              theta_arr[l]) * ( h_arr[l+1] -
              h_arr[l] )
        left = left_bounce_trapz(h_l,h_arr[l+1],0.0,
                               f_arr[l+1],l_cross[do_idx],
                               theta_arr[l+1])
        h_r = h_arr[r] + (r_cross[do_idx] -
              theta_arr[r])/(theta_arr[r+1] -
              theta_arr[r]) * ( h_arr[r+1] -
              h_arr[r] )
        right = right_bounce_trapz(h_arr[r],h_r,f_arr[r],
                                0.0,theta_arr[r],r_cross[do_idx])

        # finally, fill in full integral!
        bounce_ave[do_idx]= left + inner + right

    return bounce_ave


@jit
def w_bounce(q0,L_tot,b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr,lam,Delta_x,Delta_y):
    """
    Calculate the drift frequencies and bounce time
    """
    h_arr_0 = q0 * b_arr * sqrtg_arr
    denom_arr = bounce_average(theta_arr,h_arr_0,b_arr,lam)

    h_arr_1 = lam * Delta_x * dbdx_arr * q0 * b_arr * sqrtg_arr
    numer_arr_alpha = bounce_average(theta_arr,h_arr_1,b_arr,lam)

    h_arr_2 = -1.0 * lam * Delta_y * dbdy_arr * q0 * b_arr * sqrtg_arr
    numer_arr_psi = bounce_average(theta_arr,h_arr_2,b_arr,lam)


    # Make arrays for w_psi, w_alpha, and G
    w_psi_arr   = numer_arr_psi   / denom_arr
    w_alpha_arr = numer_arr_alpha / denom_arr
    G_arr       = denom_arr       / L_tot
    return w_psi_arr, w_alpha_arr, G_arr


@jit
def w_bounce_and_wells(q0,L_tot,b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr,lam,Delta_x,Delta_y):
    """
    Calculate the drift frequencies and bounce time
    """
    h_arr_0 = q0 * b_arr * sqrtg_arr
    denom_arr = bounce_average(theta_arr,h_arr_0,b_arr,lam)

    h_arr_1 = lam * Delta_x * dbdx_arr * q0 * b_arr * sqrtg_arr
    numer_arr_alpha = bounce_average(theta_arr,h_arr_1,b_arr,lam)

    h_arr_2 = -1.0 * lam * Delta_y * dbdy_arr * q0 * b_arr * sqrtg_arr
    numer_arr_psi = bounce_average(theta_arr,h_arr_2,b_arr,lam)

    bounce_idx, bounce_arr, num_wells = bounce_wells(theta_arr,b_arr,lam)


    # Make arrays for w_psi, w_alpha, and G
    w_psi_arr   = numer_arr_psi   / denom_arr
    w_alpha_arr = numer_arr_alpha / denom_arr
    G_arr       = denom_arr       / L_tot
    return w_psi_arr, w_alpha_arr, G_arr, bounce_arr


@jit
def ae_integrand(walpha,wpsi,G,dlnTdx,dlnndx,Delta_x,z):
    """
    The integrand of the AE (summed over all bounce wells hence scalar)

    Takes as input:
    walpha      -   Array containing binormal drift for all bounce wells
    wpsi        -   Array containing radial drift for all bounce wells
    G           -   Array containing bounce times for all bounce wells
    dlnTdx      -   The radial electron temperature variation
    dlnndx      -   The radial electron density variation
    Delta_x     -   Radial size over which energy is available
    z           -   Normalized energy

    Returns:
    AE integrand
    """
    wdia = Delta_x * ( dlnndx/z + dlnTdx * ( 1.0 - 3.0 / (2.0 * z) ) )
    AE = np.sum((G*(walpha*(-walpha + wdia) - wpsi**2 +
         np.sqrt(walpha**2 + wpsi**2)*np.sqrt((walpha - wdia)**2 + wpsi**2))*z**2.5)*np.exp(-z))
    return AE


@jit
def ae_integrand_vec(walpha,wpsi,G,dlnTdx,dlnndx,Delta_x,z):
    """
    The integrand of the AE (summed over all bounce wells hence scalar)

    Takes as input:
    walpha      -   Array containing binormal drift for all bounce wells
    wpsi        -   Array containing radial drift for all bounce wells
    G           -   Array containing bounce times for all bounce wells
    dlnTdx      -   The radial electron temperature variation
    dlnndx      -   The radial electron density variation
    Delta_x     -   Radial size over which energy is available
    z           -   Normalized energy

    Returns:
    AE integrand
    """
    wdia = Delta_x * ( dlnndx/z + dlnTdx * ( 1.0 - 3.0 / (2.0 * z) ) )
    AE = (G*(walpha*(-walpha + wdia) - wpsi**2 +
         np.sqrt(walpha**2 + wpsi**2)*np.sqrt((walpha - wdia)**2 + wpsi**2))*z**2.5)*np.exp(-z)
    return AE


@jit
def ae_integrand_GL(walpha,wpsi,G,dlnTdx,dlnndx,Delta_x,z):
    """
    The integrand of the AE (summed over all bounce wells hence scalar)
    Adjusted for gauss laguerre quadrature rules. Works poorly in weakly
    driven regimes. May be faster in strongly driven regimes.

    Takes as input:
    walpha      -   Array containing binormal drift for all bounce wells
    wpsi        -   Array containing radial drift for all bounce wells
    G           -   Array containing bounce times for all bounce wells
    dlnTdx      -   The radial electron temperature variation
    dlnndx      -   The radial electron density variation
    Delta_x     -   Radial size over which energy is available
    z           -   Normalized energy

    Returns:
    AE integrand
    """
    wdia = Delta_x * ( dlnndx/z + dlnTdx * ( 1.0 - 3.0 / (2.0 * z) ) )
    AE = z**2.5 * np.sum((G*(walpha*(-walpha + wdia) - wpsi**2 +
         np.sqrt(walpha**2 + wpsi**2)*np.sqrt((walpha - wdia)**2 + wpsi**2))))
    return AE


@jit
def make_per(b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr,Delta_theta):
    """
    Makes arrays periodic by appending first value to last, and for
    theta a small padding region of theta_last + Delta_theta is added.
    """
    # make periodic
    b_arr_p    = np.append(b_arr,b_arr[0])
    dbdx_arr_p = np.append(dbdx_arr,dbdx_arr[0])
    dbdy_arr_p = np.append(dbdy_arr,dbdy_arr[0])
    sqrtg_arr_p= np.append(sqrtg_arr,sqrtg_arr[0])
    theta_arr_p= np.append(theta_arr,theta_arr[-1]+Delta_theta)

    return b_arr_p, dbdx_arr_p, dbdy_arr_p, sqrtg_arr_p, theta_arr_p



def lambda_filtered(lambda_arr,B_arr,delta_lambda):
    """
        Returns filtered array, where all values in
        lambda arr which lie within delta_lambda*range(beta)
        are deleted.

        lambda_arr      - array with lambda values
        B_arr           - array with B values
        delta_lambda    - the singularity padding
    """
    #make new container
    lambda_arr_new  = lambda_arr
    # find idx of local maxima of beta arrays
    B_max_idx       = find_peaks(B_arr)[0]
    # Find corresponding beta vals and lambda vals
    B_local_max  = np.asarray([B_arr[i] for i in B_max_idx])
    lamdba_inf      = 1.0 / (  B_local_max )
    # construct range(lambda)
    max_B    = np.amax(B_arr)
    min_B    = np.amin(B_arr)
    lambda_max  = 1.0/(min_B)
    lambda_min  = 1.0/(max_B)
    lambda_range= lambda_max-lambda_min
    # loop over lambda inf and delete lambdas within singularity padding
    for lambda_inf_val in lamdba_inf:
        lower_bound         =   lambda_inf_val - delta_lambda*lambda_range
        upper_bound         =   lambda_inf_val + delta_lambda*lambda_range
        lambda_delete_idx   =   np.where(np.logical_and(lambda_arr_new>=lower_bound, lambda_arr_new<=upper_bound))
        lambda_arr_new      =   np.delete(lambda_arr_new, lambda_delete_idx)

    return lambda_arr_new




# integral over z
def integral_over_z(c0,c1):
    """
    Integral over normalized energies is analytical if omnigenous
    """
    if (c0>=0) and (c1<=0):
        return 2 * c0 - 5 * c1
    if (c0>=0) and (c1>0):
        return (2 * c0 - 5 * c1) * special.erf(np.sqrt(c0/c1)) + 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0 + 15 * c1 ) * np.sqrt(c0/c1) * np.exp( - c0/c1 )
    if (c0<0)  and (c1<0):
        return ( (2 * c0 - 5 * c1) * (1 - special.erf(np.sqrt(c0/c1))) - 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0 + 15 * c1 ) * np.sqrt(c0/c1) * np.exp( - c0/c1 ) )
    else:
        return 0.


vint = np.vectorize(integral_over_z, otypes=[np.float64])


def ae_total(q0,dlnTdx,dlnndx,Delta_x,Delta_y,b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,
             theta_arr,lam_res,Delta_theta,del_sing,L_tot,omnigenous=False,GL=False,N_laguerre=100):
    if (omnigenous == False) and (GL == True):
        scheme = quadpy.e1r.gauss_laguerre(N_laguerre)
    # make arrays periodic
    b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr = make_per(b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr,Delta_theta)

    # calculate the lambda range
    lam_min = 1.0/(np.amax(b_arr))
    lam_max = 1.0/(np.amin(b_arr))

    # make arrays for lambda
    lam_arr = np.linspace(lam_min,lam_max,lam_res+1,endpoint=False)
    lam_arr = np.delete(lam_arr, 0)
    lam_arr = lambda_filtered(lam_arr,b_arr,del_sing)

    # Loop over lambda indices
    ae_per_lam = np.empty(lam_res)
    for lam_idx, lam_val in np.ndenumerate(lam_arr):
        w_psi_arr, w_alpha_arr, G_arr = w_bounce(q0,L_tot,b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr,lam_val,Delta_x,Delta_y)
        if omnigenous == True:
            c0 = Delta_x * (dlnndx - 3/2 * dlnTdx) / w_alpha_arr
            c1 = 1.0 - Delta_x * dlnTdx / w_alpha_arr
            ae_per_lam[lam_idx] = 3/4 * np.sqrt(np.pi) * np.sum((w_alpha_arr**2.0) * vint(c0,c1) * G_arr)
        elif omnigenous == False:
            if GL == False:
                ae_per_lam[lam_idx] = integrate.quad(lambda z: ae_integrand(w_alpha_arr,w_psi_arr,G_arr,dlnTdx,dlnndx,Delta_x,z), 0, np.inf, epsrel=1e-6,epsabs=1e-20, limit=1000)[0]
            elif GL == True:
                func = lambda z: ae_integrand_GL(w_alpha_arr,w_psi_arr,G_arr,dlnTdx,dlnndx,Delta_x,z)
                aev  = np.vectorize(func)
                ae_per_lam[lam_idx] = scheme.integrate(lambda z: aev(z))[0]


    # calculate ae final
    ae = np.trapz(ae_per_lam,lam_arr)
    return ae


def ae_total_over_z(q0,dlnTdx,dlnndx,Delta_x,Delta_y,b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr,lam_res,Delta_theta,del_sing,L_tot,omnigenous=False):
    # check list depth
    depth = lambda L: isinstance(L,list) and max(map(depth,L))+1

    # make arrays periodic
    b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr = make_per(b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr,Delta_theta)

    # calculate the lambda range
    lam_min = 1.0/(np.amax(b_arr))
    lam_max = 1.0/(np.amin(b_arr))

    # make arrays for lambda
    lam_arr = np.linspace(lam_min,lam_max,lam_res+1,endpoint=False)
    lam_arr = np.delete(lam_arr, 0)
    lam_arr = lambda_filtered(lam_arr,b_arr,del_sing)
    ae_lam  = np.empty_like(lam_arr)

    # Loop over lambda indices
    ae_per_lam = []
    ae_bw      = []
    for lam_idx, lam_val in np.ndenumerate(lam_arr):
        w_psi_arr, w_alpha_arr, G_arr, bounce_arr = w_bounce_and_wells(q0,L_tot,b_arr,dbdx_arr,dbdy_arr,sqrtg_arr,theta_arr,lam_val,Delta_x,Delta_y)
        ae_bw.append(bounce_arr.tolist())
        if omnigenous == True:
            c0 = Delta_x * (dlnndx - 3/2 * dlnTdx) / w_alpha_arr
            c1 = 1.0 - Delta_x * dlnTdx / w_alpha_arr
            vals = 3/4 * np.sqrt(np.pi) * (w_alpha_arr**2.0) * vint(c0,c1) * G_arr
            ae_per_lam.append( vals.tolist() )
            ae_lam[lam_idx]  = np.sum(vals)
        elif omnigenous == False:
            ae_over_z, err = integrate.quad_vec(lambda z: ae_integrand_vec(w_alpha_arr,w_psi_arr,G_arr,dlnTdx,dlnndx,Delta_x,z), 0, np.inf, epsrel=1e-6,epsabs=1e-20, limit=1000)
            ae_per_lam.append(list(ae_over_z))
            ae_lam[lam_idx]  = np.sum(ae_over_z)

    ae = np.trapz(ae_lam,lam_arr)


    return ae_bw, lam_arr, ae_per_lam, ae
