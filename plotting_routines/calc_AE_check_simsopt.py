import sys  
sys.path.append('/Users/ralfmackenbach/Documents/GitHub/simsoptAE/AE_routines')  
import  numpy           as  np
import  AE_routines     as  ae
import  f90nml
import  re
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
import matplotlib.pyplot as plt
import warnings
from    matplotlib          import  cm
import  matplotlib          as      mpl
import  matplotlib.colors   as      mplc
warnings.filterwarnings("ignore")



def compute_ae_simsopt(fieldline,omn,omt,omnigenous=False,lam_res=1001,Delta_theta=1e-10,del_sing=0.0):
    # import relevant data from fieldline
    s           = (fieldline.s).flatten()
    iota        = (fieldline.iota).flatten()
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
    B_ref       = (fieldline.B_reference).flatten()
    L_ref       = (fieldline.L_reference).flatten()
    phi_edge    = (fieldline.edge_toroidal_flux_over_2pi).flatten()

    # we construct dBdpsi
    # Realize that gradB = dBdpsi * gradpsi + dBdalpha * gradalpha + dBdphi * gradphi
    # Take inner product of above expression with (gradalpha cross gradphi)
    # dBdpsi = gradB inner ( gradalpha cross grad phi ) / B_sup_phi
    gradphi         = np.stack([grad_phi_X,grad_phi_Y,grad_phi_Z],axis=1)
    gradpsi         = np.stack([grad_psi_X,grad_psi_Y,grad_psi_Z],axis=1)
    gradalpha       = np.stack([grad_alpha_X,grad_alpha_Y,grad_alpha_Z],axis=1)
    gradB           = np.stack([grad_B_X,grad_B_Y,grad_B_Z],axis=1)
    gradalphaXgradphi = np.cross(gradalpha, gradphi)
    gradBingradphiXgradpsi  = np.einsum('ij, ij->i',gradalphaXgradphi,gradB)
    jac             = 1/B_sup_phi
    d_B_d_psi     = gradBingradphiXgradpsi / B_sup_phi

    # now we construct dBdalpha
    # Realize that gradB = dBdpsi * gradpsi + dBdalpha * gradalpha + dBdphi * gradphi
    # Take inner product of above expression with (gradphi cross gradpsi)
    # dBdalpha = gradB inner ( gradphi cross grad psi ) / B_sup_phi
    gradphiXgradpsi = np.cross(gradphi, gradpsi)
    gradBingradphiXgradpsi  = np.einsum('ij, ij->i',gradphiXgradpsi,gradB)
    jac             = 1/B_sup_phi
    d_B_d_alpha     = gradBingradphiXgradpsi / B_sup_phi

    # now convert to dimless quantities as in GIST
    dbdx1       = d_B_d_psi * 2 * np.sqrt(s) / B_ref * phi_edge
    dbdx2       = d_B_d_alpha / ( np.sqrt(s) * B_ref )
    b_norm      = modB/B_ref
    sqrt_g      = jac/(L_ref**3 * iota / 2)*phi_edge

    # set scalars
    dlnTdx = -omt
    dlnndx = -omn
    Delta_x1= 1/iota
    Delta_x2= 1/iota
    L_tot  = np.trapz(b_norm*sqrt_g/iota,theta_p)

    # set numerical parameters
    bw, lam_arr, ae_list, ae_tot = ae.ae_total_over_z(1/iota,dlnTdx,dlnndx,Delta_x1,Delta_x2,b_norm,dbdx1,dbdx2,sqrt_g,theta_p,lam_res,Delta_theta,del_sing,L_tot,omnigenous=omnigenous)
    return bw, lam_arr, ae_list, ae_tot, dbdx1, b_norm, theta_p




def plot_AE_per_bouncewell(theta_arr,b_arr,dbdx_arr,lam_arr,bw,ae_list,n_pol):
    c = 0.5
    # shift by pi
    fig ,ax = plt.subplots()
    fig.set_size_inches(8*5, 3.75)
    ax.set_xlim(min(theta_arr),max(theta_arr))

    list_flat = []
    for val in ae_list:
        list_flat.extend(val)

    max_val = max(list_flat)
    cm_scale = lambda x: x
    colors_plot = [cm.plasma(cm_scale(np.asarray(x) * 1.0/max_val)) for x in ae_list]

    # iterate over all values of lambda
    for idx_lam, lam in enumerate(lam_arr):
        b_val = 1/lam

        # iterate over all bounce wells
        for idx_bw, _ in enumerate(ae_list[idx_lam]):
            # check if well crosses boundary
            if(bw[idx_lam][idx_bw][0] > bw[idx_lam][idx_bw][1]):
                ax.plot([bw[idx_lam][idx_bw][0], max(theta_arr)], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])
                ax.plot([min(theta_arr), bw[idx_lam][idx_bw][1]], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])
            # if not normal plot
            else:
                ax.plot([bw[idx_lam][idx_bw][0], bw[idx_lam][idx_bw][1]], [b_val, b_val], color=colors_plot[idx_lam][idx_bw])

    ax.plot(theta_arr,b_arr,color='black',linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(theta_arr, dbdx_arr, 'red')
    ax.set_ylabel(r'$B$')
    ax2.set_ylabel(r'$\partial B/ \partial x$',color='red')
    ax2.plot(theta_arr,theta_arr*0.0,linestyle='dashed',color='red')
    ax2.tick_params(axis='y', colors='black',labelcolor='red',direction='in')
    ax.set_xlabel(r'$\theta/n_{pol}$')
    ax.set_xticks([n_pol*-np.pi,n_pol*-np.pi/2,0,n_pol*np.pi/2,n_pol*np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$',r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax.tick_params(axis='both',direction='in')
    ax.set_title(r'AE per bounce well')
    cbar = plt.colorbar(cm.ScalarMappable(norm=mplc.Normalize(vmin=0.0, vmax=max_val, clip=False), cmap=cm.plasma), ticks=[0, max_val], ax=ax,location='bottom',label=r'$\widehat{A}_\lambda$') #'%.3f'
    cbar.ax.set_xticklabels([0, round(max_val, 1)])
    plt.show()




# Run the script
norm_input   = input("Proceed at (s,phi,omn,omt,n_pol)=(0.5,0.0,3.0,0.0,1)? (y/n): ")
omni   = False


if norm_input=='y':
    s_val   = 0.5
    phi_val = 0.0
    omn     = 3.0
    omt     = 0.0
    n_pol   = int(1)

elif norm_input=='n':
    s_val       = float(input("Enter s value: "))
    phi_val     = float(input("Enter phi_center value: "))
    omn         = float(input("Enter omn value: "))
    omt         = float(input("Enter omt value: "))
    n_pol       = int(input("Enter n_pol int: "))

vmec = Vmec(filename="wout.nc",verbose=False)
vmec.run()

gpts=128*n_pol
fieldline = vmec_fieldlines(vmec,s_val,0.0,theta1d=np.linspace(-n_pol*np.pi,n_pol*np.pi,gpts),
                                    phi_center=phi_val)
bw, lam_arr, ae_list, ae_tot, dbdx1, b_norm, theta_p = compute_ae_simsopt(fieldline,omn,omt,omnigenous=omni,lam_res=1000,Delta_theta=1e-10,del_sing=0.0)

print("Available energy is: ", ae_tot)

plot_AE_per_bouncewell(theta_p,b_norm,dbdx1,lam_arr,bw,ae_list,n_pol)
