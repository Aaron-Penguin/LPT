#from panel import widget

# %%
import os
import numpy as np
import scipy as ss                    # stats, signal
import matplotlib.pyplot as plt
%matplotlib qt
from classy import Class
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
 


from mpl_toolkits.axes_grid1 import make_axes_locatable  # Kun for at plotte colorbar uden for conturplotet.
import matplotlib.colors as mcolors                      # Kun for at plotte colorbar for begge contur med samme farve.

plt.rc("axes", labelsize=14, titlesize=22)
plt.rc("xtick", labelsize=14, top=True, direction="in")
plt.rc("ytick", labelsize=14, right=True, direction="in")
plt.rc("legend", fontsize=13, loc="upper left")
plt.rcParams["font.size"] = "26"

z_redshift = 0
N = 64  # gitter størrelse
L = 500  # Box størrelse i Mpc

fac = np.sqrt(2*np.pi/(L**3)) * N**3    # Korigerings faktor, da der arbejdes med diskrete fourie transform og delta genereries i k rum.


def test_fourie_transform(Real_space, Real_space_2):
    #Ny_K_space = np.fft.fftn(Real_space)

    print( np.allclose(Real_space, Real_space_2) )

    Afvigelse = np.abs( (Real_space - Real_space_2))   
    print("Max afvigelse:", np.max(Afvigelse))
    print("Middel afvigelse:", np.mean(Afvigelse))


params = {
        'h': 0.6736,  # Hubble parameter
        'omega_b': 0.02237,  # Baryon density
        'omega_cdm': 0.1200,  # Cold dark matter density
        'A_s': 2.1e-9,    #10e-10,  # Scalar amplitude
        'n_s': 0.9649,  # Spectral index
        'output': 'mPk',  # Request matter power spectrum
        'P_k_max_1/Mpc': 100,  # Maximum k value in 1/Mpc
        'z_max_pk': 100,  # Ensures CLASS computes P(k) for requested redshifts
        'z_pk': z_redshift,
    }

cosmo = Class()
cosmo.set(params)
cosmo.compute()


# %%
########################### Vægst factorende D1 og D2 ###############################

Omega_M = (params['omega_b'] + params['omega_cdm'] )/(params['h']**2)
Omega_L = 1 - Omega_M
H_0 = 100*params['h']    # i (km/s) / Mpc 


def D1_D2_solver(ploting = False):

    def dudt(t, y):
        a, D1, D1_dot, D2, D2_dot = y
        H_a = H_0 * np.sqrt(Omega_M / (a**3) + Omega_L)
        da_dt = a**2 * H_a

        D1_ddot = - (1/a) * da_dt * D1_dot + (3/2) * Omega_M / a * D1 * H_0**2
        D2_ddot = - (1/a) * da_dt * D2_dot + (3/2) * Omega_M / a * (D2 + D1**2) * H_0**2

        return [da_dt, D1_dot, D1_ddot, D2_dot, D2_ddot]

    initial_val = [1e-9, 1e-9, 0.0, 0.0, 0.0]

    # === Integration range ===
    t_span = (0, 0.5)
    t_eval = np.linspace(*t_span, 10**5)

    def event_stop(t, y):   # Stopper löseren Når a = 1
        return y[0] - 1 

    event_stop.terminal = True  # Der stoppes ved det givne event
    event_stop.direction = 0    # Find krysning med 0

    sol = solve_ivp(dudt, t_span = t_span, y0 = initial_val, t_eval=t_eval,
                     rtol=1e-10, events= event_stop, method='RK45')

    # === Extract solution ===
    a_val = sol.y[0]
    D1 = sol.y[1]
    D2 = sol.y[3]  

    # === Normalize ===
    D1_norm = D1 / D1[-1]
    D2_norm = -D2 / D1[-1]**2 # Det negative fortegn er nødvendigt, 
                                #for at korrigere mellem notationer.

    D2_interp = interp1d(a_val, D2_norm, kind='cubic', bounds_error=False, fill_value='extrapolate')  # Fra Jonny
    D2_final = lambda a:  D2_interp(a)

    if ploting == True:
        #################### Approximation for D2 #############

        D2_approx = - (3/7) * D1_norm**2 * Omega_M**-(1/143)
        #D2_Einstein_desitter = (3/7)*a_val**2
        #################### Analytisk for D1 ##################
        u = lambda a : - Omega_L/(Omega_M)  * a**3
        F_analytic = lambda u:  ss.special.hyp2f1( 1/3, 1, 11/6, u)

        D_analytisk = a_val* F_analytic(u(a_val))/F_analytic(u(1))
        ###################   Class løsning for D1  #################

        D1_Class = [cosmo.scale_independent_growth_factor(1/A - 1) for A in a_val] #a_Class]

        plt.figure(figsize=(10,5))
        plt.loglog(a_val, D1_Class, color='k', linestyle='-', linewidth = 2.5 ,label= r'$D_1$ [Fra CLASS]')
        plt.loglog(a_val, D1_norm, label=r'$D_1$ [Numerisk]' ,color='C0', lw=2.5, alpha = 1)
        plt.loglog(a_val, D_analytisk, color='C5', lw = 2.5, linestyle= '--', label= f'$D_1$ [Analytisk]')

        plt.loglog(a_val, abs(D2_norm), color='C1', label=r'$ | D_2 |$ [Numerisk]', lw=2)
        plt.loglog(a_val, abs(D2_approx), '--', color='g', label=r'$ | D_2 | \approx \frac{3}{7} D_1^2 \Omega_{M,0} ^{-1/143}$', lw=2)
        plt.xlabel('Skalafaktoren $a$')

        plt.axvline(x= 2*10**(-4), color='r', linestyle='-', linewidth=1.5)
        plt.ylabel('')
        plt.legend(loc='best')
        plt.grid(True)
        #plt.title('Second Order Growth Factors')

        plt.tight_layout()
        plt.show()

    return D2_final


D2 = D1_D2_solver(ploting=True)




# %%


""" Generering af tilfældige Gausiske kompleks tal."""

def Random_R():                  # Hermittisk symetri
    C = ss.stats.norm.rvs(loc= 0, scale= 1/np.sqrt(2), size=(2*N**3)).reshape(2, N**3)
    R = (C[0] + C[1]*1j).reshape(N, N, N)

    for i in range(N):
        for j in range(N):
            for k in range(N):  # Only loop over half the space
                ni, nj, nk = (-i % N), (-j % N), (-k % N)
                #if (i, j, k) != (ni, nj, nk):  # Avoid self-overwriting  "Ikke nødvendigt, men sparer tid"
                R[ni, nj, nk] = np.conj(R[i, j, k])
                #R_k[i][j][k]  = np.conj( R_k[ N - 1- i][N - 1 -j][N -1 -k] )

    # Ensure Nyquist frequencies are purely real (only needed for even N)

    if N % 2 == 0:
        R[N//2, :, :] = R[N//2, :, :].real
        R[:, N//2, :] = R[:, N//2, :].real
        R[:, :, N//2] = R[:, :, N//2].real

    # Ensure the zero-frequency mode is real
    R[0, 0, 0] = R[0, 0, 0].real
    return R

#R = Random_R()


# Find k_values and delta_k
dL = L/N  # Grid size in Mpc
k = np.fft.fftfreq(N, d = dL) * 2 * np.pi                           # Konverter til 1/Mpc
kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
k_mag = np.sqrt(kx**2 + ky**2 + kz**2)                          # k_mag har flere af de samme tal, hvilket medfører at P_k beregnes uefektivt. Når det tilfældige komplekse tal dog er tilføjet vil de dog adskille sig.
k_mag[0, 0, 0] = 10**-10                                        # Undgår at dele med 0.
# %%
#print("Maksimalt k_mag=", np.max(k_mag))

#print(np.min(k_mag))

#print(D2(1))        # skal være negativt omkring -0,432

# %%

      
def CIC(X_s, Y_s, Z_s, Psi_x, Psi_y, Psi_z, N, L):
    dL = L / N

    # Koordinater efter forskydning i enheder af grid punkter
    X_f = ((X_s + Psi_x.real.flatten()) / dL) % N
    Y_f = ((Y_s + Psi_y.real.flatten()) / dL) % N
    Z_f = ((Z_s + Psi_z.real.flatten()) / dL) % N

    mass_grid = np.zeros((N, N, N)) 

     # Determine the lower grid index for each particle (with periodic wrapping)
    i_lower = np.floor(X_f).astype(int) % N
    j_lower = np.floor(Y_f).astype(int) % N
    k_lower = np.floor(Z_f).astype(int) % N
    
    # Compute the fractional distance of each particle within its cell
    dx = X_f - np.floor(X_f)
    dy = Y_f - np.floor(Y_f)
    dz = Z_f - np.floor(Z_f)
    
    # Loop over the 8 surrounding grid points (offsets 0 or 1 in each direction)
    for ox, oy, oz in [(0,0,0), (0,0,1), (0,1,0), (0,1,1),
                       (1,0,0), (1,0,1), (1,1,0), (1,1,1)]:
        # Compute the weights: if the offset is 0, weight is (1 - fraction); if 1, weight is the fraction.
        wx = (1 - dx) if ox == 0 else dx
        wy = (1 - dy) if oy == 0 else dy
        wz = (1 - dz) if oz == 0 else dz
        weight = wx * wy * wz #* m
        
        # Compute the grid indices for this vertex (with periodic boundary conditions)
        i_idx = (i_lower + ox) % N
        j_idx = (j_lower + oy) % N
        k_idx = (k_lower + oz) % N
        
        # Deposit the particle weights into the density grid using numpy's in-place addition.
        np.add.at(mass_grid, (i_idx, j_idx, k_idx), weight) #vectorized +=, er ala et loop, tager kun 0.5 sek though så er ikke så slem
        """Gør det samme som np.add.at men er 20x langsommere (10 s vs. 0.5 s):"""
        # for i in range(len(i_idx)):
        #     density[(i_idx[i], j_idx[i], k_idx[i])] +=weight[i] 
    return mass_grid


def test_mass_conservation(X_initial, CIC_mass_array):
    total_mass_initial = len(X_initial)  # Oprindelige masse er antal af totale gridpunkter, alle med masse 1.
    total_mass_after = np.sum(CIC_mass_array)    # Mass_grid fra CIC. Efter forskydning med displacement feltet. 

    mass_conserved = np.isclose(total_mass_initial, total_mass_after) # Checker hvor størrelsen af afvigelsen.

    print(f"Oprindelig total masse: {total_mass_initial}, Total masse efter CIC: {total_mass_after}")
    print(f"Masse bevarelse test: {mass_conserved}")

    if not mass_conserved:
        print("Warning: Mass is not conserved!")


def Second_LPT(Psi_kx, Psi_ky, Psi_kz, z_redshift):    # Funktionen er testet i en seperat fil.

    Psi_k_vek = [Psi_kx, Psi_ky, Psi_kz]
    k_vec = [kx, ky, kz]

    # Compute the convolution term in Fourier space
    Sum_k  = np.zeros((N,N,N), dtype = 'complex128')   #,  dtype=Psi_kx.dtype)
    #print("Reale sum" ,Sum_real_space.size)

    for i in range(3):
        for j in range(3):
            if i > j:
                psi1_i_i = 1j * k_vec[i] * Psi_k_vek[i]  # F[ Psi_i,i ]
                psi1_j_j = 1j * k_vec[j] * Psi_k_vek[j]  # F[ Psi_j,j ]
                psi1_i_j = 1j * k_vec[j] * Psi_k_vek[i]  # F[ Psi_i,j ]       # Fourier derivative of psi1 along j
                psi1_j_i = 1j * k_vec[i] * Psi_k_vek[j]  # F[ Psi_j,i ]
            

                term1 = np.fft.ifftn(psi1_i_i) * np.fft.ifftn(psi1_j_j) #*(1/dx**6)  # Foldning i k rum. Multiplikation i real rum.
                term2 = np.fft.ifftn(psi1_i_j) * np.fft.ifftn(psi1_j_i) #*(1/dx**6)

                term1_in_k = np.fft.fftn(term1) #*(dx**3)
                term2_in_k = np.fft.fftn(term2) #*(dx**3)
                
                Sum_k += (term1_in_k - term2_in_k)* D2(1/(1 + z_redshift))

    Psi_2_kx = (- 1j * kx/ (k_mag)**2  ) * Sum_k
    Psi_2_ky = (- 1j * ky/ (k_mag)**2  ) * Sum_k
    Psi_2_kz = (- 1j * kz/ (k_mag)**2  ) * Sum_k

    Psi_2_x = np.fft.ifftn(Psi_2_kx).real
    Psi_2_y = np.fft.ifftn(Psi_2_ky).real
    Psi_2_z = np.fft.ifftn(Psi_2_kz).real

    return Psi_2_x, Psi_2_y, Psi_2_z

""" Test af CIC For en partikel. """
n = 2

X_t = np.array([0])  # Er allerede i grid enheder. Da n = L vil konvertering i CIC funktionen ikke ændre noget. 
Y_t = np.array([0])
Z_t = np.array([0])

Psi_xt = np.array([1])
Psi_yt = np.array([0])
Psi_zt = np.array([0])

CIC_array_test = CIC(X_t, Y_t, Z_t, Psi_xt, Psi_yt, Psi_zt, n, n)
test_mass_conservation(X_t, CIC_array_test)
print(CIC_array_test)


#########################################

""" Partikler for hele boxen til LPT. (Simon) """
pos_vals = np.linspace(0, L, N, endpoint = False)
xs,ys,zs = np.meshgrid(pos_vals, pos_vals, pos_vals, indexing = 'ij')
xs = xs.flatten()
ys = ys.flatten()
zs = zs.flatten()





def Compute_universe(z_redshift, P_k=0, Delta = True):
    R = Random_R()

    if Delta == True:
        P_k = np.array([cosmo.pk(k_val, z_redshift) for k_val in k_mag.flatten()]).reshape(N, N, N)  # Behöver man ikke forloope over.
    
    delta_k = np.sqrt(P_k) * R * fac
    delta_k[0, 0, 0] = 0   # Lige tilføjet (Fra Jonny)

    """ 1 Ordens LPT """
    Psi_kx = (1j *kx)/(k_mag**2)  *delta_k
    Psi_ky = (1j *ky)/(k_mag**2)  *delta_k
    Psi_kz = (1j *kz)/(k_mag**2)  *delta_k

    Psi_x = np.fft.ifftn(Psi_kx).real
    Psi_y = np.fft.ifftn(Psi_ky).real
    Psi_z = np.fft.ifftn(Psi_kz).real

    """ 2 Ordens LPT """
    Psi_2_x, Psi_2_y, Psi_2_z = Second_LPT(Psi_kx, Psi_ky, Psi_kz, z_redshift=z_redshift)
    Psi_x_total = Psi_x + Psi_2_x
    Psi_y_total = Psi_y + Psi_2_y
    Psi_z_total = Psi_z + Psi_2_z

    if Delta == True:        # Her udføres forskydningsfeltet på partiklerne.
        """ Delta fra Fourie transform """
        delta_real = np.fft.ifftn(delta_k) 

        CIC_1LPT = CIC(xs, ys, zs, Psi_x, Psi_y, Psi_z, N, L)
        #test_mass_conservation(xs, CIC_1LPT)
        Delta_x_CIC1 = CIC_1LPT - 1 

        CIC_2LPT = CIC(xs, ys, zs, Psi_x_total, Psi_y_total, Psi_z_total, N, L)
        #test_mass_conservation(xs, CIC_2LPT)
        Delta_x_CIC2 = CIC_2LPT - 1  
        #return delta_real, Delta_x_CIC1, Delta_x_CIC2

        return delta_real, Delta_x_CIC1, Delta_x_CIC2

    else:
        Psi_tot_kx = np.fft.fftn(Psi_x_total)
        Psi_tot_ky = np.fft.fftn(Psi_y_total)
        Psi_tot_kz = np.fft.fftn(Psi_z_total)

        Psi1_vec_k = [Psi_kx, Psi_ky, Psi_kz]
        Psi2_vec_k = [Psi_tot_kx, Psi_tot_ky, Psi_tot_kz]

        return Psi1_vec_k, Psi2_vec_k


Nu_freq = (np.pi*N)/L    

def W_korrektion():
    W_k = (np.sinc(kx/ (2 * Nu_freq)) * np.sinc(ky / (2 * Nu_freq)) * np.sinc(kz/ (2 * Nu_freq))) ** 2
    W_k[W_k == 0] = 1
    return W_k

W_k = W_korrektion()

def P_k_binned(delta_X, CIC_corrected = False):
    delta_K = np.fft.fftn(delta_X)   # .real
    #P_k_vector = delta_K * np.conj(delta_K)  /(fac**2)                      # P_k i med 3D k værdiger.
    P_k_vector = abs(delta_K)**2  /(fac**2)
    
    #if (P_k_vector.real != P_k_vector).all():                   
    #    print("P_k er ikke reelt.")                             # Testen giver en små imaginær del forskelligt fra 0.
    
    if CIC_corrected == True:
        P_k_vector = P_k_vector/W_k**2  # Havd skal deles med window function?     # Korrektionen med Window funktionen, ved store k. 
    
    P_k_1D = P_k_vector.flatten()[valid_indices]                 # P_k fladet ud, da den kun afhænger af længden af k. Isotropi af P_k. 
    P_k_binned, _, _ = ss.stats.binned_statistic(k_mag_filtered, P_k_1D, statistic='mean', bins=k_bins)     # Middelværdigen af forskellige P_k med´samme k magnitude bestemmes.
    return P_k_binned




""" Bestemme P_k fra Class """
k_class = np.logspace(-2, 0, 100)
P_k_class = np.array([cosmo.pk(k, z_redshift) for k in k_class])   # 1D power spectrum fra Class


k_mag_flat = k_mag.flatten()                                        # Oprindelige k værdiger for boksen bliver til 1D.
valid_indices = k_mag_flat > 0                                      # Logaritmisk skalering i plot kræver positiv k
k_mag_filtered = k_mag_flat[valid_indices]

k_bins = np.logspace(-2, 0, 130)                                     # Binning til histogrammet
k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

Nu_freq = (np.pi*N)/L                                           # Nu_quest_frequens 

def Compute_Power(delta_real, Delta_x_CIC1, Delta_x_CIC2, ploting= False):
    
    P_k_FFT = P_k_binned(delta_real)
    P_k_CIC_1LPT = P_k_binned(Delta_x_CIC1, CIC_corrected = True)    
    P_k_CIC_2LPT = P_k_binned(Delta_x_CIC2, CIC_corrected = True) 

    if ploting == True:
        """ Kun plot af Power spectre """
        plt.figure(figsize=(8, 6))
        plt.plot(k_class, P_k_class, linestyle='-', color='k', label="CLASS (theory)")
        plt.loglog(k_bin_centers, P_k_FFT, marker='o', linestyle='none', label='FFT metode', alpha=0.4)
        plt.loglog(k_bin_centers, P_k_CIC_1LPT, marker='s', linestyle='none', label='1LPT med CIC', alpha=0.4)
        plt.loglog(k_bin_centers, P_k_CIC_2LPT, marker='o', linestyle='none', label='2LPT med CIC', alpha=0.3)
        plt.axvline(x= Nu_freq, color='r', linestyle='--', linewidth=0.5, label='Nu_quest_frequens')
        
        plt.xlabel('$k$ [h/Mpc]')
        plt.ylabel('$P(k) [Mpc/h] $')
        plt.title('Power Spectrum')
        plt.legend(loc='best')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.show()
    return P_k_FFT, P_k_CIC_1LPT, P_k_CIC_2LPT



delta_real, Delta_x_CIC1, Delta_x_CIC2 = Compute_universe(z_redshift, Delta = True)
P_k_FFT, P_k_CIC_1LPT, P_k_CIC_2LPT = Compute_Power(delta_real, Delta_x_CIC1, Delta_x_CIC2, ploting= True)






##################  Til shell-crossing #######################################

def Find_Psi_i_j(Psi_vec_k, k_vec):
    Psi_i_j = np.zeros((3, 3, N, N, N), dtype=np.float32)    # Tensor der bærer de afledte af forskydningsfeltet til en given (x,z,y) koordinat.

    for i in range(3):  # Ψ_i
        for j in range(3):  # ∂/∂q_j
            deriv_k = 1j * k_vec[j] * Psi_vec_k[i]
            deriv = np.fft.ifftn(deriv_k).real  # take real part
            Psi_i_j[i, j] = deriv
    return Psi_i_j


def Det_Jacobian(Psi_k, k_vec):
    Psi_i_j = Find_Psi_i_j(Psi_vec_k = Psi_k, k_vec=k_vec)

    Det_J = np.zeros((N, N, N), dtype=np.float32)

    for x in range(N):
        for y in range(N):
            for z in range(N):
                J_at_point = np.eye(3) + Psi_i_j[:, :, x, y, z]   # Svarer til:  J(x,y,z) = delta_ij + Psi_i,j
                Det_J[x, y, z] = np.linalg.det(J_at_point)
    return Det_J


# %%


P_k = np.array([cosmo.pk(k_val, z_redshift) for k_val in k_mag.flatten()]).reshape(N, N, N)
Psi1_vec_k, Psi2_vec_k = Compute_universe(z_redshift = z_redshift, Delta = False, P_k=P_k)
k_vec = [kx, ky, kz]




# %%
Num_of_mean = 10                  # Hvor mange univers der udregnes til at bestemme middelværdigen.
Number_points = 11
Redshifts = np.linspace(10, 15, Number_points)

Shell_crossing_1LPT_mean = []
Shell_crossing_2LPT_mean = []

Shell_crossing_1LPT_std = []
Shell_crossing_2LPT_std = []

for z in Redshifts:
    S_cros_1LPT = []
    S_cros_2LPT = []
    P_k = np.array([cosmo.pk(k_val, z) for k_val in k_mag.flatten()]).reshape(N, N, N)
    
    for i in np.arange(Num_of_mean):
        Psi1_vec_k, Psi2_vec_k = Compute_universe(z_redshift = z, P_k = P_k, Delta = False)

        J_1LPT = Det_Jacobian(Psi1_vec_k, k_vec)
        J_2LPT = Det_Jacobian(Psi2_vec_k, k_vec)

        n_shell_1LPT = np.sum(J_1LPT <= 0) #/N**3     # Kriteriet for shell-crossing det(J) <= 0
        n_shell_2LPT = np.sum(J_2LPT <= 0) #/N**3

        S_cros_1LPT.append(n_shell_1LPT)
        S_cros_2LPT.append(n_shell_2LPT)

    Middel_1LPT = np.mean(S_cros_1LPT)
    STD_1LPT = np.std(S_cros_1LPT)
    Middel_2LPT  = np.mean(S_cros_2LPT)
    STD_2LPT = np.std(S_cros_2LPT)

    print("-----------Redshift =" ,f"{z}", "------")
    print("Middel 1LPT: ", Middel_1LPT)
    print("STD 1LPT", STD_1LPT)

    print("Middel 2LPT", Middel_2LPT)
    print("STD 2LPT", STD_2LPT)
    
    
    Shell_crossing_1LPT_mean.append(Middel_1LPT)
    Shell_crossing_2LPT_mean.append(Middel_2LPT)
    Shell_crossing_1LPT_std.append(STD_1LPT)
    Shell_crossing_2LPT_std.append(STD_2LPT)

# %%


""" folder = r'C:\Users\aarom\OneDrive\Dokumente\Visual Studio\Batchorlor_projekt\Power_spectre\Shell-crossing'
filename = f"Number_of_shellcrosing_events_Mean_of={Num_of_mean}_Number_of_points={Number_points}_L={L}_N={N}_from10_to15.txt"
full_path = os.path.join(folder, filename)

# Stack the data column-wise
data = np.column_stack((Redshifts, Shell_crossing_1LPT_mean, 
                        Shell_crossing_1LPT_std, Shell_crossing_2LPT_mean,
                        Shell_crossing_2LPT_std))


header = "Redshift SC_1LPT_mean SC_1LPT_std SC_2LPT_mean SC_2LPT_std"

np.savetxt(full_path, data, delimiter=" ", fmt='%.10e', header=header, comments='')"""


plt.figure(figsize=(8, 5))
plt.errorbar(Redshifts[1:], Shell_crossing_1LPT_mean[1:], yerr= Shell_crossing_1LPT_std[1:], fmt='.',
            linewidth=2, capsize=6,  label='1LPT')
plt.errorbar(Redshifts[1:], Shell_crossing_2LPT_mean[1:], yerr= Shell_crossing_2LPT_std[1:], fmt='.',
            linewidth=2, capsize=6, label='2LPT')

plt.axvspan(11.5, 12.5, color='green', alpha=0.2, label='Første shell-crossing')

plt.xlabel('z')
#plt.ylabel(r'$n_{shelcrosing}/N^{3}$')
#plt.title(r'L = 1000, $N=64^3$')
plt.ylabel(r'Antal shell-crossing begivenheder')
plt.legend(loc='best')
plt.grid()


# %%


################### For at sample flere spektre #################################
""" 
for round in np.arange(200):
    delta_real, Delta_x_CIC1, Delta_x_CIC2 = Compute_universe(z_redshift, Delta = True)
    
    P_k_FFT, P_k_CIC_1LPT, P_k_CIC_2LPT = Compute_Power(delta_real, Delta_x_CIC1, Delta_x_CIC2, ploting= False)

    #file = r"\Strectrum.nr." + f"{round}"
    #np.savetxt(r'C:\Users\aarom\OneDrive\Dokumente\Visual Studio\Batchorlor_projekt\Power_spectre' 
    #           + file + ".txt", k_bin_centers, P_k_FFT, P_k_CIC_1LPT, P_k_CIC_2LPT, delimiter=" ", fmt='%s')

    folder = r'C:\Users\aarom\OneDrive\Dokumente\Visual Studio\Batchorlor_projekt\Power_spectre_med_positiv_D2\Redshift_z='+f"{z_redshift}"
    filename = f"Strectrum.nr.{round}.txt"
    full_path = os.path.join(folder, filename)

    # Stack the data column-wise
    data = np.column_stack((k_bin_centers, P_k_FFT, P_k_CIC_1LPT, P_k_CIC_2LPT))

    # Save to file
    np.savetxt(full_path, data, delimiter=" ", fmt='%s')
    #print("Round f"{round}" ") """






