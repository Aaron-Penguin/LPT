# %%
import numpy as np
import scipy as ss                 
import matplotlib.pyplot as plt
%matplotlib qt


plt.rc("axes", labelsize=20, titlesize=22)
plt.rc("xtick", labelsize=20, top=True, direction="in")
plt.rc("ytick", labelsize=20, right=True, direction="in")
plt.rc("legend", fontsize=18, loc="upper left")
plt.rcParams["font.size"] = "26"


# Grid and Fourier setup
N = 120     # Valg af boxsens størrelse påvirker precition af fourie transform
L = 10
x = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

dx = x[1] - x[0]


k_val = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
kx, ky, kz = np.meshgrid(k_val, k_val, k_val, indexing='ij')

k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
k_mag[0, 0, 0] = 1e-10  # avoid division by zero



#######################  2LPT functionen ######################################

def Second_LPT(Psi_kx, Psi_ky, Psi_kz):   

    Psi_k_vek = [Psi_kx, Psi_ky, Psi_kz]
    k_vec = [kx, ky, kz]

    Sum_k  = np.zeros((N,N,N), dtype = 'complex128')


    for i in range(3):
        for j in range(3):
            if i > j:
                psi1_i_i = 1j * k_vec[i] * Psi_k_vek[i]  
                psi1_j_j = 1j * k_vec[j] * Psi_k_vek[j] 
                psi1_i_j = 1j * k_vec[j] * Psi_k_vek[i]      # Fourier derivative of psi1 along j
                psi1_j_i = 1j * k_vec[i] * Psi_k_vek[j]  
                

                term1 = np.fft.ifftn(psi1_i_i) * np.fft.ifftn(psi1_j_j)   # Foldning i k rum. Multiplikation i real rum.
                term2 = np.fft.ifftn(psi1_i_j) * np.fft.ifftn(psi1_j_i)

                term1_in_k = np.fft.fftn(term1)
                term2_in_k = np.fft.fftn(term2)
                
                Sum_k += (term1_in_k - term2_in_k)



    Psi_2_kx = (- 1j * kx/ (k_mag)**2  ) * Sum_k * D2
    Psi_2_ky = (- 1j * ky/ (k_mag)**2  ) * Sum_k * D2
    Psi_2_kz = (- 1j * kz/ (k_mag)**2  ) * Sum_k * D2

    Psi_2_x = np.fft.ifftn(Psi_2_kx).real
    Psi_2_y = np.fft.ifftn(Psi_2_ky).real
    Psi_2_z = np.fft.ifftn(Psi_2_kz).real

    return Psi_2_x, Psi_2_y, Psi_2_z

###############################################################################

# Test funktion: Gauss potential

c = L/2
r2 = (X - c)**2 + (Y - c)**2 + (Z - c)**2
phi = np.exp(-r2 / (2))                  # Potential i real rum. Test funktionen


phi_k = np.fft.fftn(phi) 

# 1st order displacement (Numerisk): Ψ₁ = -∇φ
Psi_kx = - 1j * kx * phi_k
Psi_ky = - 1j * ky * phi_k
Psi_kz = - 1j * kz * phi_k

Psi_x = np.fft.ifftn(Psi_kx).real 
Psi_y = np.fft.ifftn(Psi_ky).real  
Psi_z = np.fft.ifftn(Psi_kz).real  


################# Analytisk Psi 1 ##############################
Psi_x_Analytic = (X - c) *phi    #  Ψ₁ = -∇φ
Psi_y_Analytic = (Y - c) *phi
Psi_z_Analytic = (Z - c) *phi

Diff_psi_1 = Psi_x - Psi_x_Analytic

slice_idx = N//2 
vmin = np.min(Psi_x[:, :, slice_idx])
vmax = np.max(Psi_x[:, :, slice_idx])


# Plot for at teste Psi_1
def plot_slice(data, title, cmap='viridis'):  
    plt.imshow(data[:, :, slice_idx],
               extent= [0, L ,0, L] ,                              
               origin='lower',
               cmap=cmap,
               vmin= np.min(data),
               vmax= np.max(data))
    plt.title(title)
    plt.colorbar()
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_2$')
    plt.tight_layout()
    plt.grid(False)


# %%

Nabla_psi2_k = np.zeros((N,N,N), dtype = 'complex128')  #, dtype=np.complex128)

var = [X, Y, Z]

for i in np.arange(3):
    for j in np.arange(3):
        if i > j:
            Psi_i_i = phi -  ((var[i] - c)**2) * phi
            Psi_j_j = phi -  ((var[j] - c)**2) * phi
            Psi_i_j = - (var[i] - c)*(var[j] - c) *phi
            Psi_j_i = - (var[i] - c)*(var[j] - c) *phi

            Term1 = Psi_i_i * Psi_j_j
            Term2 = Psi_i_j * Psi_j_i

            Term1_k = np.fft.fftn(Term1)
            Term2_k = np.fft.fftn(Term2)

            Nabla_psi2_k += (Term1_k - Term2_k)  #



D2 = -3/7   # Kun for at teste

Psi_2_kx_a = (- 1j * kx/ (k_mag)**2  ) * Nabla_psi2_k * D2
Psi_2_ky_a = (- 1j * ky/ (k_mag)**2  ) * Nabla_psi2_k * D2
Psi_2_kz_a = (- 1j * kz/ (k_mag)**2  ) * Nabla_psi2_k * D2


Psi_2_x_A = np.fft.ifftn(Psi_2_kx_a).real #*1/(dx**3)
Psi_2_y_A = np.fft.ifftn(Psi_2_ky_a).real #*1/(dx**3)
Psi_2_z_A = np.fft.ifftn(Psi_2_kz_a).real #*1/(dx**3)

# %%
###################################################################################
Psi2_x, Psi2_y, Psi2_z = Second_LPT(Psi_kx, Psi_ky, Psi_kz)



Psi_2_x_A = Psi_2_x_A

Afvigelse_psi2_x = Psi2_x - Psi_2_x_A 
print("Max afvigelse i x for 2LPT =", np.max(Afvigelse_psi2_x))


plt.figure(figsize=(15, 4))

plt.subplot(1,3,1)
plot_slice(Psi2_x, r'$\vec{\Psi}^{(2)}_{q_1, Num}$')

plt.subplot(1,3,2)
plot_slice(Psi_2_x_A, r'$\vec{\Psi}^{(2)}_{q_1, A}$')

plt.subplot(1,3,3)
plot_slice(Afvigelse_psi2_x, r'$\vec{\Psi}^{(2)}_{q_1, Num} - \vec{\Psi}^{(2)}_{q_1, A}$') 

