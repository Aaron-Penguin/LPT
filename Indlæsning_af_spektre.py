# %%
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib qt
from classy import Class

plt.rc("axes", labelsize=16, titlesize=22)
plt.rc("xtick", labelsize=18, top=True, direction="in")
plt.rc("ytick", labelsize=18, right=True, direction="in")
plt.rc("legend", fontsize=16, loc="upper left")
plt.rcParams["font.size"] = "26"


Redshift = 0


folder = r'C:\Users\aarom\OneDrive\Dokumente\Visual Studio\Batchorlor_projekt\Power_spectre'
Folder_i_folder = f"Redshift_z_{Redshift}"


folder_path = os.path.join(folder, Folder_i_folder)

files = os.listdir(folder_path)
file_name = [f for f in files]

N = np.array(file_name).size


#--------- k værdigerne er de samme i alle filer --------------------
file_0 = os.path.join(folder_path, file_name[0])
dat_0 = np.loadtxt(file_0)
k = dat_0[:, 0]
Class_R_0 = dat_0[:, 1]
LPT1_0 = dat_0[:,2]
LPT2_0 =dat_0[:,3]

N_elements = k.size

# %%

LPT1_mean = []
LPT1_std = []

LPT2_mean = []
LPT2_std = []

Random_class_mean = []
Random_class_std = []

for j in np.arange(N_elements):
    List_LPT1 = []
    List_LPT2 = []
    List_random = []
    for i in np.arange(N):
        file_n = os.path.join(folder_path, file_name[i])

        Power_Spektre = np.loadtxt(file_n)
        Random_class = Power_Spektre[j][1]
        LPT1 = Power_Spektre[j][2]
        LPT2 = Power_Spektre[j][3]
        
        List_LPT1.append(LPT1)
        List_LPT2.append(LPT2)
        List_random.append(Random_class)

    LPT1_m = np.mean(List_LPT1)
    LPT1_s = np.std(List_LPT1)

    LPT2_m = np.mean(List_LPT2)
    LPT2_s = np.std(List_LPT2)

    R_m = np.mean(List_random)
    R_s = np.std(List_random)

    LPT1_mean.append(LPT1_m)
    LPT1_std.append(LPT1_s)
    LPT2_mean.append(LPT2_m)
    LPT2_std.append(LPT2_s)
    Random_class_mean.append(R_m)
    Random_class_std.append(R_s)



# %%

N = 64
L = 500
Nu_freq = (np.pi*N)/L  

params = {
        'h': 0.6736,  # Hubble parameter
        'omega_b': 0.02237,  # Baryon density
        'omega_cdm': 0.1200,  # Cold dark matter density
        'A_s': 2.1e-9,    #10e-10,  # Scalar amplitude
        'n_s': 0.9649,  # Spectral index
        'output': 'mPk',  # Request matter power spectrum
        'P_k_max_1/Mpc': 100,  # Maximum k value in 1/Mpc
        'z_max_pk': 100,  # Ensures CLASS computes P(k) for requested redshifts
        'z_pk': Redshift,
    }

cosmo = Class()
cosmo.set(params)
cosmo.compute()

P_k = [cosmo.pk(k_val, Redshift) for k_val in k]



#folder = r'C:\Users\aarom\OneDrive\Dokumente\Visual Studio\Batchorlor_projekt\Power_spectre\Power_spektre_midlet'
#filename = f"Redshift.{Redshift}.txt"
#full_path = os.path.join(folder, filename)

# Stack the data column-wise
#data = np.column_stack((k, LPT1_mean, LPT1_std, LPT2_mean, LPT2_std))

# Save to file
#np.savetxt(full_path, data, delimiter=" ", fmt='%s')


# %%

plt.rc("axes", labelsize=16, titlesize=22)
plt.rc("xtick", labelsize=18, top=True, direction="in")
plt.rc("ytick", labelsize=18, right=True, direction="in")
plt.rc("legend", fontsize=16, loc="upper left")
plt.rcParams["font.size"] = "26"





fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12, 5.5))
ax1.set_title(r'$z = 0$')
ax1.set_yscale("log")
ax1.set_xscale("log")

ax1.plot(k, P_k, '-', color='k', label='Class')

ax1.scatter(k, Class_R_0, marker='.', color='C2', label='Class med tilfældigt R')

ax1.scatter(k, LPT1_0, color='C0', marker='.' , label='1LPT')
ax1.scatter(k, LPT2_0, color='C1' ,marker='.', label='2LPT')

ax1.set_xlabel(r'$k\ \left[\mathrm{Mpc}^{-1}\right]$', fontsize=14)
ax1.set_ylabel(r'$P(k)\ \left[\mathrm{Mpc}^3\right]$', fontsize=14)
ax1.axvline(x= Nu_freq, color='r', linestyle='--', linewidth=1, label= r'$k_{Ny}$')

ax1.set_ylim([50, 200000])
ax1.grid(True, which='both', ls='--', alpha=0.7)
ax1.legend(loc='best')


ax2.set_title(r'$z = 0$')
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.plot(k, np.array(P_k) , '-', color='k', label='Class')
ax2.errorbar(k, np.array(Random_class_mean), yerr=Random_class_std, linestyle='none', marker='.', color='C2', capsize=5, label='Class med tilfældigt R')


ax2.errorbar(k, np.array(LPT1_mean), yerr= LPT1_std, color='C0' ,marker='.',linestyle='none', capsize=5, label='1LPT')
ax2.errorbar(k, np.array(LPT2_mean), yerr= LPT2_std, color='C1' ,marker='.',linestyle='none', capsize=5, label='2LPT')

ax2.axvline(x= Nu_freq, color='r', linestyle='--', linewidth=1, label= r'$k_{Ny}$')
ax2.set_xlabel(r'$k\ \left[ \mathrm{Mpc}^{-1} \right]$', fontsize=14)
ax2.set_ylabel(r'$P(k)\ \left[\mathrm{Mpc}^3\right]$', fontsize=14)
ax2.set_ylim([50, 200000])
ax2.grid(True, which='both', ls='--', alpha=0.7)


#######################  Genindlæs for figur 6.4 ######################

plt.rc("axes", labelsize=14, titlesize=22)
plt.rc("xtick", labelsize=14, top=True, direction="in")
plt.rc("ytick", labelsize=14, right=True, direction="in")
plt.rc("legend", fontsize=13, loc="upper left")
plt.rcParams["font.size"] = "26"


folder = r'C:\Users\aarom\OneDrive\Dokumente\Visual Studio\Batchorlor_projekt\Power_spectre\Power_spektre_midlet'

full_path_z1 = os.path.join(folder, f"Redshift.1.txt")
full_path_z12 = os.path.join(folder, f"Redshift.12.txt")

data_z1 = np.loadtxt(full_path_z1)
data_z12 = np.loadtxt(full_path_z12)

# %%

k = data_z1[:,0]

 
LPT1_mean_z1 = data_z1[:,1]
LPT1_std_z1 = data_z1[:,2]
LPT2_mean_z1 = data_z1[:,3]
LPT2_std_z1 = data_z1[:,4]

LPT1_mean_z12 = data_z12[:,1]
LPT1_std_z12 = data_z12[:,2]
LPT2_mean_z12 = data_z12[:,3]
LPT2_std_z12 = data_z12[:,4]
# %%
k_class = np.logspace(-2, 0, 200)


P_k_z1 = [cosmo.pk(k_val, 1) for k_val in k_class]
P_k_z12 = [cosmo.pk(k_val, 12) for k_val in k_class]


fig, axes = plt.subplots(2, 1, figsize=(12, 11))
ax1, ax2 = axes.flatten()

fig.subplots_adjust(hspace=0.3) 

ax1.set_title(r'$z = 1$')
ax1.set_yscale("log")
ax1.set_xscale("log")

ax1.plot(k_class, P_k_z1, '-', color='k', label='Class')

ax1.errorbar(k, LPT1_mean_z1, yerr= LPT1_std_z1, color='C0' ,marker='.',linestyle='none', capsize=5, label='1LPT')
ax1.errorbar(k, LPT2_mean_z1, yerr= LPT2_std_z1, color='C1' ,marker='.',linestyle='none', capsize=5, label='2LPT')

ax1.axvline(x= Nu_freq, color='r', linestyle='--', linewidth=1, label= r'$k_{Ny}$')
ax1.set_xlabel(r'$k\ \left[ \mathrm{Mpc}^{-1} \right]$', fontsize=14)
ax1.set_ylabel(r'$P(k)\ \left[\mathrm{Mpc}^3\right]$', fontsize=14)

ax1.set_ylim([100, 60000])
ax1.grid(True, which='both', ls='--', alpha=0.7)
ax1.legend(loc='best')

####### Plot nr. 2 #########################

ax2.set_title(r'$z = 12$')
ax2.set_yscale("log")
ax2.set_xscale("log")

ax2.plot(k_class, P_k_z12, '-', color='k', label='Class')

ax2.errorbar(k, LPT1_mean_z12, yerr= LPT1_std_z12, color='C0' ,marker='.',linestyle='none', capsize=5, label='1LPT')
ax2.errorbar(k, LPT2_mean_z12, yerr= LPT2_std_z12, color='C1' ,marker='.',linestyle='none', capsize=5, label='2LPT')

ax2.axvline(x= Nu_freq, color='r', linestyle='--', linewidth=1, label= r'$k_{Ny}$')
ax2.set_xlabel(r'$k\ \left[ \mathrm{Mpc}^{-1} \right]$', fontsize=14)
ax2.set_ylabel(r'$P(k)\ \left[\mathrm{Mpc}^3\right]$', fontsize=14)

ax2.set_ylim([5, 2000])
ax2.grid(True, which='both', ls='--', alpha=0.7)
