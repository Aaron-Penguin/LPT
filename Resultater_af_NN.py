# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy as ss  
%matplotlib qt

plt.rc("axes", labelsize=18, titlesize=22)
plt.rc("xtick", labelsize=16, top=True, direction="in")
plt.rc("ytick", labelsize=16, right=True, direction="in")
plt.rc("legend", fontsize=16, loc="upper left")
plt.rcParams["font.size"] = "26"

data_prdictions_z0 = np.loadtxt(r'Predictions-z0-Aaron.txt', skiprows=1)
data_loss_z0 = np.load(r'loss-z0-Aaron.npy')
data_val_loss_z0 = np.load(r'val-loss-z0-Aaron.npy')

True_A = data_prdictions_z0[:,0]
Prdicted_A_s = data_prdictions_z0[:,1]
Error_predicted = data_prdictions_z0[:,2]


Epoch = np.linspace(1,1000, 1000)

# %%


fig, (ax1, ax2) =  plt.subplots(1,2, figsize=(12, 5))

ax2.set_title(r'$z= 0$')
ax2.set_xlabel(r'$ \mathcal{A}_{s, \text{sand}}$')
ax2.set_ylabel(r'$\mathcal{A}_{s, \text{forudset}}$')
ax2.errorbar(True_A, Prdicted_A_s, yerr=Error_predicted, color='C0' ,marker='.',linestyle='none', capsize=5)
ax2.plot([0, 3e-9], [0, 3e-9] ,color='k')
ax2.set_xlim([1.8e-9, 2.5e-9])
ax2.set_ylim([1.8e-9, 2.5e-9])
ax2.grid()

ax1.set_title(r'$z = 0$')
ax1.set_xlabel('Epochs')
#ax1.set_ylabel(r'$ ( A_{s, \text{sand}} - A_{s, forudset} )^2 $')
ax1.plot(Epoch[5:], data_loss_z0[5:], color='C1', label= r'Tab som MSE', alpha=0.5)
ax1.plot(Epoch[5:], data_val_loss_z0[5:], color='C2', label='Valideringstab', alpha=0.5)
ax1.legend(loc='best')
ax1.set_xlim([0,1000])
ax1.grid()
# %%
