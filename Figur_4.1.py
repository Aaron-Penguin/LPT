# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy as ss  
%matplotlib qt

C = ss.stats.norm.rvs(loc= 0, scale= 1/np.sqrt(2), size=(32)).reshape(2,4,4)

print(C)

# %%
for n in np.arange(2):
    for i in np.arange(4):
        for j in np.arange(4):
            if C[n,i,j] > 0:
                C[n,i,j] = (C[n,i,j] % 1)/5
            else:
                C[n,i,j] =  - (C[n,i,j] % 1 )/5

print(C)

# %%


plt.figure( figsize=(10,4))
plt.axis('off')

for i in range(5):
    plt.plot([i, i], [0, 4], color='k', linewidth=3)
    plt.plot([0, 4], [i, i], color='k', linewidth=3)

for i in range(4):
    for j in range(4):
        plt.scatter(i +0.5 , j+0.5, color='C0', linewidths= 4)



# Box efter forskud siktse
x0 = 6

for i in range(5):
    plt.plot([i + x0, i + x0], [0, 4], color='k', linewidth=3)
    plt.plot([x0, 4 + x0], [i, i], color='k', linewidth=3)

for i in range(4):
    for j in range(4):
        plt.scatter(i +0.5 +x0 + C[0][i][j], j+0.5 + C[1][i][j], color='C0', linewidths= 4)

plt.arrow(4.3, 2, 1, 0, head_width=0.3, linewidth= 3, head_length=0.4, fc='k', ec='k')

# %%
