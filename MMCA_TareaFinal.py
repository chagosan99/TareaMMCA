import numpy as np
import matplotlib.pyplot as plt 

# Constante gravitacional
ga = 10

# Dimensiones de la presa
d = 0.1
H = 10.

npx = int(round(H/d) + 1)
X = np.zeros([npx, npx], dtype=np.float)
Y = np.zeros([npx, npx], dtype=np.float)
sigmx = np.zeros([npx, npx], dtype=np.float)
sigmy = np.zeros([npx, npx], dtype=np.float)
tauxy = np.zeros([npx, npx], dtype=np.float)
taumax = np.zeros([npx, npx], dtype=np.float)

vx = np.linspace(0, H, npx)
vy = vx

corx = 0.
cory = 0.

# Crear malla de coordenadas
for ix in range(npx):
    for iy in range(npx):
        X[ix, iy] = corx
        Y[ix, iy] = cory
        if cory == corx:
            break
        cory = cory + d
    corx = corx + d
    cory = 0.

# Calcular esfuerzos

for ix in range(npx):
    for iy in range(npx):
        corx=X[ix,iy]
        cory=Y[ix,iy]
        sigmx[ix,iy]= corx*ga-cory*ga*2.
        sigmy[ix,iy]= -corx*ga
        tauxy[ix,iy]= -cory*ga
        
sig1 = (sigmx + sigmy)/2 + (((sigmx - sigmy)/2)**2 + tauxy**2)**0.5
sig2 = (sigmx + sigmy)/2 - (((sigmx - sigmy)/2)**2 + tauxy**2)**0.5
taumax = (sig1 - sig2) / 2


# Graficar esfuerzos

plt.figure()
CS = plt.contour(X, Y, sigmx, 20)
plt.clabel(CS, inline=1, fontsize=10)
plt.plot(vx, vy, linewidth=2.0)
plt.title('Sigma x')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.savefig('sigmax.pdf')

plt.figure()
CS = plt.contour(X, Y, sigmy, 20)
plt.clabel(CS, inline=1, fontsize=10)
plt.plot(vx, vy, linewidth=2.0)
plt.title('Sigma y')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.savefig('sigmay.pdf')

# Graficar esfuerzos Sigma 1

fig, ax = plt.subplots()
CS = ax.contour(X, Y, sig1, 20)
ax.clabel(CS, inline=1, fontsize=10)
cp = ax.contourf(X, Y, sig1, 20, cmap='coolwarm')
ax.set_aspect('equal')
fig.colorbar(cp) 
ax.plot(vx, vy, linewidth=2.0)
ax.set_title('Sigma 1')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')

# Agregar etiqueta de máximo esfuerzo a tensión
i, j = np.unravel_index(np.argmax(sig1), sig1.shape)
max_sig1 = np.max(sig1)
textstr = f'Max sig: {max_sig1:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(X[i,j], Y[i,j], textstr, fontsize=10, verticalalignment='top', bbox=props)

plt.savefig('sig_1.pdf')
plt.show()

# Graficar esfuerzos Sigma 2

fig, ax = plt.subplots()
CS = ax.contour(X, Y, sig2, 20)
ax.clabel(CS, inline=1, fontsize=10)
cp = ax.contourf(X, Y, sig2, 20, cmap='coolwarm')
ax.set_aspect('equal')
fig.colorbar(cp) 
ax.plot(vx, vy, linewidth=2.0)
ax.set_title('Sigma 2')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')

# Agregar etiqueta de máximo esfuerzo a compresión
i, j = np.unravel_index(np.argmin(sig2), sig2.shape)
min_sig2 = np.min(sig2)
textstr = f'Max sig: {min_sig2:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(X[i,j], Y[i,j], textstr, fontsize=10, verticalalignment='top', bbox=props)

plt.savefig('sig_2.pdf')
plt.show()

# Graficar esfuerzos a Máxima Cortante

fig, ax = plt.subplots()
CS = ax.contour(X, Y, taumax, 20)
ax.clabel(CS, inline=1, fontsize=10)
cp = ax.contourf(X, Y, taumax, 20, cmap='coolwarm')
ax.set_aspect('equal')
fig.colorbar(cp) 
ax.plot(vx, vy, linewidth=2.0)
ax.set_title('Tao Max')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')

# Agregar etiqueta de máximo esfuerzo a Cortante
i, j = np.unravel_index(np.argmin(taumax), taumax.shape)
max_tau = np.max(taumax)
textstr = f'Max tau: {max_tau:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(X[i,j], Y[i,j], textstr, fontsize=10, verticalalignment='top', bbox=props)

plt.savefig('taumax.pdf')
plt.show()

#Zonas diferenciadas a compresión y a tracción

# Graficar zonas de compresión y tracción en dirección x
fig, ax = plt.subplots()
compresion_y = np.where(sigmx < 0, sigmx, np.nan)
traccion_y = np.where(sigmx >= 0, sigmx, np.nan)
ax.contourf(X, Y, compresion_y, levels=[-np.inf, 0], colors='blue')
ax.contourf(X, Y, traccion_y, levels=[0, np.inf], colors='red')
ax.plot(vx, vy, linewidth=2.0)
ax.set_title('Zonas de Compresión y Tracción en Dirección X')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
plt.savefig('zonas_x.pdf')
plt.show()

# Graficar zonas de compresión y tracción en dirección y
fig, ax = plt.subplots()
compresion_y = np.where(sigmy < 0, sigmy, np.nan)
traccion_y = np.where(sigmy >= 0, sigmy, np.nan)
ax.contourf(X, Y, compresion_y, levels=[-np.inf, 0], colors='blue')
ax.contourf(X, Y, traccion_y, levels=[0, np.inf], colors='red')
ax.plot(vx, vy, linewidth=2.0)
ax.set_title('Zonas de Compresión y Tracción en Dirección Y')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
plt.savefig('zonas_y.pdf')
plt.show()
