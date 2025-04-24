import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from lattice import Lattice

# Test Square

np.set_printoptions(formatter={'float': lambda x: 'float: ' + str(x)})
np.set_printoptions(linewidth=np.inf)

t1 = -0.0001
t2 = -0.5

special_site_hops = [[((1, 0, 0), t1), ((0, 1, 0), t2)], [((0, 1, 0), t1), ((1, 0, 0), t2)]]
square = Lattice((50,50), 2, [(1, 0), (0, 1)], [(0, 0), (0, 0)], site_coordination=0, site_potential=[-1, -1],special_site_hops=special_site_hops)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')

surf1 = ax1.plot_trisurf(square.momentums[0], 
                        square.momentums[1], 
                        square.energies[0],
                        cmap='viridis', alpha=0.7, edgecolor='none')

surf2 = ax1.plot_trisurf(square.momentums[0], 
                        square.momentums[1], 
                        square.energies[1],
                        cmap='plasma', alpha=0.7, edgecolor='none')

target_energy = -1.2

ax2 = fig.add_subplot(122)

triang = Triangulation(square.momentums[0], square.momentums[1])

contour_params = {'levels': [target_energy], 'colors': ['#FF0000', '#0000FF'], 'linewidths': 2}

c1 = ax2.tricontour(triang, square.energies[0], **contour_params)
c2 = ax2.tricontour(triang, square.energies[1], **contour_params)

for collection in c1.collections:
    for path in collection.get_paths():
        vertices = path.vertices
        ax1.plot(vertices[:,0], vertices[:,1], target_energy, 
                color='red', linewidth=2, zorder=10)

for collection in c2.collections:
    for path in collection.get_paths():
        vertices = path.vertices
        ax1.plot(vertices[:,0], vertices[:,1], target_energy, 
                color='blue', linewidth=2, zorder=10)

ax1.set_title(f'3D Band Structure\n(Energy Slice at E = {target_energy:.2f})')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('Energy')

ax2.set_title(f'2D Fermi Surface at E = {target_energy:.2f}')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.grid(True)

plt.tight_layout()

dxy = -0.1
special_site_hops = [[((1, 0, 0), t1), ((0, 1, 0), t2), ((0, 0, 1), dxy)], [((0, 1, 0), t1), ((1, 0, 0), t2), ((0, 0, -1), dxy)]]
square2 = Lattice((50,50), 2, [(1, 0), (0, 1)], [(0, 0), (0, 0)], site_coordination=0, site_potential=[-1, -1],special_site_hops=special_site_hops)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')

surf1 = ax1.plot_trisurf(square2.momentums[0], 
                        square2.momentums[1], 
                        square2.energies[0],
                        cmap='viridis', alpha=0.7, edgecolor='none')

surf2 = ax1.plot_trisurf(square2.momentums[0], 
                        square2.momentums[1], 
                        square2.energies[1],
                        cmap='plasma', alpha=0.7, edgecolor='none')

target_energy = -1.2

ax2 = fig.add_subplot(122)

triang = Triangulation(square2.momentums[0], square2.momentums[1])

contour_params = {'levels': [target_energy], 'colors': ['#FF0000', '#0000FF'], 'linewidths': 2}

c1 = ax2.tricontour(triang, square2.energies[0], **contour_params)
c2 = ax2.tricontour(triang, square2.energies[1], **contour_params)

for collection in c1.collections:
    for path in collection.get_paths():
        vertices = path.vertices
        ax1.plot(vertices[:,0], vertices[:,1], target_energy, 
                color='red', linewidth=2, zorder=10)

for collection in c2.collections:
    for path in collection.get_paths():
        vertices = path.vertices
        ax1.plot(vertices[:,0], vertices[:,1], target_energy, 
                color='blue', linewidth=2, zorder=10)

ax1.set_title(f'3D Band Structure\n(Energy Slice at E = {target_energy:.2f})')
ax1.set_xlabel('kx')
ax1.set_ylabel('ky')
ax1.set_zlabel('Energy')

ax2.set_title(f'2D Fermi Surface at E = {target_energy:.2f}')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.grid(True)

plt.tight_layout()

plt.show()