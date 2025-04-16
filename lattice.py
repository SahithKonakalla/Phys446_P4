import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.linalg
import time

class Lattice():
    """ def __init__(self, Nshape, sites, lattice_vectors=1, sublattice_vectors=0, site_coordination=-1, site_potential=0, special_hops=[]):
        self.Nshape = Nshape
        if isinstance(self.Nshape, tuple):
            self.dims = len(self.Nshape)
        else:
            self.dims  = 1
        self.sites = sites
        self.lattice_vectors = lattice_vectors
        self.sublattice_vectors = sublattice_vectors

        self.reciprocal_vectors = self.ReciprocalVectors()

        self.labels = self.Labels()
        self.mlabels = self.MomentumLabels()

        self.bounds = self.GetBounds()
        self.neighbor_vectors = self.NeighborVectors()

        self.matrix_dim = len(self.labels)

        self.site_coordination = site_coordination
        self.site_potential = site_potential
        if site_potential == 0 and self.sites > 1:
            self.site_potential = [0 for i in range(self.sites)]
        
        self.special_hops = special_hops

        self.H = self.generateHamiltonian()
        self.F = self.generateFourier()
        self.Hk = self.F.getH() @ self.H @ self.F

        self.energies = self.getEnergyBands()
        self.momentums = 0
        if self.dims > 1:
            self.momentums = []
            mom = np.array([self.MomentumLabelToK(self.mlabels[i]) for i in range(len(self.H))])[::self.sites]
            for dim in range(self.dims):
                self.momentums.append(mom[:, dim])
            self.momentums = np.array(self.momentums)
        else:
            self.momentums = np.array([self.MomentumLabelToK(self.mlabels[i]) for i in range(len(self.H))])[::self.sites]
        self.fermi_level = self.getFermiLevel() """ 

    def __init__(self, Nshape, sites, lattice_vectors=1, sublattice_vectors=0, site_coordination=-1, site_potential=0, special_hops=[]):
        self.Nshape = Nshape
        if isinstance(self.Nshape, tuple):
            self.dims = len(self.Nshape)
        else:
            self.dims = 1
        self.sites = sites
        self.lattice_vectors = lattice_vectors
        self.sublattice_vectors = sublattice_vectors

        # Time measurement for ReciprocalVectors
        start_time = time.time()
        self.reciprocal_vectors = self.ReciprocalVectors()
        print(f"ReciprocalVectors took {time.time() - start_time:.4f} seconds")

        # Time measurement for Labels
        start_time = time.time()
        self.labels = self.Labels()
        print(f"Labels took {time.time() - start_time:.4f} seconds")

        # Time measurement for MomentumLabels
        start_time = time.time()
        self.mlabels = self.MomentumLabels()
        print(f"MomentumLabels took {time.time() - start_time:.4f} seconds")

        # Time measurement for GetBounds
        start_time = time.time()
        self.bounds = self.GetBounds()
        print(f"GetBounds took {time.time() - start_time:.4f} seconds")

        # Time measurement for NeighborVectors
        start_time = time.time()
        self.neighbor_vectors = self.NeighborVectors()
        print(f"NeighborVectors took {time.time() - start_time:.4f} seconds")

        self.matrix_dim = len(self.labels)

        self.site_coordination = site_coordination
        self.site_potential = site_potential
        if site_potential == 0 and self.sites > 1:
            self.site_potential = [0 for i in range(self.sites)]
        
        self.special_hops = special_hops

        # Time measurement for generateHamiltonian
        start_time = time.time()
        self.H = self.generateHamiltonian()
        print(f"generateHamiltonian took {time.time() - start_time:.4f} seconds")

        # Time measurement for generateFourier
        start_time = time.time()
        self.F = self.generateFourier()
        print(f"generateFourier took {time.time() - start_time:.4f} seconds")

        # Time measurement for Hk calculation
        start_time = time.time()
        #self.Hk = self.F.getH() @ self.H @ self.F
        self.Hk = self.getHk()
        print(f"Hk calculation took {time.time() - start_time:.4f} seconds")

        # Time measurement for getEnergyBands
        start_time = time.time()
        self.energies = self.getEnergyBands()
        print(f"getEnergyBands took {time.time() - start_time:.4f} seconds")

        self.momentums = 0
        if self.dims > 1:
            # Time measurement for momentum calculations in higher dimensions
            start_time = time.time()
            self.momentums = []
            mom = np.array([self.MomentumLabelToK(self.mlabels[i]) for i in range(len(self.H))])[::self.sites]
            for dim in range(self.dims):
                self.momentums.append(mom[:, dim])
            self.momentums = np.array(self.momentums)
            print(f"Momentum calculations (dims>1) took {time.time() - start_time:.4f} seconds")
        else:
            # Time measurement for momentum calculation in 1D
            start_time = time.time()
            self.momentums = np.array([self.MomentumLabelToK(self.mlabels[i]) for i in range(len(self.H))])[::self.sites]
            print(f"Momentum calculation (1D) took {time.time() - start_time:.4f} seconds")

        # Time measurement for getFermiLevel
        start_time = time.time()
        self.fermi_level = self.getFermiLevel()
        print(f"getFermiLevel took {time.time() - start_time:.4f} seconds")
    
    def Labels(self):
        labels = []

        max_index = 1
        if self.dims > 1:
            for n in self.Nshape:
                max_index *= n
        
            max_index *= self.sites
        else:
            max_index = self.Nshape*self.sites

        for i in range(max_index):
            site = i % self.sites

            j = i // self.sites

            if self.dims > 1:
                pos = []
                for dim in self.Nshape:
                    pos.append(j % dim + 1)
                    j =  j // dim
                
                labels.append((tuple(pos[::-1]), chr(ord('a')+site)))
            else:
                labels.append((j % self.Nshape + 1, chr(ord('a')+site)))

        
        return labels
    
    def LabelToR(self, label):
        pos = np.zeros(self.dims)
        if self.dims > 1:
            for dim in range(self.dims):
                pos = pos + (label[0][dim]-1)*np.array(self.lattice_vectors[dim])
        else:
            pos = pos + (label[0]-1)*self.lattice_vectors

        j = ord(label[1]) - ord('a')

        if self.sites > 1:
            pos = pos + self.sublattice_vectors[j]
        else:
            pos = pos + self.sublattice_vectors

        if self.dims > 1:
            return tuple(pos)
        else:
            return pos[0]
    
    def LabelToIndex(self, label):
        index = 0
        if self.dims > 1:
            for dim in range(self.dims):
                mult = self.sites
                for i in range(dim+1, self.dims):
                    mult *= self.Nshape[i]
                index += (label[0][dim]-1)*mult
        else:
            index += (label[0]-1)*self.sites

        index += ord(label[1]) - ord('a')

        return index
    
    def ReciprocalVectors(self):
        if self.dims > 1:
            A = np.array(self.lattice_vectors)
            B = 2 * np.pi * np.linalg.pinv(A).T
            return [tuple(B[dim]/self.Nshape[dim]) for dim in range(self.dims)]
        else:
            return 2 * np.pi / (self.Nshape * self.lattice_vectors)

    def MomentumLabels(self):
        labels = []

        max_index = 1
        if self.dims > 1:
            for n in self.Nshape:
                max_index *= n
        
            max_index *= self.sites
        else:
            max_index = self.Nshape*self.sites

        for i in range(max_index):
            site = i % self.sites

            j = i // self.sites

            if self.dims > 1:
                pos = []
                for dim in self.Nshape:
                    pos.append(j % dim)
                    j =  j // dim
                
                labels.append((tuple(pos[::-1]), chr(ord('a')+site)))
            else:
                labels.append((j % self.Nshape, chr(ord('a')+site)))

        return labels

    def MomentumLabelToK(self, label):
        pos = np.zeros(self.dims)
        if self.dims > 1:
            for dim in range(self.dims):
                pos = pos + (label[0][dim])*np.array(self.reciprocal_vectors[dim])
            return tuple(pos)
        else:
            return label[0]*self.reciprocal_vectors
            

    def getNeighbor(self, label, vector):
        pos = []
        if self.dims > 1:
            for dim in range(self.dims):
                pos.append((label[0][dim] + vector[dim] - 1) % self.Nshape[dim] + 1)
        else:
            pos = (label[0] + vector[0] - 1) % self.Nshape + 1

        site = chr((ord(label[1]) - ord("a") + vector[-1]) % self.sites + ord("a"))

        if self.dims > 1:
            return (tuple(pos), site)
        else:
            return (pos, site)

    def GetBounds(self):
        if self.dims > 1:
            bounds = [0 for dim in range(self.dims)]
            for dim in range(self.dims):
                for vector in self.lattice_vectors:
                    bounds[dim] = max(bounds[dim], vector[dim]*self.Nshape[dim])
            return bounds
        else:
            return self.lattice_vectors * self.Nshape
    
    def getDistance(self, label1, label2):
        if self.dims > 1:
            dist = np.array(self.LabelToR(label2)) - np.array(self.LabelToR(label1))
            for dim in range(self.dims):
                if label2[0][dim] - label1[0][dim] > self.Nshape[dim]/2:
                    dist = dist - np.array(self.lattice_vectors[dim])*self.Nshape[dim]
            return np.linalg.norm(dist)   
        else:
            # Manhattan Distance
            dist = abs(self.LabelToR(label2) - self.LabelToR(label1))
            if label2[0] - label1[0] > self.Nshape/2:
                dist = dist - np.array(self.lattice_vectors)*self.Nshape
            return abs(dist)
            
    def getVector(self, label1, label2):
        vector = []
        if self.dims > 1:
            for dim in range(self.dims):
                diff = label2[0][dim] - label1[0][dim]
                if abs(diff) > self.Nshape[dim]/2:
                    vector.append(np.round(diff - self.Nshape[dim], 5))
                else:
                    vector.append(diff)
        else:
            diff = label2[0] - label1[0]
            if abs(diff) > self.Nshape/2:
                vector.append(np.round(diff - self.Nshape, 5))
            else:
                vector.append(diff)
        
        cdiff = ord(label2[1]) - ord(label1[1])
        if abs(cdiff) > self.sites/2:
            vector.append(self.sites - cdiff)
        else:
            vector.append(cdiff)
        
        return tuple(vector)
    
    def NeighborVectors(self):
        neighbor_vectors = []
        
        for site in range(self.sites):
            dist_dict = {}
            label1 = (self.labels[0][0], chr(site + ord("a")))
            for label2 in self.labels:
                dist = np.round(self.getDistance(label1, label2), 5)
                if dist == 0:
                    continue
                vector = self.getVector(label1, label2)
                if dist in dist_dict:
                    dist_dict[dist].append(vector)
                else:
                    dist_dict[dist] = [vector]
            
            sorted_dict = dict(sorted(dist_dict.items()))
            neighbor_vectors.append(list(sorted_dict.values()))

        return neighbor_vectors

    def generateHamiltonian(self):
        H = np.zeros((self.matrix_dim, self.matrix_dim), dtype=complex)
        for i in range(self.matrix_dim): # Loop through all sites
            label1 = self.labels[i]
            current_site = ord(label1[1]) - ord("a")

            # Chemical Potentials
            if self.sites > 1:
                H[i, i] = self.site_potential[current_site]
            else:
                H[i, i] = self.site_potential

            # Nearest Neighbor Hopping
            site_vector_list = self.neighbor_vectors[current_site] # Get all possible vectors from site
            if type(self.site_coordination) is list:
                for dist_site in range(len(self.site_coordination)): # Loop nearest to farthest sites asked for
                    vector_list = site_vector_list[dist_site] # Get list of vectors at that distance
                    for vector in vector_list: # Loop through those vectors
                        label2 = self.getNeighbor(label1, vector) # Get label of site after moving vector
                        j = self.LabelToIndex(label2)
                        
                        H[i, j] = self.site_coordination[dist_site]
            else:
                vector_list = site_vector_list[0] # Get list of vectors at that distance
                for vector in vector_list: # Loop through those vectors
                    label2 = self.getNeighbor(label1, vector) # Get label of site after moving vector
                    j = self.LabelToIndex(label2)
                    
                    H[i, j] = self.site_coordination
            
            # Special Hops
            for hop in self.special_hops:
                label2 = self.getNeighbor(label1, hop[0]) # Get label of site after moving vector
                j = self.LabelToIndex(label2)
                
                H[i, j] = hop[1]

        return H

    """def generateFourier(self):
        F = np.zeros((self.matrix_dim, self.matrix_dim), dtype=complex)
        amp = (1/np.sqrt(self.matrix_dim/self.sites))
        for i in range(len(self.labels)):
            for j in range(len(self.mlabels)):
                if self.mlabels[j][1] != self.labels[i][1]:
                    F[i, j] = 0
                    continue
                phase = 0
                if self.dims > 1:
                    phase = (np.array(self.MomentumLabelToK(self.mlabels[j])) @ np.array(self.LabelToR(self.labels[i])))
                else:
                    phase = (self.MomentumLabelToK(self.mlabels[j]) * self.LabelToR(self.labels[i]))
                F[i, j] = amp * np.exp(1j * phase)
        return np.matrix(F) """
    
    def generateFourier(self):
        matrix_dim = self.matrix_dim
        sites = self.sites
        amp = (1/np.sqrt(matrix_dim/sites))
        
        if self.dims > 1:
            R = np.array([self.LabelToR(label) for label in self.labels])
            K = np.array([self.MomentumLabelToK(mlabel) for mlabel in self.mlabels])
            
            site_labels = np.array([ord(label[1]) - ord('a') for label in self.labels])
            mlabel_sites = np.array([ord(mlabel[1]) - ord('a') for mlabel in self.mlabels])

            F = np.zeros((matrix_dim, matrix_dim), dtype=complex)
            
            for site in range(sites):
                site_mask = (site_labels == site)
                m_site_mask = (mlabel_sites == site)
                
                phases = np.dot(R[site_mask], K[m_site_mask].T)
                
                F[np.ix_(site_mask, m_site_mask)] = amp * np.exp(1j * phases)
        else:
            R = np.array([self.LabelToR(label) for label in self.labels])
            K = np.array([self.MomentumLabelToK(mlabel) for mlabel in self.mlabels])

            site_labels = np.array([ord(label[1]) - ord('a') for label in self.labels])
            mlabel_sites = np.array([ord(mlabel[1]) - ord('a') for mlabel in self.mlabels])
            
            F = np.zeros((matrix_dim, matrix_dim), dtype=complex)
            
            for site in range(sites):
                site_mask = (site_labels == site)
                m_site_mask = (mlabel_sites == site)
                
                phases = np.outer(R[site_mask], K[m_site_mask])
                
                F[np.ix_(site_mask, m_site_mask)] = amp * np.exp(1j * phases)
        
        return np.matrix(F)

    def getHk(self):
        n = self.matrix_dim
        s = self.sites
        
        if n > 1000:
            k_blocks = n // s
            Hk = np.zeros((n, n), dtype=complex)

            chunk_size = min(100, k_blocks)
            for i in range(0, k_blocks, chunk_size):
                end = min(i + chunk_size, k_blocks)
                F_chunk = self.F[:, i*s:end*s]
                Hk_chunk = F_chunk.getH() @ self.H @ F_chunk
                Hk[i*s:end*s, i*s:end*s] = Hk_chunk
        else:
            Hk = np.linalg.multi_dot([self.F.getH(), self.H, self.F])
        
        return Hk

    def getEnergyBands(self):
        if self.sites == 1:
            return np.array([self.Hk[i, i] for i in range(len(self.H))])
        energies = [[] for i in range(self.sites)]
        for i in range(self.matrix_dim//self.sites):
            block_Hk = self.Hk[self.sites*(i):self.sites*(i+1), self.sites*(i):self.sites*(i+1)]
            eigs = np.linalg.eigh(block_Hk).eigenvalues
            for site in range(self.sites):
                energies[site].append(eigs[site])
        
        return np.array(energies).real

    def getFermiLevel(self):
        if self.sites == 1:
            sorted_energy = np.sort(self.energies)
            return (sorted_energy[self.matrix_dim//2-1] + sorted_energy[self.matrix_dim//2])/2
        else:
            sorted_energy = np.sort(self.energies.ravel())
            return (sorted_energy[self.matrix_dim//2-1] + sorted_energy[self.matrix_dim//2])/2
        
    #def getBandSlice(self, momentum):


    
# Hydrogen
#hydrogen = Lattice(100, 1, 1, 0)
#print("Hydrogen:")
#print(hydrogen.labels)
#print([hydrogen.LabelToR(hydrogen.labels[i]) for i in range(len(hydrogen.labels))])
#print([hydrogen.LabelToIndex(hydrogen.labels[i]) for i in range(len(hydrogen.labels))] == [i for i in range(3)])
#print(hydrogen.mlabels)

#H = hydrogen.generateHamiltonian(-1)
#print(H)

#F = hydrogen.generateFourier()
#print(np.round(F, 2))
#print(np.round(F @ F.getH(), 2))

""" save = True
plt.figure()
plt.matshow(hydrogen.Hk.real)
if save:
    plt.savefig("images/Hydrogen_Hk.png")

plt.figure()
plt.scatter(hydrogen.momentums, hydrogen.energies)
plt.axhline(hydrogen.fermi_level)
if save:
    plt.savefig("images/Hydrogen_Evk.png")
plt.show() """

# Distorted Hydrogen
#dist_hydrogen = Lattice(100, 2, 2, [0, 0.8], [-1, -0.1])
#print("Distorted Hydrogen:")
#print(dist_hydrogen.labels)
#print([dist_hydrogen.LabelToR(dist_hydrogen.labels[i]) for i in range(len(dist_hydrogen.labels))])
#print([dist_hydrogen.LabelToIndex(dist_hydrogen.labels[i]) for i in range(len(dist_hydrogen.labels))] == [i for i in range(3*2)])
#print(dist_hydrogen.mlabels)

#print(dist_hydrogen.H)

""" save = True
plt.figure()
plt.matshow(dist_hydrogen.Hk.real)
if save:
    plt.savefig("images/Dist_Hydrogen_Hk.png")

plt.figure()
plt.scatter(dist_hydrogen.momentums, dist_hydrogen.energies[0])
plt.scatter(dist_hydrogen.momentums, dist_hydrogen.energies[1])
plt.axhline(dist_hydrogen.fermi_level)
if save:
    plt.savefig("images/Dist_Hydrogen_Evk.png")
plt.show() """

#Graphene

""" graphene = Lattice((50,50), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))])
#print("Graphene:")
#print(graphene.labels)
#print([graphene.LabelToR(graphene.labels[i]) for i in range(len(lgraphene.abels))])
#print([graphene.LabelToIndex(graphene.labels[i]) for i in range(len(graphene.labels))] == [i for i in range(9*9*2)])
#print(graphene.mlabels)
#print(graphene.reciprocal_vectors)

#print(H)
#H = graphene.generateHamiltonian([(1, 0, -1), (1, -1, -1), (0, 0, -1)], [-1, -1], -1)
#H = graphene.generateHamiltonian([(0, 1, -1), (-1, 1, -1), (0, 0, -1)], [-1, -1], -1)

save = True
plt.figure()
plt.matshow(graphene.Hk.real)
if save:
    plt.savefig("images/Graphene_Hk.png")

plt.figure()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(graphene.momentums[0], graphene.momentums[1], graphene.energies[0])
ax.plot_trisurf(graphene.momentums[0], graphene.momentums[1], graphene.energies[1])

plt.axhline(graphene.fermi_level)
if save:
    plt.savefig("images/Graphene_Evk.png")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(graphene.momentums[0], graphene.momentums[1], graphene.energies[1]-graphene.energies[0])
if save:
    plt.savefig("images/Graphene_gap.png")

fig = plt.figure()
ax2 = fig.add_subplot(111)

slice_ky_value = 0.0
tolerance = 0.1

mask = np.abs(graphene.momentums[1] - slice_ky_value) < tolerance
sort_idx = np.argsort(graphene.momentums[0][mask])

ax2.plot(graphene.momentums[0][mask][sort_idx], graphene.energies[0][mask][sort_idx], label='Lower band')
ax2.plot(graphene.momentums[0][mask][sort_idx], graphene.energies[1][mask][sort_idx], label='Upper band')

ax2.axhline(graphene.fermi_level, color='gray', linestyle='--', label='Fermi level')
ax2.set_title(f'Slice at ky = {slice_ky_value:.2f}')
ax2.set_xlabel('kx')
ax2.set_ylabel('Energy')
ax2.legend()

plt.tight_layout()

if save:
    plt.savefig("images/Graphene_Evk_with_slice.png")

save = True
pos = [graphene.LabelToR(label) for label in graphene.labels]

plt.figure()
plt.plot(*zip(*pos), marker='o', color='r', ls='')

annotations = [label for label in graphene.labels]
for (x, y), label in zip(pos, annotations):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8) """

""" for i in range(len(H)):
    for j in range(len(H)):
        if H[i, j] == -1:
            pos1 = graphene.LabelToR(graphene.labels[i])
            pos2 = graphene.LabelToR(graphene.labels[j])
            plt.plot(np.linspace(pos1[0],pos2[0], 100), np.linspace(pos1[1],pos2[1], 100), 'r-')
plt.title("Graphene Atom Positions")
plt.xlabel("x")
plt.xlabel("y")
if save:
    plt.savefig("images\\N_Graphene_Pos.png") """


#F = graphene.generateFourier()
#print(F)
#rint(np.round(F @ F.getH(), 2))

""" save = False
pos = [graphene.LabelToR(label) for label in graphene.labels]
plt.figure()
plt.plot(*zip(*pos), marker='o', color='r', ls='')
plt.title("Graphene Atom Positions")
plt.xlabel("x")
plt.xlabel("y")
if save:
    plt.savefig("images\Graphene_Pos.png")

mom = [graphene.MomentumLabelToK(mlabel) for mlabel in graphene.mlabels]
plt.figure()
plt.plot(*zip(*mom), marker='o', color='r', ls='')
plt.title("Graphene Atom Momentums")
plt.xlabel("kx")
plt.xlabel("ky")
if save:
    plt.savefig("images\Graphene_Mom.png")

plt.show() """

# Boron Nitride
""" M = 1
boron_nitride = Lattice((50,50), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))], site_potential=[M, -M])
print("Boron Nitride:")
save = True
plt.figure()
plt.matshow(boron_nitride.Hk.real)
if save:
    plt.savefig("images/Boron_Nitride_Hk.png")

plt.figure()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(boron_nitride.momentums[0], boron_nitride.momentums[1], boron_nitride.energies[0])
ax.plot_trisurf(boron_nitride.momentums[0], boron_nitride.momentums[1], boron_nitride.energies[1])
plt.axhline(boron_nitride.fermi_level)
if save:
    plt.savefig("images/Boron_Nitride_Evk.png")
plt.show()

fig = plt.figure()
ax2 = fig.add_subplot(111)

slice_ky_value = 0.0
tolerance = 0.1

mask = np.abs(boron_nitride.momentums[1] - slice_ky_value) < tolerance
sort_idx = np.argsort(boron_nitride.momentums[0][mask])

ax2.plot(boron_nitride.momentums[0][mask][sort_idx], boron_nitride.energies[0][mask][sort_idx], label='Lower band')
ax2.plot(boron_nitride.momentums[0][mask][sort_idx], boron_nitride.energies[1][mask][sort_idx], label='Upper band')

ax2.axhline(boron_nitride.fermi_level, color='gray', linestyle='--', label='Fermi level')
ax2.set_title(f'Slice at ky = {slice_ky_value:.2f}')
ax2.set_xlabel('kx')
ax2.set_ylabel('Energy')
ax2.legend()

plt.tight_layout()

if save:
    plt.savefig("images/Boron_Nitride_Evk_with_slice.png")

plt.show() """

# Haldane
""" M = 0.1
tp = 0.3
phi = 0.7

forward = -tp*np.exp(1j*phi)
backward = -tp*np.exp(-1j*phi) 

special_hops = [((1, 0, 0), forward), ((-1, 1, 0), forward), ((0, -1, 0), forward), ((-1, 0, 0), backward), ((1, -1, 0), backward), ((0, 1, 0), backward)]
haldane = Lattice((50,50), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))], site_potential=[M, -M], special_hops=special_hops)
print("Haldane:")

save = True
plt.figure()
plt.matshow(haldane.Hk.real)
if save:
    plt.savefig("images/Haldane_Hk.png")

plt.figure()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(haldane.momentums[0], haldane.momentums[1], haldane.energies[0])
ax.plot_trisurf(haldane.momentums[0], haldane.momentums[1], haldane.energies[1])

plt.axhline(haldane.fermi_level)
if save:
    plt.savefig("images/Haldane_Evk.png")

slice_ky_value = 0.0
tolerance = 0.1

mask = np.abs(haldane.momentums[1] - slice_ky_value) < tolerance
sort_idx = np.argsort(haldane.momentums[0][mask])

fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.plot(haldane.momentums[0][mask][sort_idx], haldane.energies[0][mask][sort_idx], label='Lower band')
ax2.plot(haldane.momentums[0][mask][sort_idx], haldane.energies[1][mask][sort_idx], label='Upper band')

ax2.axhline(haldane.fermi_level, color='gray', linestyle='--', label='Fermi level')
ax2.set_title(f'Slice at ky = {slice_ky_value:.2f}')
ax2.set_xlabel('kx')
ax2.set_ylabel('Energy')
ax2.legend()

plt.tight_layout()

if save:
    plt.savefig("images/Haldane_Evk_with_slice.png")

plt.show() """
        