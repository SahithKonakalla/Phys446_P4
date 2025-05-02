from matplotlib.tri import Triangulation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.linalg
import time

arrow_list = []

class Lattice():
    def __init__(self, Nshape, sites, lattice_vectors=1, sublattice_vectors=0, site_coordination=-1, site_potential=0, special_hops=[], special_site_hops=[], periodic=[]):
        
        debug = False

        self.Nshape = Nshape
        if isinstance(self.Nshape, tuple):
            self.dims = len(self.Nshape)
        else:
            self.dims = 1
        self.sites = sites
        self.lattice_vectors = lattice_vectors
        self.sublattice_vectors = sublattice_vectors

        start_time = time.time()
        self.reciprocal_vectors = self.ReciprocalVectors()
        if debug:
            print(f"ReciprocalVectors took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        self.labels = self.Labels()
        if debug:
            print(f"Labels took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        self.mlabels = self.MomentumLabels()
        if debug:
            print(f"MomentumLabels took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        self.bounds = self.GetBounds()
        if debug:
            print(f"GetBounds took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        self.neighbor_vectors = self.NeighborVectors()
        if debug:
            print(f"NeighborVectors took {time.time() - start_time:.4f} seconds")

        self.matrix_dim = len(self.labels)

        self.site_coordination = site_coordination
        self.site_potential = site_potential
        if site_potential == 0 and self.sites > 1:
            self.site_potential = [0 for i in range(self.sites)]
        
        self.special_hops = special_hops
        self.special_site_hops = special_site_hops

        start_time = time.time()
        self.H = self.generateHamiltonian()
        if debug:
            print(f"generateHamiltonian took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        self.F = self.generateFourier()
        if debug:
            print(f"generateFourier took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        self.Hk = self.getHk()
        if debug:
            print(f"Hk calculation took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        self.energies, self.eigenvectors = self.getEnergyBands()
        if debug:
            print(f"getEnergyBands took {time.time() - start_time:.4f} seconds")

        self.momentums = 0
        if self.dims > 1:
            self.momentums = []
            mom = np.array([self.MomentumLabelToK(self.mlabels[i]) for i in range(len(self.H))])[::self.sites]
            for dim in range(self.dims):
                self.momentums.append(mom[:, dim])
            self.momentums = np.array(self.momentums)
        else:
            self.momentums = np.array([self.MomentumLabelToK(self.mlabels[i]) for i in range(len(self.H))])[::self.sites]

        start_time = time.time()
        self.fermi_level = self.getFermiLevel()
        if debug:
            print(f"getFermiLevel took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        self.berry_flux = self.getBerryFlux()
        if debug:
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

        #print("max index:", max_index)
        for i in range(max_index):
            site = i % self.sites

            j = i // self.sites

            #print(i, j, site)

            if self.dims > 1:
                pos = []
                for dim in self.Nshape:
                    pos.append(j % dim + 1)
                    j =  j // dim
                
                labels.append((tuple(pos), chr(ord('a')+site)))
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
                for i in range(0, dim):
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
                
                labels.append((tuple(pos), chr(ord('a')+site)))
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
        
    def MomentumLabelToLabel(self, mlabel):
        if self.dims > 1:
            pos = []
            for dim in range(self.dims):
                pos.append(mlabel[0][dim] + 1)
            label = (tuple(pos), mlabel[1])
            return label
        else:
            label = (mlabel[0]+1, mlabel[1])
            return label
        
    def LabelToMomentumLabel(self, label):
        if self.dims > 1:
            pos = []
            for dim in range(self.dims):
                pos.append(label[0][dim] + 1)
            mlabel = (tuple(pos), label[1])
            return mlabel
        else:
            mlabel = (label[0]+1, label[1])
            return mlabel

    def getNeighbor(self, label, vector):
        pos = []
        if self.dims > 1:
            for dim in range(self.dims):
                pos.append((label[0][dim] + vector[dim] - 1) % self.Nshape[dim] + 1)
        else:
            pos = (label[0] + vector[0] - 1) % self.Nshape + 1

        site = chr((ord(label[1]) - ord("a") + vector[-1]) % self.sites + ord("a"))

        if len(pos) < self.dims: # No site at vector
            return (-1,), False

        if self.dims > 1:
            return (tuple(pos), site), True
        else:
            return (pos, site), True
    
    def getMomentumNeighbor(self, mlabel, vector):
        mom = []
        if self.dims > 1:
            for dim in range(self.dims):
                mom.append((mlabel[0][dim] + vector[dim]) % self.Nshape[dim])
        else:
            mom = (mlabel[0] + vector[0]) % self.Nshape

        site = chr((ord(mlabel[1]) - ord("a") + vector[-1]) % self.sites + ord("a"))

        if self.dims > 1:
            return (tuple(mom), site), True
        else:
            return (mom, site), True

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
                    vector.append(np.round(diff - self.Nshape[dim], 5)) # round
                else:
                    vector.append(diff)
        else:
            diff = label2[0] - label1[0]
            if abs(diff) > self.Nshape/2:
                vector.append(np.round(diff - self.Nshape, 5)) # round
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
                dist = np.round(self.getDistance(label1, label2), 5) # Round
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
                        label2, exists = self.getNeighbor(label1, vector) # Get label of site after moving vector
                        if not exists:
                            continue
                        j = self.LabelToIndex(label2)
                        
                        arrow_list.append((label1, label2))
                        H[i, j] = self.site_coordination[dist_site]
            else:
                vector_list = site_vector_list[0] # Get list of vectors at that distance
                for vector in vector_list: # Loop through those vectors
                    label2, exists = self.getNeighbor(label1, vector) # Get label of site after moving vector
                    if not exists:
                        continue
                    j = self.LabelToIndex(label2)
                    
                    arrow_list.append((label1, label2))
                    H[i, j] = self.site_coordination
            
            # Special Hops
            for hop in self.special_hops:
                label2, exists = self.getNeighbor(label1, hop[0]) # Get label of site after moving vector
                if not exists:
                    continue
                j = self.LabelToIndex(label2)
                
                H[i, j] = hop[1]
            
            # Special Site Hops
            if len(self.special_site_hops) > 0:
                for hop in self.special_site_hops[current_site]:
                    label2, exists = self.getNeighbor(label1, hop[0]) # Get label of site after moving vector
                    if not exists:
                        continue
                    j = self.LabelToIndex(label2)
                    
                    arrow_list.append((label1, label2))
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
        energy_list = [[] for i in range(self.sites)]
        eigenvector_list = [[] for i in range(self.sites)]
        for i in range(self.matrix_dim//self.sites):
            block_Hk = self.Hk[self.sites*(i):self.sites*(i+1), self.sites*(i):self.sites*(i+1)]
            eigenvalues, eigenvectors = np.linalg.eigh(block_Hk)
            for site in range(self.sites):
                energy_list[site].append(eigenvalues[site])
                eigenvector_list[site].append(eigenvectors[:,site])
        
        return np.array(energy_list).real, np.array(eigenvector_list)

    def getFermiLevel(self):
        if self.sites == 1:
            sorted_energy = np.sort(self.energies)
            return (sorted_energy[self.matrix_dim//2-1] + sorted_energy[self.matrix_dim//2])/2
        else:
            sorted_energy = np.sort(self.energies.ravel())
            return (sorted_energy[self.matrix_dim//2-1] + sorted_energy[self.matrix_dim//2])/2

    def calculateLabelBerryFlux(self, mlabel):
        total = 1
        phase = 0
        vector_list = np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])
        mlabel1 = mlabel
        current_site = ord(mlabel1[1]) - ord("a")
        for i in range(len(vector_list)):
            mlabel2, exists = self.getMomentumNeighbor(mlabel1, vector_list[i])
            if not exists:
                return 0

            #print(mlabel1, mlabel2)
            #arrow_list.append((mlabel1, mlabel2))

            index1 = (self.LabelToIndex(self.MomentumLabelToLabel(mlabel1)) // self.sites)
            index2 = (self.LabelToIndex(self.MomentumLabelToLabel(mlabel2)) // self.sites)
            
            #p = np.angle(self.eigenvectors[current_site][index1].ravel() @ self.eigenvectors[current_site][index2].ravel())
            #phase += np.angle(np.conj(self.eigenvectors[current_site][index1].ravel()) @ self.eigenvectors[current_site][index2].ravel())
            #print(self.eigenvectors.shape)
            total *= (np.conj(self.eigenvectors[current_site][index1].ravel()) @ self.eigenvectors[current_site][index2].ravel())
            mlabel1 = mlabel2
        
        #if abs(phase - np.angle(total)) > 0.001:
        #    print(phase, np.angle(total))
        #return phase
        return np.angle(total)
    
    def getBerryFlux(self):
        berry_flux = [[] for i in range(self.sites)]
        for mlabel in self.mlabels:
            current_site = ord(mlabel[1]) - ord("a")
            berry_flux[current_site].append(self.calculateLabelBerryFlux(mlabel))
        
        return np.array(berry_flux)
    
    def calculateChernNumber(self):
        return np.round(np.sum(self.berry_flux[0])/(2*np.pi),5)
        # return np.array([np.sum(self.berry_flux[i])/(2*np.pi) for i in range(self.sites)])


""" test = Lattice((4,2), 1, [(1, 0), (0, 1)])
pos = [test.LabelToR(label) for label in test.labels]
plt.plot(*zip(*pos), marker='o', color='r', ls='')
plt.show() """

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

""" graphene = Lattice((9,10), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))])
pos = [graphene.LabelToR(label) for label in graphene.labels]
plt.plot(*zip(*pos), marker='o', color='r', ls='')
plt.show() """

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

save = False
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
save = False
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

special_hops_a = [((1, 0, 0), forward), ((-1, 1, 0), forward), ((0, -1, 0), forward), ((-1, 0, 0), backward), ((1, -1, 0), backward), ((0, 1, 0), backward)]
special_hops_b = [((1, 0, 0), backward), ((-1, 1, 0), backward), ((0, -1, 0), backward), ((-1, 0, 0), forward), ((1, -1, 0), forward), ((0, 1, 0), forward)]
special_site_hops = [special_hops_a,special_hops_b]
haldane = Lattice((50,50), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))], site_potential=[M, -M], special_site_hops=special_site_hops)
print("Haldane:")

save = True
#plt.figure()
#plt.matshow(haldane.Hk.real)
#if save:
#    plt.savefig("images/Haldane_Hk.png")

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


# Haldane Pt. 2

""" save = True

tp = 0.3
phi = 0.7
forward = -tp*np.exp(1j*phi)
backward = -tp*np.exp(-1j*phi)

special_hops_a = [((1, 0, 0), forward), ((-1, 1, 0), forward), ((0, -1, 0), forward), ((-1, 0, 0), backward), ((1, -1, 0), backward), ((0, 1, 0), backward)]
special_hops_b = [((1, 0, 0), backward), ((-1, 1, 0), backward), ((0, -1, 0), backward), ((-1, 0, 0), forward), ((1, -1, 0), forward), ((0, 1, 0), forward)]
special_site_hops = [special_hops_a,special_hops_b]

E_list = []
M_list = np.linspace(0,2, 50)
for M in M_list:
    haldane = Lattice((25,25), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))], site_potential=[M, -M], special_site_hops=special_site_hops)
    E_list.append(np.min(haldane.energies[1] - haldane.energies[0]))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(M_list, np.array(E_list))
plt.xlabel("M")
plt.ylabel("E")

if save:
    plt.savefig("images/Haldane_EvM.png") 
plt.show() """

""" save = True

M_1 = 0.8
M_2 = 1.2
tp = 0.3
phi = 0.7
forward = -tp*np.exp(1j*phi)
backward = -tp*np.exp(-1j*phi)
special_hops_a = [((1, 0, 0), forward), ((-1, 1, 0), forward), ((0, -1, 0), forward), ((-1, 0, 0), backward), ((1, -1, 0), backward), ((0, 1, 0), backward)]
special_hops_b = [((1, 0, 0), backward), ((-1, 1, 0), backward), ((0, -1, 0), backward), ((-1, 0, 0), forward), ((1, -1, 0), forward), ((0, 1, 0), forward)]
special_site_hops = [special_hops_a,special_hops_b]
haldane1 = Lattice((50,50), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))], site_potential=[M_1, -M_1], special_site_hops=special_site_hops)
haldane2 = Lattice((50,50), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))], site_potential=[M_2, -M_2], special_site_hops=special_site_hops)

kx = haldane1.momentums[0]
ky = haldane1.momentums[1]
berry_flux = haldane1.berry_flux

N = haldane1.Nshape
kx_grid = kx.reshape(N[0], N[1])
ky_grid = ky.reshape(N[0], N[1])

plt.figure(figsize=(10, 8))

for site in range(haldane1.sites):
    flux_grid = berry_flux[site].reshape(N[0], N[1])
    
    plt.subplot(1, 2, site + 1)
    plt.contourf(kx_grid, ky_grid, flux_grid, levels=20, cmap='viridis')
    plt.colorbar(label='Berry Flux (radians)')
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.title(f'Berry Flux Contour for Sublattice {chr(ord("a") + site)}')
    plt.grid(True, linestyle='--', alpha=0.5)

if save:
    plt.savefig("images/Berry_0_8_contour.png")

kx = haldane2.momentums[0]
ky = haldane2.momentums[1]
berry_flux = haldane2.berry_flux

N = haldane2.Nshape
kx_grid = kx.reshape(N[0], N[1])
ky_grid = ky.reshape(N[0], N[1])

plt.figure(figsize=(10, 8))

for site in range(haldane2.sites):
    flux_grid = berry_flux[site].reshape(N[0], N[1])
    
    plt.subplot(1, 2, site + 1)
    plt.contourf(kx_grid, ky_grid, flux_grid, levels=20, cmap='viridis')
    plt.colorbar(label='Berry Flux (radians)')
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.title(f'Berry Flux Contour for Sublattice {chr(ord("a") + site)}')
    plt.grid(True, linestyle='--', alpha=0.5)

if save:
    plt.savefig("images/Berry_1_2_contour.png")

print("Chern M = 0.8:", haldane1.calculateChernNumber())
print("Chern M = 1.2:", haldane2.calculateChernNumber())

plt.show() """

""" save = True

tp = 0.3
phi = 0.7
forward = -tp*np.exp(1j*phi)
backward = -tp*np.exp(-1j*phi)

special_hops_a = [((1, 0, 0), forward), ((-1, 1, 0), forward), ((0, -1, 0), forward), ((-1, 0, 0), backward), ((1, -1, 0), backward), ((0, 1, 0), backward)]
special_hops_b = [((1, 0, 0), backward), ((-1, 1, 0), backward), ((0, -1, 0), backward), ((-1, 0, 0), forward), ((1, -1, 0), forward), ((0, 1, 0), forward)]
special_site_hops = [special_hops_a,special_hops_b]

C_list = []
M_list = np.linspace(0,2,50)
for M in M_list:
    haldane = Lattice((25,25), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))], site_potential=[M, -M], special_site_hops=special_site_hops)
    chern = haldane.calculateChernNumber()
    C_list.append(chern)
    print(M, chern)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(M_list, np.array(C_list))
plt.xlabel("M")
plt.ylabel("Chern Number")

if save:
    plt.savefig("images/Haldane_CvM.png") 
plt.show() """

""" save = True

M = 1

tp = 0.3
phi = 0.7
forward = -tp*np.exp(1j*phi)
backward = -tp*np.exp(-1j*phi)

special_hops_a = [((1, 0, 0), forward), ((-1, 1, 0), forward), ((0, -1, 0), forward), ((-1, 0, 0), backward), ((1, -1, 0), backward), ((0, 1, 0), backward)]
special_hops_b = [((1, 0, 0), backward), ((-1, 1, 0), backward), ((0, -1, 0), backward), ((-1, 0, 0), forward), ((1, -1, 0), forward), ((0, 1, 0), forward)]
special_site_hops = [special_hops_a,special_hops_b]
haldane = Lattice((5,5), 2, [(1, 0), (1/2, np.sqrt(3)/2)], [(0,0), (0, 1/np.sqrt(3))], site_potential=[M, -M], special_site_hops=special_site_hops)

mom = [haldane.MomentumLabelToK(mlabel) for mlabel in haldane.mlabels]
plt.figure()
plt.plot(*zip(*mom), marker='o', color='b', ls='')
plt.title("Haldane Atom Momentums")
plt.xlabel("kx")
plt.xlabel("ky")

for arrow in arrow_list:
    mlabel1 = arrow[0]
    mlabel2 = arrow[1]

    mdx = mlabel2[0][0] - mlabel1[0][0]
    mdy = mlabel2[0][1] - mlabel1[0][1]

    if mdx > 1:
        mlabel2 = ((mlabel1[0][0]-1, mlabel1[0][1]), mlabel1[1])
    if mdx < -1:
        mlabel2 = ((mlabel1[0][0]+1, mlabel1[0][1]), mlabel1[1])
    
    if mdy > 1:
        mlabel2 = ((mlabel1[0][0], mlabel1[0][1]-1), mlabel1[1])
    if mdy < -1:
        mlabel2 = ((mlabel1[0][0], mlabel1[0][1]+1), mlabel1[1])
    
    color="b"
    if mdx > 1 or mdx < -1 or mdy > 1 or mdy < -1:
        color = "r"

    mom1 = haldane.MomentumLabelToK(mlabel1)
    mom2 = haldane.MomentumLabelToK(mlabel2)

    dx = mom2[0] - mom1[0]
    dy = mom2[1] - mom1[1]

    if color == "r":
        plt.arrow(mom1[0]+0.1, mom1[1]+0.1, dx, dy, length_includes_head=True, head_width=0.15, head_length=0.2, color=color)
    else:
        plt.arrow(mom1[0],mom1[1], dx, dy, length_includes_head=True, head_width=0.15, head_length=0.2, color=color)
    #plt.annotate("", xytext=mom1, xy=(dx, dy),arrowprops=dict(arrowstyle="->"))

if save:
    plt.savefig("images\Haldane_arrows.png")

plt.show() """

""" save = True

M_1 = 0.2
M_2 = 2.0
tp = 0.3
phi = 0.7
forward = -tp*np.exp(1j*phi)
backward = -tp*np.exp(-1j*phi)

Ny = 25

site_vectors = []
for i in range(Ny):
    for j in range(2):
        site_vectors.append((i*1, j*1/np.sqrt(3)))

site_potential_1 = []
for i in range(Ny):
    for j in range(2):
        if j == 0:
            site_potential_1.append(M_1)
        else:
            site_potential_1.append(-M_1)

site_potential_2 = []
for i in range(Ny):
    for j in range(2):
        if j == 0:
            site_potential_2.append(M_2)
        else:
            site_potential_2.append(-M_2)


special_site_hops = []
for i in range(Ny):
    for j in range(2):
        if i == 0:
            if j == 0:
                special_site_hops.append([((0, 0, 2), forward), 
                                          ((0, -1, 0), forward), 
                                          ((0, -1, 2), backward), 
                                          ((0, 1, 0), backward)])
            else:
                special_site_hops.append([((0, 0, 2), backward), 
                                          ((0, -1, 0), backward), 
                                          ((0, -1, 2), forward), 
                                          ((0, 1, 0), forward)])
        elif i == Ny-1:
            if j == 0:
                special_site_hops.append([((0, 1, -2), forward), 
                                          ((0, -1, 0), forward), 
                                          ((0, 0, -2), backward), 
                                          ((0, 1, 0), backward)])
            else:
                special_site_hops.append([((0, 1, -2), backward), 
                                          ((0, -1, 0), backward), 
                                          ((0, 0, -2), forward), 
                                          ((0, 1, 0), forward)])
        else:
            if j == 0:
                special_site_hops.append([((0, 0, 2), forward), 
                                          ((0, 1, -2), forward), 
                                          ((0, -1, 0), forward), 
                                          ((0, 0, -2), backward), 
                                          ((0, -1, 2), backward), 
                                          ((0, 1, 0), backward)])
            else:
                special_site_hops.append([((0, 0, 2), backward), 
                                          ((0, 1, -2), backward), 
                                          ((0, -1, 0), backward), 
                                          ((0, 0, -2), forward), 
                                          ((0, -1, 2), forward), 
                                          ((0, 1, 0), forward)])


haldane1 = Lattice((1,Ny), 2*Ny, [(1, 0), (1/2, np.sqrt(3)/2)], site_vectors, site_potential=site_potential_1, special_site_hops=special_site_hops, periodic=[False, True])
haldane2 = Lattice((1,Ny), 2*Ny, [(1, 0), (1/2, np.sqrt(3)/2)], site_vectors, site_potential=site_potential_2, special_site_hops=special_site_hops, periodic=[False, True])

print("Chern M = 0.2:", haldane1.calculateChernNumber())
print("Chern M = 2.0:", haldane2.calculateChernNumber())

pos = [haldane1.LabelToR(label) for label in haldane1.labels]
plt.figure()
plt.plot(*zip(*pos), marker='o', color='b', ls='')
plt.title("Haldane Atoms")
plt.xlabel("x")
plt.ylabel("y") """

""" 
for arrow in arrow_list:
    label1 = arrow[0]
    label2 = arrow[1]

    print(label1, label2)
    color_list = "rbrbrbrbrbrbrbrbrbrb"
    color = color_list[ord(label1[1]) - ord("a")]

    dx = label2[0][0] - label1[0][0]
    dy = label2[0][1] - label1[0][1]

    if dx > 1:
        label2 = ((label1[0][0]-1, label1[0][1]), label1[1])
    if dx < -1:
        label2 = ((label1[0][0]+1, label1[0][1]), label1[1])
    
    if dy > 1:
        label2 = ((label1[0][0], label1[0][1]-1), label1[1])
    if dy < -1:
        label2 = ((label1[0][0], label1[0][1]+1), label1[1])

    pos1 = haldane1.LabelToR(label1)
    pos2 = haldane1.LabelToR(label2)

    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]

    plt.arrow(pos1[0],pos1[1], dx, dy, length_includes_head=True, head_width=0.15, head_length=0.2, color=color) """

""" plt.show()

plt.matshow(haldane2.H.real)
plt.show()
plt.matshow(haldane2.Hk.real)
plt.title("Hk")
plt.show()

for i in range(2*Ny):
    plt.scatter(haldane1.momentums[1], haldane1.energies[i], color = "b")
#plt.scatter(haldane1.momentums[1], haldane1.energies[1])
plt.xlabel("ky")
plt.ylabel("E")
if save:
    plt.savefig("images/Haldane_02_non_periodic.png")

plt.show()

for i in range(2*Ny):
    plt.scatter(haldane2.momentums[1], haldane2.energies[i], color = "b")
plt.xlabel("ky")
plt.ylabel("E")
if save:
    plt.savefig("images/Haldane_20_non_periodic.png")

plt.show() """