import numpy as np
from scipy.integrate import nquad

# Parameters
a = 0.1          # Lattice spacing
alpha = 1.0      # Exponent for s-orbital
beta = 1.0       # Exponent for d-orbitals

# Define s-orbital: e^(-alpha * r^2)
def s_orbital(x, y, z):
    r_sq = x**2 + y**2 + z**2
    return np.exp(-alpha * r_sq)

# Define d-orbitals: (angular part) * e^(-beta * r^2)
def d_xy(x, y, z):
    r_sq = x**2 + y**2 + z**2
    return x * y * np.exp(-beta * r_sq)

def d_x2y2(x, y, z):
    r_sq = x**2 + y**2 + z**2
    return (x**2 - y**2) * np.exp(-beta * r_sq)

def d_z2(x, y, z):
    r_sq = x**2 + y**2 + z**2
    return (3 * z**2 - r_sq) * np.exp(-beta * r_sq)

def d_xz(x, y, z):
    r_sq = x**2 + y**2 + z**2
    return x * z * np.exp(-beta * r_sq)

def d_yz(x, y, z):
    r_sq = x**2 + y**2 + z**2
    return y * z * np.exp(-beta * r_sq)

# Nearest-neighbor displacements (3D for generality, but z=0 for 2D lattice)
neighbors = [
    ("+x", np.array([a, 0, 0])),
    ("+y", np.array([0, a, 0])),
    ("-x", np.array([-a, 0, 0])),
    ("-y", np.array([0, -a, 0])),
]

# Compute overlap integral for a given d-orbital and displacement
def compute_overlap_sd(d_func, delta):
    def integrand(x, y, z):
        return s_orbital(x, y, z) * d_func(x - delta[0], y - delta[1], z - delta[2])
    limits = [[-5, 5], [-5, 5], [-5, 5]]  # Large enough for decay
    result, _ = nquad(integrand, limits)
    return result

def compute_overlap_sd(d_func1, d_func2, delta):
    def integrand(x, y, z):
        return d_func1(x, y, z) * d_func2(x - delta[0], y - delta[1], z - delta[2])
    limits = [[-5, 5], [-5, 5], [-5, 5]]  # Large enough for decay
    result, _ = nquad(integrand, limits)
    return result

# Test all d-orbitals
d_orbitals = {
    "d_xy": d_xy,
    #"d_x2y2": d_x2y2,
    #"d_z2": d_z2,
    "d_xz": d_xz,
    "d_yz": d_yz,
}

""" print("Overlap integrals for s-d hybridization in a square lattice (by direction):")
for name, d_func in d_orbitals.items():
    print(f"\nOrbital: {name}")
    for direction, delta in neighbors:
        overlap = compute_overlap_sd(d_func, delta)
        print(f"  Direction {direction}: ⟨s|{name}⟩ = {overlap:.6f}") """

for i in range(len(d_orbitals.items())):
    name1, d_func1 = list(d_orbitals.items())[i]
    for j in range(i, len(d_orbitals.items())):
        name2, d_func2 = list(d_orbitals.items())[j]
        print(f"\nOrbital 1: {name1}, Orbital 2: {name2}")
        for direction, delta in neighbors:
            overlap = compute_overlap_sd(d_func1, d_func2, delta)
            print(f"  Direction {direction}: ⟨{name1}|{name2}⟩ = {overlap:.6f}")