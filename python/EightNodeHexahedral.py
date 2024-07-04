import numpy as np
import sympy as sp
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

np.set_printoptions(linewidth=np.inf)

# Define the natural coordinates
xi, eta, zeta = sp.symbols('xi eta zeta')

# Define the shape functions for the 8-node element in terms of natural coordinates
N = [
    (1/8)*(1 - xi)*(1 - eta)*(1 - zeta),
    (1/8)*(1 + xi)*(1 - eta)*(1 - zeta),
    (1/8)*(1 + xi)*(1 + eta)*(1 - zeta),
    (1/8)*(1 - xi)*(1 + eta)*(1 - zeta),
    (1/8)*(1 - xi)*(1 - eta)*(1 + zeta),
    (1/8)*(1 + xi)*(1 - eta)*(1 + zeta),
    (1/8)*(1 + xi)*(1 + eta)*(1 + zeta),
    (1/8)*(1 - xi)*(1 + eta)*(1 + zeta)
]

# Symbolically compute the derivatives of the shape functions
dN_dxi = [sp.diff(N[i], xi) for i in range(8)]
dN_deta = [sp.diff(N[i], eta) for i in range(8)]
dN_dzeta = [sp.diff(N[i], zeta) for i in range(8)]

# Convert symbolic derivatives to numerical functions
dN_dxi_func = [sp.lambdify((xi, eta, zeta), dN_dxi[i]) for i in range(8)]
dN_deta_func = [sp.lambdify((xi, eta, zeta), dN_deta[i]) for i in range(8)]
dN_dzeta_func = [sp.lambdify((xi, eta, zeta), dN_dzeta[i]) for i in range(8)]

def construct_Jacobian_matrix(xi, eta, zeta, nodes_physical):
    J = np.zeros((3, 3))
    for i in range(8):
        J[0, 0] += dN_dxi_func[i](xi, eta, zeta) * nodes_physical[i, 0]
        J[0, 1] += dN_deta_func[i](xi, eta, zeta) * nodes_physical[i, 0]
        J[0, 2] += dN_dzeta_func[i](xi, eta, zeta) * nodes_physical[i, 0]
        J[1, 0] += dN_dxi_func[i](xi, eta, zeta) * nodes_physical[i, 1]
        J[1, 1] += dN_deta_func[i](xi, eta, zeta) * nodes_physical[i, 1]
        J[1, 2] += dN_dzeta_func[i](xi, eta, zeta) * nodes_physical[i, 1]
        J[2, 0] += dN_dxi_func[i](xi, eta, zeta) * nodes_physical[i, 2]
        J[2, 1] += dN_deta_func[i](xi, eta, zeta) * nodes_physical[i, 2]
        J[2, 2] += dN_dzeta_func[i](xi, eta, zeta) * nodes_physical[i, 2]
    return J

def construct_B_matrix(xi, eta, zeta, J_inv_T):
    B1 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0]
    ]) # 6x9

    B2 = np.array([
        [J_inv_T[0,0], J_inv_T[0,1], J_inv_T[0,2], 0, 0, 0, 0, 0, 0],
        [J_inv_T[1,0], J_inv_T[1,1], J_inv_T[1,2], 0, 0, 0, 0, 0, 0],
        [J_inv_T[2,0], J_inv_T[2,1], J_inv_T[2,2], 0, 0, 0, 0, 0, 0],
        [0, 0, 0, J_inv_T[0,0], J_inv_T[0,1], J_inv_T[0,2], 0, 0, 0],
        [0, 0, 0, J_inv_T[1,0], J_inv_T[1,1], J_inv_T[1,2], 0, 0, 0],
        [0, 0, 0, J_inv_T[2,0], J_inv_T[2,1], J_inv_T[2,2], 0, 0, 0],
        [0, 0, 0, 0, 0, 0, J_inv_T[0,0], J_inv_T[0,1], J_inv_T[0,2]],
        [0, 0, 0, 0, 0, 0, J_inv_T[1,0], J_inv_T[1,1], J_inv_T[1,2]],
        [0, 0, 0, 0, 0, 0, J_inv_T[2,0], J_inv_T[2,1], J_inv_T[2,2]]
    ]) # 9x9

    B3 = np.zeros((9, 24)) # 9x24
    for i in range(8):
        B3[0, 3*i] = dN_dxi_func[i](xi,eta,zeta)
        B3[1, 3*i] = dN_deta_func[i](xi,eta,zeta)
        B3[2, 3*i] = dN_dzeta_func[i](xi,eta,zeta)

        B3[3, 3*i+1] = dN_dxi_func[i](xi,eta,zeta)
        B3[4, 3*i+1] = dN_deta_func[i](xi,eta,zeta)
        B3[5, 3*i+1] = dN_dzeta_func[i](xi,eta,zeta)

        B3[6, 3*i+2] = dN_dxi_func[i](xi,eta,zeta)
        B3[7, 3*i+2] = dN_deta_func[i](xi,eta,zeta)
        B3[8, 3*i+2] = dN_dzeta_func[i](xi,eta,zeta)

    B = B1@B2@B3 # 6x24
    return B

# Material properties
E = 71e9  # Young's modulus in Pa
nu = 0.33   # Poisson's ratio

# Constitutive matrix for isotropic material (3D case)
D = E / ((1 + nu)*(1 - 2*nu)) * np.array([
    [1 - nu, nu, nu, 0, 0, 0],
    [nu, 1 - nu, nu, 0, 0, 0],
    [nu, nu, 1 - nu, 0, 0, 0],
    [0, 0, 0, (1 - 2*nu)/2, 0, 0],
    [0, 0, 0, 0, (1 - 2*nu)/2, 0],
    [0, 0, 0, 0, 0, (1 - 2*nu)/2]
])

# Define Gauss points and weights for 3-point Gauss quadrature
gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
gauss_weights = np.array([5/9, 8/9, 5/9])

def compute_stiffness_matrix(nodes_physical):
    K_FE = np.zeros((24, 24))
    for i, xi in enumerate(gauss_points):
        for j, eta in enumerate(gauss_points):
            for k, zeta in enumerate(gauss_points):
                J = construct_Jacobian_matrix(xi, eta, zeta, nodes_physical)
                J_inv = np.linalg.inv(J)
                J_inv_T = J_inv.T
                J_det = np.linalg.det(J)
                B = construct_B_matrix(xi, eta, zeta, J_inv_T)
                w = gauss_weights[i] * gauss_weights[j] * gauss_weights[k]
                K_FE += w * B.T @ D @ B * J_det
    return K_FE

# Number of elements in each direction
num_elements_x = 50
num_elements_y = 50
num_elements_z = 1

element_size_x = 2
element_size_y = 2
element_size_z = 20

# Number of nodes in each direction
dof_per_node = 3
num_nodes_x = num_elements_x + 1
num_nodes_y = num_elements_y + 1
num_nodes_z = num_elements_z + 1

# Total number of nodes
num_nodes = num_nodes_x * num_nodes_y * num_nodes_z
size = num_nodes * dof_per_node

# Initialize the global stiffness matrix using a sparse matrix
K_global = lil_matrix((size, size))

def node_index(x, y, z):
    return z * num_nodes_x * num_nodes_y + y * num_nodes_x + x

connectivity = []
for k in range(num_elements_z):
    for j in range(num_elements_y):
        for i in range(num_elements_x):
            n0 = k * (num_elements_x + 1) * (num_elements_y + 1) + j * (num_elements_x + 1) + i
            n1 = n0 + 1
            n2 = n0 + (num_elements_x + 1)
            n3 = n2 + 1
            n4 = n0 + (num_elements_x + 1) * (num_elements_y + 1)
            n5 = n4 + 1
            n6 = n4 + (num_elements_x + 1)
            n7 = n6 + 1
            connectivity.append([n0, n1, n3, n2, n4, n5, n7, n6])

# Assemble the global stiffness matrix
for element in tqdm(connectivity):
    nodes_physical = np.array([
        [element[0] % num_nodes_x * element_size_x, element[0] // num_nodes_x % num_nodes_y * element_size_y, element[0] // (num_nodes_x * num_nodes_y) * element_size_z],
        [element[1] % num_nodes_x * element_size_x, element[1] // num_nodes_x % num_nodes_y * element_size_y, element[1] // (num_nodes_x * num_nodes_y) * element_size_z],
        [element[2] % num_nodes_x * element_size_x, element[2] // num_nodes_x % num_nodes_y * element_size_y, element[2] // (num_nodes_x * num_nodes_y) * element_size_z],
        [element[3] % num_nodes_x * element_size_x, element[3] // num_nodes_x % num_nodes_y * element_size_y, element[3] // (num_nodes_x * num_nodes_y) * element_size_z],
        [element[4] % num_nodes_x * element_size_x, element[4] // num_nodes_x % num_nodes_y * element_size_y, element[4] // (num_nodes_x * num_nodes_y) * element_size_z],
        [element[5] % num_nodes_x * element_size_x, element[5] // num_nodes_x % num_nodes_y * element_size_y, element[5] // (num_nodes_x * num_nodes_y) * element_size_z],
        [element[6] % num_nodes_x * element_size_x, element[6] // num_nodes_x % num_nodes_y * element_size_y, element[6] // (num_nodes_x * num_nodes_y) * element_size_z],
        [element[7] % num_nodes_x * element_size_x, element[7] // num_nodes_x % num_nodes_y * element_size_y, element[7] // (num_nodes_x * num_nodes_y) * element_size_z]
    ], dtype=float)
    
    K_FE = compute_stiffness_matrix(nodes_physical)
    
    for local_i in range(8):  # Loop over local nodes
        for local_j in range(8):
            global_i = element[local_i]
            global_j = element[local_j]
            for k in range(3):  # Loop over DOFs per node
                for l in range(3):
                    GI = 3 * global_i + k
                    GJ = 3 * global_j + l
                    K_global[GI, GJ] += K_FE[3 * local_i + k, 3 * local_j + l]

K_global = csr_matrix(K_global).toarray()
K_global_np = K_global.view()

print(f"Global Stiffness Matrix ({K_global_np.shape}):")
print(K_global_np)
