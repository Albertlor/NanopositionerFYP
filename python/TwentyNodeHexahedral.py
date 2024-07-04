import numpy as np
import sympy as sp
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

np.set_printoptions(linewidth=np.inf)

# Define the natural coordinates
xi, eta, zeta = sp.symbols('xi eta zeta')

# Define the shape functions for the 20-node element in terms of natural coordinates
N = [
    (1/8)*(1 - xi)*(1 - eta)*(1 - zeta)*(-xi - eta - zeta - 2),
    (1/8)*(1 + xi)*(1 - eta)*(1 - zeta)*( xi - eta - zeta - 2),
    (1/8)*(1 + xi)*(1 + eta)*(1 - zeta)*( xi + eta - zeta - 2),
    (1/8)*(1 - xi)*(1 + eta)*(1 - zeta)*(-xi + eta - zeta - 2),
    (1/8)*(1 - xi)*(1 - eta)*(1 + zeta)*(-xi - eta + zeta - 2),
    (1/8)*(1 + xi)*(1 - eta)*(1 + zeta)*( xi - eta + zeta - 2),
    (1/8)*(1 + xi)*(1 + eta)*(1 + zeta)*( xi + eta + zeta - 2),
    (1/8)*(1 - xi)*(1 + eta)*(1 + zeta)*(-xi + eta + zeta - 2),
    (1/4)*(1 - xi**2)*(1 - eta)*(1 - zeta),
    (1/4)*(1 + xi)*(1 - eta**2)*(1 - zeta),
    (1/4)*(1 - xi**2)*(1 + eta)*(1 - zeta),
    (1/4)*(1 - xi)*(1 - eta**2)*(1 - zeta),
    (1/4)*(1 - xi**2)*(1 - eta)*(1 + zeta),
    (1/4)*(1 + xi)*(1 - eta**2)*(1 + zeta),
    (1/4)*(1 - xi**2)*(1 + eta)*(1 + zeta),
    (1/4)*(1 - xi)*(1 - eta**2)*(1 + zeta),
    (1/4)*(1 - xi)*(1 - eta)*(1 - zeta**2),
    (1/4)*(1 + xi)*(1 - eta)*(1 - zeta**2),
    (1/4)*(1 + xi)*(1 + eta)*(1 - zeta**2),
    (1/4)*(1 - xi)*(1 + eta)*(1 - zeta**2)
]

# Symbolically compute the derivatives of the shape functions
dN_dxi = [sp.diff(N[i], xi) for i in range(20)]
dN_deta = [sp.diff(N[i], eta) for i in range(20)]
dN_dzeta = [sp.diff(N[i], zeta) for i in range(20)]

# Convert symbolic derivatives to numerical functions
dN_dxi_func = [sp.lambdify((xi, eta, zeta), dN_dxi[i]) for i in range(20)]
dN_deta_func = [sp.lambdify((xi, eta, zeta), dN_deta[i]) for i in range(20)]
dN_dzeta_func = [sp.lambdify((xi, eta, zeta), dN_dzeta[i]) for i in range(20)]

def construct_Jacobian_matrix(xi, eta, zeta, nodes_physical):
    J = np.zeros((3, 3))
    for i in range(20):
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

    B3 = np.zeros((9, 60)) # 9x60
    for i in range(20):
        B3[0, 3*i] = dN_dxi_func[i](xi,eta,zeta)
        B3[1, 3*i] = dN_deta_func[i](xi,eta,zeta)
        B3[2, 3*i] = dN_dzeta_func[i](xi,eta,zeta)

        B3[3, 3*i+1] = dN_dxi_func[i](xi,eta,zeta)
        B3[4, 3*i+1] = dN_deta_func[i](xi,eta,zeta)
        B3[5, 3*i+1] = dN_dzeta_func[i](xi,eta,zeta)

        B3[6, 3*i+2] = dN_dxi_func[i](xi,eta,zeta)
        B3[7, 3*i+2] = dN_deta_func[i](xi,eta,zeta)
        B3[8, 3*i+2] = dN_dzeta_func[i](xi,eta,zeta)

    B = B1@B2@B3 # 6x60
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

# Define Gauss points and weights for 4-point Gauss quadrature
gauss_points = np.array([-np.sqrt(3/7-2/7*np.sqrt(6/5)), np.sqrt(3/7-2/7*np.sqrt(6/5)), 
                         -np.sqrt(3/7+2/7*np.sqrt(6/5)), np.sqrt(3/7+2/7*np.sqrt(6/5))])
gauss_weights = np.array([(18+np.sqrt(30))/36, (18+np.sqrt(30))/36, 
                          (18-np.sqrt(30))/36, (18-np.sqrt(30))/36])

def compute_stiffness_matrix(nodes_physical):
    K_FE = np.zeros((60, 60))
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

# Total number of nodes
num_nodes = 18003
size = num_nodes * dof_per_node

# Initialize the global stiffness matrix using a sparse matrix
K_global = lil_matrix((size, size))

connectivity = []
nodes_per_row = 101
mid_nodes_per_row = 51

bottom_layer_start = 0
middle_layer_start = 7701
top_layer_start = 10302

for j in range(num_elements_y):
    for i in range(num_elements_x):
        # Calculate the base indices for the bottom and top layers
        bottom_base = bottom_layer_start + (j * (nodes_per_row + mid_nodes_per_row)) + (i * 2)
        top_base = top_layer_start + (j * (nodes_per_row + mid_nodes_per_row)) + (i * 2)

        # Adjust middle layer node indexing
        middle_base = middle_layer_start + (j * mid_nodes_per_row) + i
        middle_nodes = [
            middle_base, middle_base + 1, middle_base + 52, middle_base + 51
        ]

        bottom_nodes = [
            bottom_base, bottom_base + 2, bottom_base + 154, bottom_base + 152,
            bottom_base + 1, bottom_base + 102, bottom_base + 153, bottom_base + 101
        ]

        top_nodes = [
            top_base, top_base + 2, top_base + 154, top_base + 152,
            top_base + 1, top_base + 102, top_base + 153, top_base + 101
        ]

        # Combine nodes for the element in the correct sequence
        element_nodes = [
            bottom_nodes[0], bottom_nodes[1], bottom_nodes[2], bottom_nodes[3],
            top_nodes[0], top_nodes[1], top_nodes[2], top_nodes[3],
            bottom_nodes[4], bottom_nodes[5], bottom_nodes[6], bottom_nodes[7],
            top_nodes[4], top_nodes[5], top_nodes[6], top_nodes[7],
            middle_nodes[0], middle_nodes[1], middle_nodes[2], middle_nodes[3]
        ]
        
        connectivity.append(element_nodes)
        
# Starting coordinates for the first element
first_element_coordinates = np.array([
    [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
    [0, 0, 20], [2, 0, 20], [2, 2, 20], [0, 2, 20],
    [1, 0, 0], [2, 1, 0], [1, 2, 0], [0, 1, 0],
    [1, 0, 20], [2, 1, 20], [1, 2, 20], [0, 1, 20],
    [0, 0, 10], [2, 0, 10], [2, 2, 10], [0, 2, 10]
], dtype=float)

def get_element_coordinates(i, j):
    # Adjust base coordinates based on element position (i, j)
    element_coordinates = first_element_coordinates.copy()
    element_coordinates[:, 0] += i * element_size_x  # Adjust x-coordinates
    element_coordinates[:, 1] += j * element_size_y  # Adjust y-coordinates
    
    return element_coordinates

# Assemble the global stiffness matrix
for element_index,element in enumerate(tqdm(connectivity)):
    j = element_index // num_elements_x
    i = element_index % num_elements_x

    # Get the physical coordinates for the current element
    nodes_physical = get_element_coordinates(i, j)
    K_FE = compute_stiffness_matrix(nodes_physical)
    
    for local_i in range(20):  # Loop over local nodes
        for local_j in range(20):
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
