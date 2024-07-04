import numpy as np
import sympy as sp
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

np.set_printoptions(linewidth=np.inf)

class Hexahedral20NodeElement:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.D = self.construct_D_matrix()
        self.xi, self.eta, self.zeta = sp.symbols('xi eta zeta')
        self.N = self.define_shape_functions()
        self.dN_dxi, self.dN_deta, self.dN_dzeta = self.compute_shape_function_derivatives()
        self.dN_dxi_func = [sp.lambdify((self.xi, self.eta, self.zeta), dN) for dN in self.dN_dxi]
        self.dN_deta_func = [sp.lambdify((self.xi, self.eta, self.zeta), dN) for dN in self.dN_deta]
        self.dN_dzeta_func = [sp.lambdify((self.xi, self.eta, self.zeta), dN) for dN in self.dN_dzeta]

    def define_shape_functions(self):
        xi, eta, zeta = self.xi, self.eta, self.zeta
        return [
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

    def compute_shape_function_derivatives(self):
        xi, eta, zeta = self.xi, self.eta, self.zeta
        dN_dxi = [sp.diff(N, xi) for N in self.N]
        dN_deta = [sp.diff(N, eta) for N in self.N]
        dN_dzeta = [sp.diff(N, zeta) for N in self.N]
        return dN_dxi, dN_deta, dN_dzeta

    def construct_Jacobian_matrix(self, xi, eta, zeta, nodes_physical):
        J = np.zeros((3, 3))
        for i in range(20):
            J[0, 0] += self.dN_dxi_func[i](xi, eta, zeta) * nodes_physical[i, 0]
            J[0, 1] += self.dN_deta_func[i](xi, eta, zeta) * nodes_physical[i, 0]
            J[0, 2] += self.dN_dzeta_func[i](xi, eta, zeta) * nodes_physical[i, 0]
            J[1, 0] += self.dN_dxi_func[i](xi, eta, zeta) * nodes_physical[i, 1]
            J[1, 1] += self.dN_deta_func[i](xi, eta, zeta) * nodes_physical[i, 1]
            J[1, 2] += self.dN_dzeta_func[i](xi, eta, zeta) * nodes_physical[i, 1]
            J[2, 0] += self.dN_dxi_func[i](xi, eta, zeta) * nodes_physical[i, 2]
            J[2, 1] += self.dN_deta_func[i](xi, eta, zeta) * nodes_physical[i, 2]
            J[2, 2] += self.dN_dzeta_func[i](xi, eta, zeta) * nodes_physical[i, 2]
        return J

    def construct_B_matrix(self, xi, eta, zeta, J_inv_T):
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
            B3[0, 3*i] = self.dN_dxi_func[i](xi, eta, zeta)
            B3[1, 3*i] = self.dN_deta_func[i](xi, eta, zeta)
            B3[2, 3*i] = self.dN_dzeta_func[i](xi, eta, zeta)

            B3[3, 3*i+1] = self.dN_dxi_func[i](xi, eta, zeta)
            B3[4, 3*i+1] = self.dN_deta_func[i](xi, eta, zeta)
            B3[5, 3*i+1] = self.dN_dzeta_func[i](xi, eta, zeta)

            B3[6, 3*i+2] = self.dN_dxi_func[i](xi, eta, zeta)
            B3[7, 3*i+2] = self.dN_deta_func[i](xi, eta, zeta)
            B3[8, 3*i+2] = self.dN_dzeta_func[i](xi, eta, zeta)

        B = B1 @ B2 @ B3 # 6x60
        return B

    def construct_D_matrix(self):
        D = self.E / ((1 + self.nu)*(1 - 2*self.nu)) * np.array([
            [1 - self.nu, self.nu, self.nu, 0, 0, 0],
            [self.nu, 1 - self.nu, self.nu, 0, 0, 0],
            [self.nu, self.nu, 1 - self.nu, 0, 0, 0],
            [0, 0, 0, (1 - 2*self.nu)/2, 0, 0],
            [0, 0, 0, 0, (1 - 2*self.nu)/2, 0],
            [0, 0, 0, 0, 0, (1 - 2*self.nu)/2]
        ])
        return D

    def compute_element_stiffness_matrix(self, nodes_physical):
        gauss_points = np.array([-np.sqrt(3/7-2/7*np.sqrt(6/5)), np.sqrt(3/7-2/7*np.sqrt(6/5)), 
                                 -np.sqrt(3/7+2/7*np.sqrt(6/5)), np.sqrt(3/7+2/7*np.sqrt(6/5))])
        gauss_weights = np.array([(18+np.sqrt(30))/36, (18+np.sqrt(30))/36, 
                                  (18-np.sqrt(30))/36, (18-np.sqrt(30))/36])
        K_FE = np.zeros((60, 60))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                for k, zeta in enumerate(gauss_points):
                    J = self.construct_Jacobian_matrix(xi, eta, zeta, nodes_physical)
                    J_inv = np.linalg.inv(J)
                    J_inv_T = J_inv.T
                    J_det = np.linalg.det(J)
                    B = self.construct_B_matrix(xi, eta, zeta, J_inv_T)
                    w = gauss_weights[i] * gauss_weights[j] * gauss_weights[k]
                    K_FE += w * B.T @ self.D @ B * J_det
        return K_FE

    def define_connectivity(self, num_elements_x, num_elements_y, num_elements_z):
        connectivity = []
        nodes_per_row = num_elements_x*2 + 1
        mid_nodes_per_row = num_elements_x + 1

        bottom_layer_start = 0
        middle_layer_start = bottom_layer_start + ((num_elements_x*2 + 1) * (num_elements_y*2 + 1) - num_elements_x*num_elements_y)
        top_layer_start = middle_layer_start + (num_elements_x + 1) * (num_elements_y + 1)
        for _ in range(num_elements_z):
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
                    print(connectivity[-1])
            bottom_layer_start = top_layer_start
            middle_layer_start = bottom_layer_start + ((num_elements_x*2 + 1) * (num_elements_y*2 + 1) - num_elements_x*num_elements_y)
            top_layer_start = middle_layer_start + (num_elements_x + 1) * (num_elements_y + 1)

        return connectivity

    def get_element_coordinates(self, i, j, d, element_size_x, element_size_y, element_size_z):
        first_element_coordinates = np.array([
            [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
            [0, 0, 20], [2, 0, 20], [2, 2, 20], [0, 2, 20],
            [1, 0, 0], [2, 1, 0], [1, 2, 0], [0, 1, 0],
            [1, 0, 20], [2, 1, 20], [1, 2, 20], [0, 1, 20],
            [0, 0, 10], [2, 0, 10], [2, 2, 10], [0, 2, 10]
        ], dtype=float)
        # Adjust base coordinates based on element position (i, j)
        element_coordinates = first_element_coordinates.copy()
        element_coordinates[:, 0] += i * element_size_x  # Adjust x-coordinates
        element_coordinates[:, 1] += j * element_size_y  # Adjust y-coordinates
        element_coordinates[:, 2] += d * element_size_z  # Adjust y-coordinates

        return element_coordinates
    
    def assemble_global_stiffness_matrix(self, num_elements_x, num_elements_y, num_elements_z, element_size_x, element_size_y, element_size_z):
        dof_per_node = 3
        bottom_num_nodes = (num_elements_x*2 + 1) * (num_elements_y*2 +1) - num_elements_x*num_elements_y
        top_num_nodes = bottom_num_nodes
        middle_num_nodes = (num_elements_x + 1) * (num_elements_y + 1)
        num_nodes = (bottom_num_nodes + top_num_nodes + middle_num_nodes)*num_elements_z - (num_elements_z - 1)*top_num_nodes
        size = num_nodes * dof_per_node
        K_global = lil_matrix((size, size))

        connectivity = self.define_connectivity(num_elements_x, num_elements_y, num_elements_z)

        d = 0
        for element_index, element in enumerate(tqdm(connectivity)):
            i = element_index % num_elements_x
            j = element_index // num_elements_x
            if i==(num_elements_x-1) and j==(num_elements_y-1):
                d+=1
            
            nodes_physical = self.get_element_coordinates(i, j, d, element_size_x, element_size_y, element_size_z)
            K_FE = self.compute_element_stiffness_matrix(nodes_physical)
            
            for local_i in range(20):  # Loop over local nodes
                for local_j in range(20):
                    global_i = element[local_i]
                    global_j = element[local_j]
                    for k in range(3):  # Loop over DOFs per node
                        for l in range(3):
                            GI = 3 * global_i + k
                            GJ = 3 * global_j + l
                            K_global[GI, GJ] += K_FE[3 * local_i + k, 3 * local_j + l]

        self.K_global = csr_matrix(K_global).toarray()

    def get_global_stiffness_matrix(self):
        return self.K_global