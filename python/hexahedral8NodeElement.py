import numpy as np
import sympy as sp
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

np.set_printoptions(linewidth=np.inf)

class Hexahedral8NodeElement:
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
            (1/8)*(1 - xi)*(1 - eta)*(1 - zeta),
            (1/8)*(1 + xi)*(1 - eta)*(1 - zeta),
            (1/8)*(1 + xi)*(1 + eta)*(1 - zeta),
            (1/8)*(1 - xi)*(1 + eta)*(1 - zeta),
            (1/8)*(1 - xi)*(1 - eta)*(1 + zeta),
            (1/8)*(1 + xi)*(1 - eta)*(1 + zeta),
            (1/8)*(1 + xi)*(1 + eta)*(1 + zeta),
            (1/8)*(1 - xi)*(1 + eta)*(1 + zeta)
        ]

    def compute_shape_function_derivatives(self):
        xi, eta, zeta = self.xi, self.eta, self.zeta
        dN_dxi = [sp.diff(N, xi) for N in self.N]
        dN_deta = [sp.diff(N, eta) for N in self.N]
        dN_dzeta = [sp.diff(N, zeta) for N in self.N]
        return dN_dxi, dN_deta, dN_dzeta

    def construct_Jacobian_matrix(self, xi, eta, zeta, nodes_physical):
        J = np.zeros((3, 3))
        for i in range(8):
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

        B3 = np.zeros((9, 24)) # 9x24
        for i in range(8):
            B3[0, 3*i] = self.dN_dxi_func[i](xi, eta, zeta)
            B3[1, 3*i] = self.dN_deta_func[i](xi, eta, zeta)
            B3[2, 3*i] = self.dN_dzeta_func[i](xi, eta, zeta)

            B3[3, 3*i+1] = self.dN_dxi_func[i](xi, eta, zeta)
            B3[4, 3*i+1] = self.dN_deta_func[i](xi, eta, zeta)
            B3[5, 3*i+1] = self.dN_dzeta_func[i](xi, eta, zeta)

            B3[6, 3*i+2] = self.dN_dxi_func[i](xi, eta, zeta)
            B3[7, 3*i+2] = self.dN_deta_func[i](xi, eta, zeta)
            B3[8, 3*i+2] = self.dN_dzeta_func[i](xi, eta, zeta)

        B = B1 @ B2 @ B3 # 6x24
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
        gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        gauss_weights = np.array([5/9, 8/9, 5/9])
        K_FE = np.zeros((24, 24))
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
        K_FE = Hexahedral8NodeElement.enforce_symmetry(K_FE)
        return K_FE
    
    def define_connectivity(self, num_elements_x, num_elements_y, num_elements_z):
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
        return connectivity

    def assemble_subchain_stiffness_matrix(self, num_elements_x, num_elements_y, num_elements_z, element_size_x, element_size_y, element_size_z):
        dof_per_node = 3
        num_nodes_x = num_elements_x + 1
        num_nodes_y = num_elements_y + 1
        num_nodes_z = num_elements_z + 1
        num_nodes = num_nodes_x * num_nodes_y * num_nodes_z
        size = num_nodes * dof_per_node
        K_subchain = lil_matrix((size, size))

        connectivity = self.define_connectivity(num_elements_x, num_elements_y, num_elements_z)

        for element_idx, element in enumerate(tqdm(connectivity)):
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
            
            K_FE = self.compute_element_stiffness_matrix(nodes_physical)
            p = 1E-6
            solid_elements = list(range(0,2450,50))+list(range(2450,2500))+list(range(49,2499,50))
            #Hexahedral8NodeElement.visualize_design_domain(solid_elements,(num_elements_y,num_elements_x))
            if element_idx not in solid_elements:
                K_FE = p * K_FE
            for local_i in range(8):  # Loop over local nodes
                for local_j in range(8):
                    global_i = element[local_i]
                    global_j = element[local_j]
                    for k in range(3):  # Loop over DOFs per node
                        for l in range(3):
                            GI = 3 * global_i + k
                            GJ = 3 * global_j + l
                            K_subchain[GI, GJ] += K_FE[3 * local_i + k, 3 * local_j + l]

        self.K_subchain = csr_matrix(K_subchain).toarray()

    def get_subchain_stiffness_matrix(self):
        return self.K_subchain
    
    @staticmethod
    def enforce_symmetry(matrix):
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(i + 1, cols):
                avg_value = (matrix[i, j] + matrix[j, i]) / 2
                matrix[i, j] = avg_value
                matrix[j, i] = avg_value
        return matrix
    
    @staticmethod
    def visualize_design_domain(solid_elements,domain_size):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        design_domain = np.zeros(domain_size)
        for element in solid_elements:
            design_domain[element//domain_size[0],element%domain_size[1]] = 1
        df = pd.DataFrame(design_domain)
        print(df)