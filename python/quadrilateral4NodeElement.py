import numpy as np
import sympy as sp
import pandas as pd

from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

np.set_printoptions(linewidth=np.inf)

class Quadrilateral4NodeElement:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.D = self.construct_D_matrix()
        self.xi, self.eta = sp.symbols('xi eta')
        self.N = self.define_shape_functions()
        self.dN_dxi, self.dN_deta = self.compute_shape_function_derivatives()
        self.dN_dxi_func = [sp.lambdify((self.xi, self.eta), dN) for dN in self.dN_dxi]
        self.dN_deta_func = [sp.lambdify((self.xi, self.eta), dN) for dN in self.dN_deta]

    def define_shape_functions(self):
        xi, eta = self.xi, self.eta
        return [
            (1/4)*(1 - xi)*(1 - eta),
            (1/4)*(1 + xi)*(1 - eta),
            (1/4)*(1 + xi)*(1 + eta),
            (1/4)*(1 - xi)*(1 + eta)
        ]

    def compute_shape_function_derivatives(self):
        xi, eta = self.xi, self.eta
        dN_dxi = [sp.diff(N, xi) for N in self.N]
        dN_deta = [sp.diff(N, eta) for N in self.N]
        return dN_dxi, dN_deta

    def construct_Jacobian_matrix(self, xi, eta, nodes_physical):
        J = np.zeros((2, 2))
        for i in range(4):
            J[0, 0] += self.dN_dxi_func[i](xi, eta) * nodes_physical[i, 0]
            J[0, 1] += self.dN_deta_func[i](xi, eta) * nodes_physical[i, 0]
            J[1, 0] += self.dN_dxi_func[i](xi, eta) * nodes_physical[i, 1]
            J[1, 1] += self.dN_deta_func[i](xi, eta) * nodes_physical[i, 1]
        return J

    def construct_B_matrix(self, xi, eta, J_inv_T):
        B1 = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 0]
        ]) # 3x4

        B2 = np.array([
            [J_inv_T[0,0], J_inv_T[0,1], 0, 0],
            [J_inv_T[1,0], J_inv_T[1,1], 0, 0],
            [0, 0, J_inv_T[0,0], J_inv_T[0,1]],
            [0, 0, J_inv_T[1,0], J_inv_T[1,1]]
        ]) # 4x4

        B3 = np.zeros((4, 8)) # 4x8
        for i in range(4):
            B3[0, 2*i] = self.dN_dxi_func[i](xi,eta)
            B3[1, 2*i] = self.dN_deta_func[i](xi,eta)

            B3[2, 2*i+1] = self.dN_dxi_func[i](xi,eta)
            B3[3, 2*i+1] = self.dN_deta_func[i](xi,eta)

        B = B1 @ B2 @ B3 # 6x24
        return B

    def construct_D_matrix(self):
        D = self.E / ((1 + self.nu)*(1 - 2*self.nu)) * np.array([
            [1 - self.nu, self.nu, 0],
            [self.nu, 1 - self.nu, 0],
            [0, 0, (1 - 2*self.nu)/2]
        ])
        return D

    def compute_element_stiffness_matrix(self, nodes_physical):
        gauss_points = np.array([-np.sqrt(1/3), np.sqrt(1/3)])
        gauss_weights = np.array([1, 1])
        K_FE = np.zeros((8, 8))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                J = self.construct_Jacobian_matrix(xi, eta, nodes_physical)
                J_inv = np.linalg.inv(J)
                J_inv_T = J_inv.T
                J_det = np.linalg.det(J)
                B = self.construct_B_matrix(xi, eta, J_inv_T)
                w = gauss_weights[i] * gauss_weights[j]
                K_FE += w * B.T @ self.D @ B * J_det
        K_FE = Quadrilateral4NodeElement.enforce_symmetry(K_FE)
        return K_FE
    
    def define_connectivity(self, num_elements_x, num_elements_y):
        num_nodes_x = num_elements_x + 1

        connectivity = []
        for j in range(num_elements_y):
            for i in range(num_elements_x):
                n0 = i * num_nodes_x + j
                n1 = n0 + 1
                n2 = n1 + num_nodes_x
                n3 = n2 - 1
                connectivity.append([n0, n1, n2, n3])
        return connectivity

    def assemble_subchain_stiffness_matrix(self, num_elements_x, num_elements_y, element_size_x, element_size_y):
        dof_per_node = 2
        num_nodes_x = num_elements_x + 1
        num_nodes_y = num_elements_y + 1
        num_nodes = num_nodes_x * num_nodes_y
        size = num_nodes * dof_per_node
        K_subchain = lil_matrix((size, size))

        connectivity = self.define_connectivity(num_elements_x, num_elements_y)

        for element_idx, element in enumerate(tqdm(connectivity)):
            nodes_physical = np.array([
                [element[0] % num_nodes_x, element[0] // num_nodes_x],
                [element[1] % num_nodes_x, element[1] // num_nodes_x],
                [element[2] % num_nodes_x, element[2] // num_nodes_x],
                [element[3] % num_nodes_x, element[3] // num_nodes_x]
            ], dtype=float)
           
            K_FE = self.compute_element_stiffness_matrix(nodes_physical)
            K_FE = K_FE.round(7)
            print(np.linalg.det(K_FE))
            # p = 1E-6
            # solid_elements = list(range(0,2450,50))+list(range(2450,2500))+list(range(49,2499,50))
            # #Hexahedral8NodeElement.visualize_design_domain(solid_elements,(num_elements_y,num_elements_x))
            # if element_idx not in solid_elements:
            #     K_FE = p * K_FE
            for local_i in range(4):  # Loop over local nodes
                for local_j in range(4):
                    global_i = element[local_i]
                    global_j = element[local_j]
                    for k in range(2):  # Loop over DOFs per node
                        for l in range(2):
                            GI = 2*global_i + k
                            GJ = 2*global_j + l
                            K_subchain[GI, GJ] += K_FE[2*local_i + k, 2*local_j + l]

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