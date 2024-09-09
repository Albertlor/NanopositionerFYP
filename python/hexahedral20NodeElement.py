import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

np.set_printoptions(linewidth=np.inf)

class Hexahedral20NodeElement:
    def __init__(self, E, nu, element_size, loading_point, support_point, solid_elements):
        self.E = E
        self.nu = nu
        self.element_size = element_size
        self.loading_point = loading_point
        self.support_point = support_point
        self.solid_elements = solid_elements
        self.D = self.construct_D_matrix()
        self.xi, self.eta, self.zeta = sp.symbols('xi eta zeta')
        self.N = self.define_shape_functions()
        self.N_func = [sp.lambdify((self.xi, self.eta, self.zeta), N) for N in self.N]
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
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0]
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
        gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        gauss_weights = np.array([5/9, 8/9, 5/9])
        K_FE = np.zeros((60, 60))
        for i, xi in enumerate(gauss_points):
            for j, eta in enumerate(gauss_points):
                for k, zeta in enumerate(gauss_points):
                    J = self.construct_Jacobian_matrix(xi, eta, zeta, nodes_physical).round(9)
                    J_inv = np.linalg.inv(J).round(9)
                    J_inv_T = J_inv.T
                    J_det = np.linalg.det(J)
                    B = self.construct_B_matrix(xi, eta, zeta, J_inv_T)
                    w = gauss_weights[i] * gauss_weights[j] * gauss_weights[k]
                    K_FE += w * B.T @ self.D @ B * J_det
        K_FE = K_FE.round(7)
        return K_FE

    def define_connectivity(self, num_elements):
        num_elements_x, num_elements_y, num_elements_z = num_elements
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
                        middle_base, middle_base + 1, middle_base + mid_nodes_per_row+1, middle_base + mid_nodes_per_row
                    ]

                    bottom_nodes = [
                        bottom_base, bottom_base + 2, bottom_base + nodes_per_row+mid_nodes_per_row+2, bottom_base + nodes_per_row+mid_nodes_per_row,
                        bottom_base + 1, bottom_base + nodes_per_row+1, bottom_base + nodes_per_row+mid_nodes_per_row+1, bottom_base + nodes_per_row
                    ]

                    top_nodes = [
                        top_base, top_base + 2, top_base + nodes_per_row+mid_nodes_per_row+2, top_base + nodes_per_row+mid_nodes_per_row,
                        top_base + 1, top_base + nodes_per_row+1, top_base + nodes_per_row+mid_nodes_per_row+1, top_base + nodes_per_row
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
                    
            bottom_layer_start = top_layer_start
            middle_layer_start = bottom_layer_start + ((num_elements_x*2 + 1) * (num_elements_y*2 + 1) - num_elements_x*num_elements_y)
            top_layer_start = middle_layer_start + (num_elements_x + 1) * (num_elements_y + 1)

        return connectivity

    # def get_element_coordinates(self, i, j, d, element_size_x, element_size_y, element_size_z):
    #     first_element_coordinates = np.array([
    #         [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
    #         [0, 0, 20], [2, 0, 20], [2, 2, 20], [0, 2, 20],
    #         [1, 0, 0], [2, 1, 0], [1, 2, 0], [0, 1, 0],
    #         [1, 0, 20], [2, 1, 20], [1, 2, 20], [0, 1, 20],
    #         [0, 0, 10], [2, 0, 10], [2, 2, 10], [0, 2, 10]
    #     ], dtype=float)
    #     # Adjust base coordinates based on element position (i, j)
    #     element_coordinates = first_element_coordinates.copy()
    #     element_coordinates[:, 0] += i * element_size_x  # Adjust x-coordinates
    #     element_coordinates[:, 1] += j * element_size_y  # Adjust y-coordinates
    #     element_coordinates[:, 2] += d * element_size_z  # Adjust y-coordinates

    #     return element_coordinates
    
    def assemble_subchain_stiffness_matrix(self, num_elements):
        num_elements_x, num_elements_y, num_elements_z = num_elements
        dof_per_node = 3
        bottom_num_nodes = (num_elements_x*2 + 1) * (num_elements_y*2 +1) - num_elements_x*num_elements_y
        top_num_nodes = bottom_num_nodes
        middle_num_nodes = (num_elements_x + 1) * (num_elements_y + 1)
        num_nodes = (bottom_num_nodes + top_num_nodes + middle_num_nodes)*num_elements_z - (num_elements_z - 1)*top_num_nodes
        size = num_nodes * dof_per_node
        K_subchain = lil_matrix((size, size))

        connectivity = self.define_connectivity(num_elements)
        # self.loading_nodes = [
        #     connectivity[self.loading_point][2],
        #     connectivity[self.loading_point][3],
        #     connectivity[self.loading_point][6],
        #     connectivity[self.loading_point][7],
        #     connectivity[self.loading_point][10],
        #     connectivity[self.loading_point][14],
        #     connectivity[self.loading_point][18],
        #     connectivity[self.loading_point][19]
        # ]
        self.loading_nodes = [
            connectivity[self.loading_point][0],
            connectivity[self.loading_point][3],
            connectivity[self.loading_point][4],
            connectivity[self.loading_point][7],
            connectivity[self.loading_point][11],
            connectivity[self.loading_point][15],
            connectivity[self.loading_point][16],
            connectivity[self.loading_point][19]
        ]

        self.support_nodes = []
        for support in self.support_point:
            nodes = [
                connectivity[support][0], connectivity[support][1], connectivity[support][4], connectivity[support][5],
                connectivity[support][8], connectivity[support][12], connectivity[support][16], connectivity[support][17]
            ]
            self.support_nodes+=nodes

        #d = 0
        for element_idx, element in enumerate(tqdm(connectivity)):
            # i = element_index % num_elements_x
            # j = element_index // num_elements_x
            # if i==(num_elements_x-1) and j==(num_elements_y-1):
            #     d+=1
            
            # nodes_physical = self.get_element_coordinates(i, j, d, element_size_x, element_size_y, element_size_z)
            x, y, z = self.element_size
            nodes_physical = np.array([
                [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
                [0, 0, z], [x, 0, z], [x, y, z], [0, y, z],
                [x/2, 0, 0], [x, y/2, 0], [x/2, y, 0], [0, y/2, 0],
                [x/2, 0, z], [x, y/2, z], [x/2, y, z], [0, y/2, z],
                [0, 0, z/2], [x, 0, z/2], [x, y, z/2], [0, y, z/2]
            ])

            if element_idx==self.loading_point:
                self.loading_nodes_coordinates = nodes_physical
            K_FE = self.compute_element_stiffness_matrix(nodes_physical)
            
            p = 1E-6
            if element_idx not in self.solid_elements:
                K_FE = p * K_FE
            for local_i in range(20):  # Loop over local nodes
                for local_j in range(20):
                    global_i = element[local_i]
                    global_j = element[local_j]
                    for k in range(3):  # Loop over DOFs per node
                        for l in range(3):
                            GI = 3 * global_i + k
                            GJ = 3 * global_j + l
                            K_subchain[GI, GJ] += K_FE[3 * local_i + k, 3 * local_j + l]

        self.K_subchain = csr_matrix(K_subchain).toarray()

    def get_subchain_stiffness_matrix(self):
        print("Getting Subchain's Stiffness Matrix...", flush=True)
        # self.K_subchain = Hexahedral8NodeElement.enforce_symmetry(self.K_subchain)
        # self.K_subchain = self.K_subchain.round(6)
        print(self.K_subchain[:8,:8])
        return self.K_subchain
    
    def get_loading_point_info(self):
        loading_J = self.construct_Jacobian_matrix(0, 1, 0, self.loading_nodes_coordinates)
        loading_J_inv = np.linalg.inv(loading_J)
        loading_J_inv_T = loading_J_inv.T
        self.N_func = [
            self.N_func[0],
            self.N_func[3],
            self.N_func[4],
            self.N_func[7],
            self.N_func[11],
            self.N_func[15],
            self.N_func[16],
            self.N_func[19],
        ]
        N_func_diff = [
            [
                self.dN_dxi_func[0],
                self.dN_dxi_func[3],
                self.dN_dxi_func[4],
                self.dN_dxi_func[7],
                self.dN_dxi_func[11],
                self.dN_dxi_func[15],
                self.dN_dxi_func[16],
                self.dN_dxi_func[19]
            ], 
            [
                self.dN_deta_func[0],
                self.dN_deta_func[3],
                self.dN_deta_func[4],
                self.dN_deta_func[7],
                self.dN_deta_func[11],
                self.dN_deta_func[15],
                self.dN_deta_func[16],
                self.dN_deta_func[19]
            ], 
            [
                self.dN_dzeta_func[0],
                self.dN_dzeta_func[3],
                self.dN_dzeta_func[4],
                self.dN_dzeta_func[7],
                self.dN_dzeta_func[11],
                self.dN_dzeta_func[15],
                self.dN_dzeta_func[16],
                self.dN_dzeta_func[19]
            ]
        ]
        return self.loading_nodes, self.N_func, N_func_diff, loading_J_inv_T
    
    def get_support_point_info(self):
        print(self.support_nodes)
        return self.support_nodes

    def visualize_design_domain(self, domain_size):
        design_domain = np.zeros(domain_size)
        for element in self.solid_elements:
            design_domain[element//domain_size[1],element%domain_size[1]] = 1
        
        # Plotting the design domain
        plt.figure(figsize=(8, 8))
        plt.imshow(design_domain, cmap='Blues', origin='lower', vmin=0, vmax=1)
        plt.title('Design Domain with Solid and Void Elements')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, domain_size[1], 1), np.arange(0, domain_size[1] + 1, 1), rotation=90)
        plt.yticks(np.arange(-0.5, domain_size[0], 1), np.arange(0, domain_size[0] + 1, 1))
        plt.show(block=True)

        # Wait for user input to close the plot
        input("Press any key to close the plot and continue...")
        plt.close()

    @staticmethod
    def enforce_symmetry(matrix):
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(i + 1, cols):
                avg_value = (matrix[i, j] + matrix[j, i]) / 2
                matrix[i, j] = avg_value
                matrix[j, i] = avg_value
        return matrix