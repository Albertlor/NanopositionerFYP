import numpy as np
import sympy as sp

from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm

np.set_printoptions(linewidth=np.inf)


class Model_Generator:
    def __init__(self, K_FE, num_elements, element_size, supporting_point, local_supporting_nodes, loading_point, local_loading_nodes, solid_elements):
        self.K_FE = K_FE
        self.num_elements = num_elements
        self.element_size = element_size
        self.supporting_point = supporting_point
        self.loading_point = loading_point
        self.local_supporting_nodes = local_supporting_nodes
        self.local_loading_nodes = local_loading_nodes
        self.solid_elements = solid_elements

    def define_connectivity(self):
        num_elements_x, num_elements_y, num_elements_z = self.num_elements
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
    
    def assemble_subchain_stiffness_matrix(self):
        num_elements_x, num_elements_y, num_elements_z = self.num_elements
        dof_per_node = 3
        bottom_num_nodes = (num_elements_x*2 + 1) * (num_elements_y*2 +1) - num_elements_x*num_elements_y
        top_num_nodes = bottom_num_nodes
        middle_num_nodes = (num_elements_x + 1) * (num_elements_y + 1)
        num_nodes = (bottom_num_nodes + top_num_nodes + middle_num_nodes)*num_elements_z - (num_elements_z - 1)*top_num_nodes
        size = num_nodes * dof_per_node
        K_subchain = lil_matrix((size, size))

        connectivity = self.define_connectivity()
        self.global_loading_nodes = []
        for load in self.loading_point:
            nodes1 = [connectivity[load][i] for i in self.local_loading_nodes]
            self.global_loading_nodes+=nodes1

        self.global_supporting_nodes = []
        for support in self.supporting_point:
            nodes2 = [connectivity[support][j] for j in self.local_supporting_nodes]
            self.global_supporting_nodes+=nodes2

        for element_idx, element in enumerate(tqdm(connectivity)):    
            K_FE = self.K_FE        
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
        return self.K_subchain

    def get_boundary_conditions(self):
        return self.global_supporting_nodes
    
    def get_loading_info(self):
        return self.global_loading_nodes

    @staticmethod
    def enforce_symmetry(matrix):
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(i + 1, cols):
                avg_value = (matrix[i, j] + matrix[j, i]) / 2
                matrix[i, j] = avg_value
                matrix[j, i] = avg_value
        return matrix