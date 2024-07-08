import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

class SubchainFEA:
    def __init__(self, K_subchain, force_vector):
        self.K_sc = csc_matrix(K_subchain)  # Use CSC format for efficiency in solving
        self.F = lil_matrix(force_vector)   # Use LIL format for constructing the force vector

    def define_extraction_matrix(self, loading_point):
        self.A = lil_matrix((self.F.shape[1], self.K_sc.shape[0]))  # Use LIL format for construction
        dof_per_node = 3
        start_col = loading_point * dof_per_node

        for i in range(dof_per_node):
            self.A[i, start_col + i] = 1
            self.A[i + dof_per_node, start_col + i] = 1

        self.A = self.A.tocsc()  # Convert to CSC format for operations

    def define_force_vector(self, loading_point):
        dof_per_node = 3
        start_row = loading_point * dof_per_node

        self.F[start_row, 0] = 1.0  # Load in x at node 1
        self.F[start_row + 1, 1] = 1.0  # Load in y at node 1
        self.F[start_row + 2, 2] = 1.0  # Load in z at node 1
        self.F[start_row, 3] = 1.0  # Moment about x at node 1
        self.F[start_row + 1, 4] = 1.0  # Moment about y at node 1
        self.F[start_row + 2, 5] = 1.0  # Moment about z at node 1

        self.F = self.F.tocsc()  # Convert to CSC format for operations

    def compute_deformation_matrix(self):
        # Using spsolve for efficient solution
        self.U = spsolve(self.K_sc, self.F.toarray())
        self.U = SubchainFEA.enforce_symmetry(self.U)

    def apply_boundary_conditions(self, fix_nodes):
        for node in fix_nodes:
            dof_per_node = 3
            start_row = node * dof_per_node
            self.U[start_row:start_row+dof_per_node, 0:6] = 0

    def extract_compliance_matrix(self):
        self.C_extracted = self.A @ self.U
        return self.C_extracted
    
    def assemble_global_stiffness_matrix(self, K_extracted):
        p = 0.06

        theta1_z = 3*np.pi/2
        p1_x = p*np.cos(theta1_z)
        p1_y = p*np.sin(theta1_z)
        p1_z = 0
        J1 = np.array([
            [1, 0, 0,  0,   -p1_z, p1_y],
            [0, 1, 0,  p1_z, 0,   -p1_x],
            [0, 0, 1, -p1_y, p1_x, 0   ],
            [0, 0, 0,  1,    0,    0   ],
            [0, 0, 0,  0,    1,    0   ],
            [0, 0, 0,  0,    0,    1   ]
        ])

        theta2_z = np.pi/6
        p2_x = p*np.cos(theta2_z)
        p2_y = p*np.sin(theta2_z)
        p2_z = 0
        J2 = np.array([
            [1, 0, 0,  0,   -p2_z, p2_y],
            [0, 1, 0,  p2_z, 0,   -p2_x],
            [0, 0, 1, -p2_y, p2_x, 0   ],
            [0, 0, 0,  1,    0,    0   ],
            [0, 0, 0,  0,    1,    0   ],
            [0, 0, 0,  0,    0,    1   ]
        ])

        theta3_z = 5*np.pi/6
        p3_x = p*np.cos(theta3_z)
        p3_y = p*np.sin(theta3_z)
        p3_z = 0
        J3 = np.array([
            [1, 0, 0,  0,   -p3_z, p3_y],
            [0, 1, 0,  p3_z, 0,   -p3_x],
            [0, 0, 1, -p3_y, p3_x, 0   ],
            [0, 0, 0,  1,    0,    0   ],
            [0, 0, 0,  0,    1,    0   ],
            [0, 0, 0,  0,    0,    1   ]
        ])

        J = [J1, J2, J3]
        self.K_global = np.zeros((K_extracted.shape[0], K_extracted.shape[1]))
        for i in range(3):
            J_inv = np.linalg.inv(J[i])
            J_inv = SubchainFEA.enforce_symmetry(J_inv)
            K_extracted = SubchainFEA.enforce_symmetry(K_extracted)
            self.K_global += J_inv.T @ K_extracted @ J_inv

    def get_global_stiffness_matrix(self):
        self.K_global = SubchainFEA.format_matrix_scientific(self.K_global)
        return self.K_global
    
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
    def format_matrix_scientific(matrix, precision=1):
        # Round the matrix first
        rounded_matrix = np.round(matrix, decimals=precision)
        
        # Define a function to apply scientific formatting
        def format_scientific(x):
            return np.format_float_scientific(x, precision=precision)
        
        # Vectorize the function
        vectorized_format = np.vectorize(format_scientific)
        
        # Apply the function to the entire matrix
        formatted_matrix = vectorized_format(rounded_matrix)
        return formatted_matrix

# Example usage:
# K_subchain = np.random.rand(6, 6)  # Replace with your stiffness matrix
# force_vector = np.zeros((6, 6))    # Replace with your force vector

# fea = SubchainFEA(K_subchain, force_vector)
# fea.define_extraction_matrix(loading_point=0)
# fea.define_force_vector(loading_point=0)
# fea.compute_deformation_matrix()
# fea.apply_boundary_conditions(fix_nodes=[0])
# compliance_matrix = fea.extract_compliance_matrix()
# fea.assemble_global_stiffness_matrix(compliance_matrix)
# K_global = fea.get_global_stiffness_matrix()

# Round the global stiffness matrix to 3 decimal places
# rounded_K_global = SubchainFEA.round_matrix(K_global, decimals=3)
# print(rounded_K_global)

# Format the global stiffness matrix in scientific notation
# formatted_K_global = SubchainFEA.format_matrix_scientific(K_global, precision=3)
# print(formatted_K_global)
