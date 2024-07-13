import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

class SubchainFEA:
    def __init__(self, K_subchain, force_vector, loading_nodes, N_func, dN_dxi_func, dN_deta_func, dN_dzeta_func, loading_J_inv_T, dof_per_node):
        self.K_sc = K_subchain  # Use CSC format for efficiency in solving
        self.F = force_vector   # Use LIL format for constructing the force vector
        self.loading_nodes = loading_nodes  
        self.N_func = N_func
        self.dN_dxi_func = dN_dxi_func
        self.dN_deta_func = dN_deta_func
        self.dN_dzeta_func = dN_dzeta_func
        self.loading_J_inv_T = loading_J_inv_T
        self.dof_per_node = dof_per_node

    def define_force_vector(self, element_size_x):
        load = 1/len(self.loading_nodes)
        mx_y = ['-','-','-','-', '+','+','+','+']
        mx_z = ['-','-','+','+', '-','-','+','+']
        my_x = ['+','+','+','+', '-','-','-','-']
        my_z = ['+','-','-','+', '+','-','-','+']
        mz_x = ['+','+','-','-', '+','+','-','-']
        mz_y = ['-','+','+','-', '-','+','+','-']
        for i, node in enumerate(self.loading_nodes):
            start_row = node * self.dof_per_node

            self.F[start_row, 0] = load  # Load in x 
            self.F[start_row+1, 1] = load  # Load in y 
            self.F[start_row+2, 2] = load  # Load in z 

            if mx_y[i]=='+':
                self.F[start_row+1, 3] = load/(element_size_x/2)  # Moment about x 
            elif mx_y[i]=='-':
                self.F[start_row+1, 3] = -load/(element_size_x/2)  # Moment about x 
            if mx_z[i]=='+':
                self.F[start_row+2, 3] = load/(element_size_x/2)  # Moment about x 
            elif mx_z[i]=='-':
                self.F[start_row+2, 3] = -load/(element_size_x/2)  # Moment about x 

            if my_x[i]=='+':
                self.F[start_row, 4] = load/(element_size_x/2)  # Moment about y 
            elif my_x[i]=='-':
                self.F[start_row, 4] = -load/(element_size_x/2)  # Moment about y
            if my_z[i]=='+':
                self.F[start_row+2, 4] = load/(element_size_x/2)  # Moment about y
            elif my_z[i]=='-':
                self.F[start_row+2, 4] = -load/(element_size_x/2)  # Moment about y 

            if mz_x[i]=='+':
                self.F[start_row, 5] = load/(element_size_x/2)  # Moment about z 
            elif mz_x[i]=='-':
                self.F[start_row, 5] = -load/(element_size_x/2)  # Moment about z 
            if mz_y[i]=='+':
                self.F[start_row+1, 5] = load/(element_size_x/2)  # Moment about z
            elif mz_y[i]=='-':
                self.F[start_row+1, 5] = -load/(element_size_x/2)  # Moment about z

            print(f"Force at node {node}:")
            print(self.F[node*3:node*3+3,:])

    def apply_boundary_conditions(self, fix_nodes):
        print("Applying boundary conditions...", flush=True)
        for node in tqdm(fix_nodes):
            start_row = node * self.dof_per_node
            start_col = start_row
            for i in range(self.dof_per_node):
                self.K_sc[start_row+i, :] = 0
                self.K_sc[:, start_col+i] = 0
                self.K_sc[start_row+i, start_col+i] = 1.0
                self.F[start_row+i, :] = 0

    def compute_deformation_matrix(self):
        # Using spsolve for efficient solution
        print("Solving for global nodal deformations vector...", flush=True)
        self.U = np.linalg.solve(self.K_sc,self.F)
        for node in self.loading_nodes:
            print(f"Global nodal deformations at node {node}:")
            print(self.U[node*3:node*3+3,:])

    def define_extraction_matrix(self):
        print("Preparing extraction matrix...", flush=True)
        self.A = np.zeros((self.F.shape[1], self.K_sc.shape[0]))  # Use LIL format for construction
        for i, node in enumerate(self.loading_nodes):
            start_col = node * self.dof_per_node

            self.A[0, start_col] = self.N_func[i](0,0,0)
            self.A[1, start_col + 1] = self.N_func[i](0,0,0)
            self.A[2, start_col + 2] = self.N_func[i](0,0,0)

            self.A[3, start_col + 1] = -self.dN_dzeta_func[i](0,0,0)*self.loading_J_inv_T[2,2]
            self.A[3, start_col + 2] = self.dN_deta_func[i](0,0,0)*self.loading_J_inv_T[1,1]
            self.A[4, start_col] = self.dN_dzeta_func[i](0,0,0)*self.loading_J_inv_T[2,2]
            self.A[4, start_col + 2] = -self.dN_dxi_func[i](0,0,0)*self.loading_J_inv_T[0,0]
            self.A[5, start_col] = -self.dN_deta_func[i](0,0,0)*self.loading_J_inv_T[2,2]
            self.A[5, start_col + 1] = self.dN_dxi_func[i](0,0,0)*self.loading_J_inv_T[0,0]

    def extract_compliance_matrix(self):
        print("Extracting compliance matrix...", flush=True)
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
        self.K_global = self.K_global
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
