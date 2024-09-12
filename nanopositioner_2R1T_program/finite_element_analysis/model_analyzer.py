import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
from tqdm import tqdm


class Model_Analyzer:
    def __init__(self, K_sc, loading_center, global_supporting_nodes, global_loading_nodes, N_func, N_func_diff, load_J_inv, dof_per_node):
        self.K_sc = K_sc
        self.loading_center = loading_center
        self.global_supporting_nodes = global_supporting_nodes
        self.global_loading_nodes = global_loading_nodes  
        self.N_func = N_func
        self.dN_dxi_func = N_func_diff[0]
        self.dN_deta_func = N_func_diff[1]
        self.dN_dzeta_func = N_func_diff[2]
        self.load_J_inv = load_J_inv
        self.dof_per_node = dof_per_node

    def define_force_vector(self):
        self.F = np.zeros((self.K_sc.shape[0], 6))
        c1, c2, c3 = self.loading_center
        f = [N(c1, c2, c3) for N in self.N_func]
        print(self.global_loading_nodes)
        for i, node in enumerate(self.global_loading_nodes):
            start_row = node * self.dof_per_node

            self.F[start_row, 0] = f[i]  # Load in x 
            self.F[start_row+1, 1] = f[i]
            self.F[start_row+2, 2] = f[i]

            self.F[start_row+1, 3] = -1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,2] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,2] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,2])
            self.F[start_row+2, 3] = 1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,1] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,1] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,1])

            self.F[start_row, 4] = 1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,2] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,2] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,2])
            self.F[start_row+2, 4] = -1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,0] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,0] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,0])

            self.F[start_row, 5] = -1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,1] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,1] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,1])
            self.F[start_row+1, 5] = 1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,0] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,0] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,0])

            print(f"Force at node {node}:")
            print(self.F[node*3:node*3+3,:])

    def apply_boundary_conditions(self):
        print("Applying boundary conditions...", flush=True)
        for node in tqdm(self.global_supporting_nodes):
            start_row = node * self.dof_per_node
            start_col = start_row
            for i in range(self.dof_per_node):
                self.K_sc[start_row+i, :] = 0
                self.K_sc[:, start_col+i] = 0
                self.K_sc[start_row+i, start_col+i] = 1
                self.F[start_row+i, :] = 0

    def compute_deformation_matrix(self):
        print("Solving for global nodal deformations vector...", flush=True)
        self.K_sc_sparse = csr_matrix(self.K_sc.round(6))
        # for i in range(6):
        #     U, info = cg(self.K_sc_sparse, self.F[:,i])
        #     if i==0:
        #         self.U = U
        #     else:
        #         self.U = np.hstack((self.U,U))
        self.U = spsolve(self.K_sc_sparse, self.F)
        #self.U = np.linalg.solve(self.K_sc,self.F)
        for node in self.global_loading_nodes:
            print(f"Global nodal deformations at node {node}:")
            print(self.U[node*3:node*3+3,:])
        return self.U

    def define_extraction_matrix(self,node_of_interest=None):
        if node_of_interest==None:
            node_of_interest = self.global_loading_nodes
        
        c1, c2, c3 = self.loading_center
        self.A = np.zeros((self.F.shape[1], self.K_sc.shape[0]))
        for i, node in enumerate(node_of_interest):
            start_col = node * self.dof_per_node

            self.A[0, start_col] = self.N_func[i](c1, c2, c3)
            self.A[1, start_col+1] = self.N_func[i](c1, c2, c3)
            self.A[2, start_col+2] = self.N_func[i](c1, c2, c3)

            self.A[3, start_col + 1] = -1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,2] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,2] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,2])
            self.A[3, start_col + 2] = 1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,1] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,1] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,1])

            self.A[4, start_col] = 1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,2] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,2] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,2])
            self.A[4, start_col+2] = -1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,0] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,0] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,0])

            self.A[5, start_col] = -1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,1] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,1] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,1])
            self.A[5, start_col+1] = 1/2*(self.dN_dxi_func[i](c1, c2, c3)*self.load_J_inv[0,0] + self.dN_deta_func[i](c1, c2, c3)*self.load_J_inv[1,0] + self.dN_dzeta_func[i](c1, c2, c3)*self.load_J_inv[2,0])

    def extract_compliance_matrix(self):
        self.C_extracted = self.A @ self.U
        return self.C_extracted.round(9)
    
    def assemble_global_stiffness_matrix(self, K_extracted):
        K_global = np.zeros((K_extracted.shape))
        p = 0.06

        theta1_z = np.pi/6
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

        theta2_z = 5*np.pi/6
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

        theta3_z = 3*np.pi/2
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
            K_global += (J_inv.T @ K_extracted @ J_inv)
        
        return K_global
    
    def displacement_visualization(self, displacement_dict):
        # Extract the x and y values from the dictionary
        x = list(displacement_dict.keys())   # Get the keys as x-values
        y = list(displacement_dict.values())  # Get the values as y-values

        # Plotting the data
        plt.plot(x, y, marker='o')  # You can use plt.scatter(x, y) for a scatter plot

        # Adding titles and labels
        plt.title("Displacement of Each Element of a Beam under Loading of 1N")
        plt.xlabel("Element")
        plt.ylabel("Displacement in x Direction (m)")

        # Set fixed axis limits
        plt.ylim([-1E-5, 1E-5])  # Set y-axis limits (example: from 0 to 25)

        plt.gca().invert_yaxis()

        # Display the graph
        plt.show()
        