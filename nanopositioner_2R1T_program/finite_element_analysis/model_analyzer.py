import numpy as np

from scipy.sparse import csr_matrix
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
        # ### Using spsolve for efficient solution
        # K_sc_sparse = csr_matrix(self.K_sc)
        # Ux, info_x = cg(K_sc_sparse, self.F[:, 0])
        # Uy, info_y = cg(K_sc_sparse, self.F[:, 1])
        # Uz, info_z = cg(K_sc_sparse, self.F[:, 2])

        # ### Check if all solvers converged successfully
        # if info_x == info_y == info_z == 0:
        #     print("Conjugate gradient solver converged successfully for all components.")
        # else:
        #     print("Solver did not converge for some components.")
        # U = np.column_stack((Ux, Uy, Uz))
        # self.U = np.array(U.reshape(self.F.shape))
        self.U = np.linalg.solve(self.K_sc,self.F)
        for node in self.global_loading_nodes:
            print(f"Global nodal deformations at node {node}:")
            print(self.U[node*3:node*3+3,:])

    def define_extraction_matrix(self):
        print("Preparing extraction matrix...", flush=True)
        c1, c2, c3 = self.loading_center
        self.A = np.zeros((self.F.shape[1], self.K_sc.shape[0]))
        for i, node in enumerate(self.global_loading_nodes):
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
        print("Extracting compliance matrix...", flush=True)
        self.C_extracted = self.A @ self.U
        return self.C_extracted.round(9)