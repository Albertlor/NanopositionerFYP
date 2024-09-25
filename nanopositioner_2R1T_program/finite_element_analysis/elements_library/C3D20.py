import numpy as np
import sympy as sp


class Hexahedral20NodeElement:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
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
        return J.round(9)

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
                    J = self.construct_Jacobian_matrix(xi, eta, zeta, nodes_physical)
                    J_inv = np.linalg.inv(J).round(9)
                    J_inv_T = J_inv.T
                    J_det = np.linalg.det(J)
                    B = self.construct_B_matrix(xi, eta, zeta, J_inv_T)
                    w = gauss_weights[i] * gauss_weights[j] * gauss_weights[k]
                    K_FE += w * B.T @ self.D @ B * J_det
        K_FE = K_FE.round(7)
        return K_FE
    
    def get_jacobian_matrix(self, coors, nodes_physical):
        J = self.construct_Jacobian_matrix(coors[0], coors[1], coors[2], nodes_physical)
        #J_inv = np.linalg.inv(J).round(9)
        return J
    
    def get_shape_functions(self):
        return self.N_func, self.dN_dxi_func, self.dN_deta_func, self.dN_dzeta_func