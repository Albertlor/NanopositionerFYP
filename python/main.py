import numpy as np
import plotly.express as px

from quadrilateral4NodeElement import Quadrilateral4NodeElement
from hexahedral8NodeElement import Hexahedral8NodeElement
from hexahedral20NodeElement import Hexahedral20NodeElement
from subchainFEA import SubchainFEA

# Parameters
E = 71e9  # Young's modulus in Pa
nu = 0.33   # Poisson's ratio

# Number of elements in each direction
num_elements_x = 50
num_elements_y = 50
num_elements_z = 1

element_size_x = 0.002
element_size_y = 0.002
element_size_z = 0.02

dof_per_node = 3

# """
# 4-Node Linear Quadrilateral Finite Element
# """
# # Create element instance
# quadrilateral4Node = Quadrilateral4NodeElement(E, nu)

# # Assemble subchain stiffness matrix
# quadrilateral4Node.assemble_subchain_stiffness_matrix(num_elements_x, num_elements_y, element_size_x, element_size_y)
# K_subchain = quadrilateral4Node.get_subchain_stiffness_matrix()

# print(f"Stiffness Matrix of Subchain ({K_subchain.shape}):")
# print(K_subchain[55*24:55*24+24,55*24:55*24+24])

# force_vector = np.zeros((K_subchain.shape[0], 6))
# loading_point = 2575  # node subjected to the loading, have to choose two to describe the six wrenches
# fix_nodes = [0,1,52,51] + [49, 50, 101, 100]
# subchain = SubchainFEA(K_subchain, force_vector, dof_per_node)
# subchain.define_extraction_matrix(loading_point)
# subchain.define_force_vector(loading_point)
# subchain.compute_deformation_matrix()
# #subchain.apply_boundary_conditions(fix_nodes)
# C_extracted = subchain.extract_compliance_matrix()

# # Condition the compliance matrix to avoid singularity
# epsilon = 1e-8
# C_extracted += np.eye(C_extracted.shape[0]) * epsilon

# K_extracted = np.linalg.inv(C_extracted)
# print(C_extracted)
# print(K_extracted)

# # Check for symmetry and positive definiteness
# if not np.allclose(C_extracted, C_extracted.T):
#     print("Compliance matrix is not symmetric!")
# else:
#     print("Compliance matrix is symmetric.")

# subchain.assemble_global_stiffness_matrix(K_extracted)
# K_global = subchain.get_global_stiffness_matrix()
# print(K_global)


"""
8-Node Linear Hexahedral Finite Element
"""
# Create element instance
hexahedral8Node = Hexahedral8NodeElement(E, nu)

# Assemble subchain stiffness matrix
hexahedral8Node.assemble_subchain_stiffness_matrix(num_elements_x, num_elements_y, num_elements_z, element_size_x, element_size_y, element_size_z)
K_subchain = hexahedral8Node.get_subchain_stiffness_matrix()
loading_nodes, N_func, dN_dxi_func, dN_deta_func, dN_dzeta_func, loading_J_inv_T = hexahedral8Node.get_loading_point_info()
print(f"Stiffness Matrix of Subchain ({K_subchain.shape}):")
print(K_subchain[5176*3:5176*3+3,5176*3:5176*3+3])

force_vector = np.zeros((K_subchain.shape[0], 6))
fix_nodes = [0,1, 2601, 2602] + [49, 50, 2650, 2651]
subchain = SubchainFEA(K_subchain, force_vector, loading_nodes, N_func, dN_dxi_func, dN_deta_func, dN_dzeta_func, loading_J_inv_T, dof_per_node)
subchain.define_force_vector(element_size_x)
subchain.apply_boundary_conditions(fix_nodes)
subchain.compute_deformation_matrix()
subchain.define_extraction_matrix()

C_extracted = subchain.extract_compliance_matrix().round(6)
print("Loading point compliance matrix, C_6x6:")
print(C_extracted)
print(np.allclose(C_extracted, C_extracted.T, rtol=1e-5, atol=1e-6))

K_extracted = np.linalg.inv(C_extracted).round(6)
print("Loading point stiffness matrix, K_6x6:")
print(K_extracted)

# # Condition the compliance matrix to avoid singularity
# epsilon = 1e-8
# C_extracted += np.eye(C_extracted.shape[0]) * epsilon

# K_extracted = np.linalg.inv(C_extracted)
# print(C_extracted)
# print(K_extracted)

# # Check for symmetry and positive definiteness
# if not np.allclose(C_extracted, C_extracted.T):
#     print("Compliance matrix is not symmetric!")
# else:
#     print("Compliance matrix is symmetric.")

# subchain.assemble_global_stiffness_matrix(K_extracted)
# K_global = subchain.get_global_stiffness_matrix()
# print(K_global)

# def visualize_matrix_interactive(matrix):
#     fig = px.imshow(matrix, color_continuous_scale='viridis', title='Matrix Visualization')
#     fig.show()