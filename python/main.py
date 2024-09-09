import numpy as np
import plotly.express as px

from structure import Structure
from quadrilateral4NodeElement import Quadrilateral4NodeElement
from hexahedral8NodeElement import Hexahedral8NodeElement
from hexahedral20NodeElement import Hexahedral20NodeElement
from subchainFEA import SubchainFEA

structure = Structure()

### Materials Parameters ###
E = 71e9  # Young's modulus in Pa
nu = 0.33   # Poisson's ratio

# loading_point = 612
#loading_point = 2474

#########################################################################################################################
"""
20-Node Quadratic Hexahedral Finite Element
"""
### Create element instance ###
element_type = 'hexahedral20'
structure_type = 'fixed_free_rect_beam' #change here for different structure
if structure_type=='fixed_free_rect_beam':
    num_elements, element_size, dof_per_node, loading_point, support_points, solid_elements = structure.fixed_free_rect_beam()
elif structure_type=='fixed_fixed_rect_beam':
    num_elements, element_size, dof_per_node, loading_point, support_points, solid_elements = structure.fixed_fixed_rect_beam()
elif structure_type=='arch':
    num_elements, element_size, dof_per_node, loading_point, support_points, solid_elements = structure.arch()

structure.fixed_free_rect_beam()
hexahedral20Node = Hexahedral20NodeElement(E, nu, element_size, loading_point, support_points, solid_elements)

### Assemble subchain stiffness matrix ###
hexahedral20Node.assemble_subchain_stiffness_matrix(num_elements)
K_subchain = hexahedral20Node.get_subchain_stiffness_matrix()
loading_nodes, N_func, N_func_diff, loading_J_inv_T = hexahedral20Node.get_loading_point_info()
support_nodes = hexahedral20Node.get_support_point_info()
print(f"Stiffness Matrix of Subchain ({K_subchain.shape}):")
print(K_subchain)
#hexahedral20Node.visualize_design_domain((num_elements[1],num_elements[0]))

### FEA Analysis ###
force_vector = np.zeros((K_subchain.shape[0], 1))
#fix_nodes = [0,1, 676, 677] + [24, 25, 700, 701]
#fix_nodes = [0,1, 2601, 2602] + [49, 50, 2650, 2651]
subchain = SubchainFEA(K_subchain, force_vector, loading_nodes, N_func, N_func_diff, loading_J_inv_T, dof_per_node)
subchain.define_force_vector(element_type, element_size[0], element_size[2])
subchain.apply_boundary_conditions(support_nodes)
subchain.compute_deformation_matrix()
subchain.define_extraction_matrix()

C_extracted = subchain.extract_compliance_matrix().round(10)
print("Loading point compliance matrix, C_6x6:")
print(C_extracted)
print(np.allclose(C_extracted, C_extracted.T, rtol=1e-5, atol=1e-6))

K_extracted = np.linalg.inv(C_extracted).round(10)
print("Loading point stiffness matrix, K_6x6:")
print(K_extracted)
#########################################################################################################################

#########################################################################################################################
"""
4-Node Linear Quadrilateral Finite Element
"""
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
#########################################################################################################################

#########################################################################################################################
"""
8-Node Linear Hexahedral Finite Element
"""
# # Create element instance
# element_type = 'hexahedral8'
# support_point = [0, 49]
# hexahedral8Node = Hexahedral8NodeElement(E, nu, loading_point, support_point, element_size)

# # Assemble subchain stiffness matrix
# hexahedral8Node.assemble_subchain_stiffness_matrix(num_elements)
# K_subchain = hexahedral8Node.get_subchain_stiffness_matrix()
# loading_nodes, N_func, N_func_diff, loading_J_inv_T = hexahedral8Node.get_loading_point_info()
# support_nodes = hexahedral8Node.get_support_point_info()
# print(f"Stiffness Matrix of Subchain ({K_subchain.shape}):")
# print(K_subchain[5176*3:5176*3+3,5176*3:5176*3+3])
# hexahedral8Node.visualize_design_domain((num_elements_y,num_elements_x))

# force_vector = np.zeros((K_subchain.shape[0], 6))
# #fix_nodes = [0,1, 676, 677] + [24, 25, 700, 701]
# fix_nodes = [0,1, 2601, 2602] + [49, 50, 2650, 2651]
# subchain = SubchainFEA(K_subchain, force_vector, loading_nodes, N_func, N_func_diff, loading_J_inv_T, dof_per_node)
# subchain.define_force_vector(element_type, element_size_x)
# subchain.apply_boundary_conditions(fix_nodes)
# subchain.compute_deformation_matrix()
# subchain.define_extraction_matrix()

# C_extracted = subchain.extract_compliance_matrix().round(6)
# print("Loading point compliance matrix, C_6x6:")
# print(C_extracted)
# print(np.allclose(C_extracted, C_extracted.T, rtol=1e-5, atol=1e-6))

# K_extracted = np.linalg.inv(C_extracted).round(6)
# print("Loading point stiffness matrix, K_6x6:")
# print(K_extracted)

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
#########################################################################################################################