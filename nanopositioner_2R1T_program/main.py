import numpy as np
import time

from finite_element_analysis.structures_library import cantilever_rect_beam, arch, fixed_fixed_rect_beam
from finite_element_analysis.elements_library import C3D20
from finite_element_analysis.model_generator import Model_Generator
from finite_element_analysis.model_analyzer import Model_Analyzer


start_time = time.perf_counter()
######################################################################################################################
"""
Define Materials Properties
"""
E = 71E9
nu = 0.33 # Poisson ratio 
######################################################################################################################

######################################################################################################################
"""
Define Structural Geometry
"""
rect_beam = cantilever_rect_beam.Cantilever_Rect_Beam()
structure_info = rect_beam.create_structure()
# arch_structure = arch.Arch()
# structure_info = arch_structure.create_structure()
# fixed_fixed_beam = fixed_fixed_rect_beam.Fixed_Fixed_Rect_Beam()
# structure_info = fixed_fixed_beam.create_structure()

num_ele_x, num_ele_y, num_ele_z = structure_info[0] # Number of elements
x, y, z = structure_info[1] # Element size
element_size = [x, y, z]
dof_per_node = structure_info[2]
solid_elements = structure_info[3]
supporting_point, local_supporting_nodes = structure_info[4]
loading_point, local_loading_nodes = structure_info[5]
loading_center = (0,1,0)

nodes_physical = np.array([
    [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
    [0, 0, z], [x, 0, z], [x, y, z], [0, y, z],
    [x/2, 0, 0], [x, y/2, 0], [x/2, y, 0], [0, y/2, 0],
    [x/2, 0, z], [x, y/2, z], [x/2, y, z], [0, y/2, z],
    [0, 0, z/2], [x, 0, z/2], [x, y, z/2], [0, y, z/2]
])
######################################################################################################################

######################################################################################################################
"""
Create Element
"""
hexahedral20 = C3D20.Hexahedral20NodeElement(E, nu)
K_FE = hexahedral20.compute_element_stiffness_matrix(nodes_physical)
load_J_inv = hexahedral20.get_jacobian_matrix(loading_center, nodes_physical) # Jacobian matrix for loading element
N_func, dN_dxi_func, dN_deta_func, dN_dzeta_func = hexahedral20.get_shape_functions()
N_func = [N_func[i] for i in local_loading_nodes] # Shape functions for loading nodes
N_func_diff = [
    [dN_dxi_func[i] for i in local_loading_nodes], 
    [dN_deta_func[j] for j in local_loading_nodes],
    [dN_dzeta_func[k] for k in local_loading_nodes]
] # Derivative of shape functions for loading nodes
######################################################################################################################

######################################################################################################################
"""
Generate Meshed Model
"""
model = Model_Generator(
            K_FE, 
            [num_ele_x, num_ele_y, num_ele_z], 
            [x, y, z], 
            supporting_point, local_supporting_nodes, 
            loading_point, local_loading_nodes, 
            solid_elements
        )
model.assemble_subchain_stiffness_matrix()
K_sc = model.get_subchain_stiffness_matrix()
global_supporting_nodes = model.get_boundary_conditions()
global_loading_nodes, connectivity = model.get_loading_info()
print(K_sc)
print(K_sc.shape)
######################################################################################################################

######################################################################################################################
"""
Evaluate the model with unit loading
"""
fea_analysis = Model_Analyzer(
                    K_sc,
                    loading_center,
                    global_supporting_nodes,
                    global_loading_nodes,
                    N_func,
                    N_func_diff,
                    load_J_inv,
                    dof_per_node
                )
fea_analysis.define_force_vector()
fea_analysis.apply_boundary_conditions()
U = fea_analysis.compute_deformation_matrix()

print("Preparing extraction matrix...", flush=True)
fea_analysis.define_extraction_matrix()

print("Extracting compliance matrix...", flush=True)
C_extracted = fea_analysis.extract_compliance_matrix()
K_extracted = np.linalg.inv(C_extracted)
print(f"6x6 Compliance Matrix at Loading Point:")
print(C_extracted)
print(f"6x6 Stiffness Matrix at Loading Point:")
print(K_extracted)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds",flush=True)

# K_global = fea_analysis.assemble_global_stiffness_matrix(K_extracted)
# print(f"Global K:")
# print(K_global)
######################################################################################################################

######################################################################################################################
"""
Visualization of FEA Result for Displacements
"""
element_of_interest = list(range(200))
displacement_dict = {}
for element in element_of_interest:
    nodes_of_interest = [connectivity[element][i] for i in local_loading_nodes]
    fea_analysis.define_extraction_matrix(nodes_of_interest)

    C_extracted = fea_analysis.extract_compliance_matrix()
    displacement_dict[element] = C_extracted[0,0]
fea_analysis.displacement_visualization(displacement_dict)
######################################################################################################################