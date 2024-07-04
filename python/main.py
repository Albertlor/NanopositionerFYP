from EightNodeHexahedral import Hexahedral8NodeElement
from TwentyNodeHexahedral import Hexahedral20NodeElement


# Parameters
E = 71e9  # Young's modulus in Pa
nu = 0.33   # Poisson's ratio


# Number of elements in each direction
num_elements_x = 50
num_elements_y = 50
num_elements_z = 1

element_size_x = 2
element_size_y = 2
element_size_z = 20

"""
8-Node Linear Hexahedral Finite Element
"""
# # Create element instance
# hexahedral8Node = Hexahedral8NodeElement(E, nu)

# # Assemble global stiffness matrix
# hexahedral8Node.assemble_global_stiffness_matrix(num_elements_x, num_elements_y, num_elements_z, element_size_x, element_size_y, element_size_z)
# K_global_np = hexahedral8Node.get_global_stiffness_matrix()

# print(f"Global Stiffness Matrix ({K_global_np.shape}):")
# print(K_global_np)


"""
20-Node Quadratic Hexahedral Finite Element
"""
# Create element instance
hexahedral20Node = Hexahedral20NodeElement(E, nu)

# Assemble global stiffness matrix
hexahedral20Node.assemble_global_stiffness_matrix(num_elements_x, num_elements_y, num_elements_z, element_size_x, element_size_y, element_size_z)
K_global_np = hexahedral20Node.get_global_stiffness_matrix()

print(f"Global Stiffness Matrix ({K_global_np.shape}):")
print(K_global_np)