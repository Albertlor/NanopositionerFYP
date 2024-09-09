class Structure:
    def __init__(self):
        pass

    def fixed_free_rect_beam(self):
        #  ||  #
        #  ||  #
        #  ||  #
        # ==== #
        num_elements_x = 1
        num_elements_y = 201
        num_elements_z = 1
        num_elements = [num_elements_x, num_elements_y, num_elements_z]

        element_size_x = 0.02
        element_size_y = 0.002
        element_size_z = 0.002
        element_size = [element_size_x, element_size_y, element_size_z]

        dof_per_node = 3

        loading_point = 200 #element of the end
        support_points = [0]

        solid_elements = list(range(0,201)) #solid elements within the design domain

        return [num_elements, element_size, dof_per_node, loading_point, support_points, solid_elements]
    
    def arch(self):
        #      ||      #
        # |==========| #
        # |          | #
        # |          | #
        num_elements_x = 25
        num_elements_y = 25
        num_elements_z = 1
        num_elements = [num_elements_x, num_elements_y, num_elements_z]

        element_size_x = 0.002
        element_size_y = 0.002
        element_size_z = 0.02
        element_size = [element_size_x, element_size_y, element_size_z]

        dof_per_node = 3

        loading_point = 612 #middle element of the top row
        support_points = [0, 24]

        solid_elements = list(range(0,550,25))+list(range(550,575))+list(range(24,574,25)) + [575+12,600+12] #solid elements within the design domain

        return [num_elements, element_size, dof_per_node, loading_point, support_points, solid_elements]
