class Arch:
    def __init__(self):
        self.supporting_points = [0, 24]
        self.local_supporting_nodes = [0,1,4,5,8,12,16,17]
        self.loading_points = [612]
        self.local_loading_nodes = list(range(20))
        # self.local_loading_nodes = [2,3,6,7,10,14,18,19]

    def create_structure(self):
        #  ||  #
        #  ||  #
        #  ||  #
        # ==== #
        num_elements_x = 25
        num_elements_y = 25
        num_elements_z = 1
        num_elements = [num_elements_x, num_elements_y, num_elements_z]

        element_size_x = 0.002
        element_size_y = 0.002
        element_size_z = 0.02
        element_size = [element_size_x, element_size_y, element_size_z]

        dof_per_node = 3

        solid_elements = list(range(0,550,25))+list(range(550,575))+list(range(24,574,25)) + [575+12,600+12] #solid elements within the design domain

        support_info = [self.supporting_points, self.local_supporting_nodes]
        loading_info = [self.loading_points, self.local_loading_nodes]

        return [num_elements, element_size, dof_per_node, solid_elements, support_info, loading_info]

    # def create_structure(self):
    #     #  ||  #
    #     #  ||  #
    #     #  ||  #
    #     # ==== #
    #     num_elements_x = 51
    #     num_elements_y = 51
    #     num_elements_z = 1
    #     num_elements = [num_elements_x, num_elements_y, num_elements_z]

    #     element_size_x = 0.001
    #     element_size_y = 0.001
    #     element_size_z = 0.02
    #     element_size = [element_size_x, element_size_y, element_size_z]

    #     dof_per_node = 3

    #     solid_elements = list(range(0,2448,51))+list(range(2448,2499))+list(range(50,2499,51)) + [2500+25,2551+25] #solid elements within the design domain

    #     support_info = [self.supporting_points, self.local_supporting_nodes]
    #     loading_info = [self.loading_points, self.local_loading_nodes]

    #     return [num_elements, element_size, dof_per_node, solid_elements, support_info, loading_info]