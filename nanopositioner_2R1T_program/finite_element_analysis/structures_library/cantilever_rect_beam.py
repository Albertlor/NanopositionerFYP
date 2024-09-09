class Cantilever_Rect_Beam:
    def __init__(self):
        self.supporting_points = [0]
        self.local_supporting_nodes = [0,1,4,5,8,12,16,17]
        self.loading_points = [199]
        self.local_loading_nodes = [2,3,6,7,10,14,18,19]

    def create_structure(self):
        #  ||  #
        #  ||  #
        #  ||  #
        # ==== #
        num_elements_x = 1
        num_elements_y = 200
        num_elements_z = 1
        num_elements = [num_elements_x, num_elements_y, num_elements_z]

        element_size_x = 0.02
        element_size_y = 0.002
        element_size_z = 0.002
        element_size = [element_size_x, element_size_y, element_size_z]

        dof_per_node = 3

        solid_elements = list(range(0,200)) #solid elements within the design domain

        support_info = [self.supporting_points, self.local_supporting_nodes]
        loading_info = [self.loading_points, self.local_loading_nodes]

        return [num_elements, element_size, dof_per_node, solid_elements, support_info, loading_info]