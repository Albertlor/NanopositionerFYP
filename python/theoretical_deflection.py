class TheoreticalDeflection:
    def __init__(self):
        self.E = 71E9
        self.v = 0.33
        self.G = self.E/(2*(1+self.v))
        self.L = 402E-3
        self.W = 2E-3
        self.D = 2E-3
        self.Iz = 1333.33E-12
        self.Ix = 1.33333E-12
        self.J = (self.W*self.D) * (self.W**2 + self.D**2) / 12
        self.F = 1
        self.M = 1
        

    def fixed_free_end_point(self):
        delta_x = (self.F*self.L**3) / (3*self.E*self.Iz)
        delta_y = (self.F*self.L) / ((self.W*self.D)*self.E) + ((self.F*self.W/2)*self.L**2) / (2*self.E*self.Iz)
        delta_z = (self.F*self.L**3) / (3*self.E*self.Ix)
        
        theta_x = (self.M*self.L) / (self.E*self.Ix)
        theta_y = (self.M*self.L) / (self.J*self.G)
        theta_z = (self.M*self.L) / (self.E*self.Iz)

        return [delta_x, delta_y, delta_z, theta_x, theta_y, theta_z]
    
    def fixed_free_mid_point(self):
        delta_x = (self.F*self.L**3) / (24*self.E*self.Iz)
        delta_y = (self.F*self.L) / (2*(self.W*self.D)*self.E) + ((self.F*self.W/2)*self.L**2) / (8*self.E*self.Iz)
        # delta_y = (self.L/(2*self.E)) * (self.F/(self.W*self.D) + (self.M*(self.W/2))/self.Iz)
        delta_z = (self.F*self.L**3) / (24*self.E*self.Ix)
        
        theta_x = (self.M*self.L) / (2*self.E*self.Ix)
        theta_y = (self.M*self.L) / (2*self.J*self.G)
        theta_z = (self.M*self.L) / (2*self.E*self.Iz)

        return [delta_x, delta_y, delta_z, theta_x, theta_y, theta_z]
    
    def fixed_free_25_percent_from_end_point(self):
        delta_x = (self.F*(3*self.L/4)**3) / (3*self.E*self.Iz)
        delta_y = (self.F*(3*self.L/4)) / ((self.W*self.D)*self.E) + ((self.F*self.W/2)*(3*self.L/4)**2) / (2*self.E*self.Iz)
        delta_z = (self.F*(3*self.L/4)**3) / (3*self.E*self.Ix)
        
        theta_x = (self.M*(3*self.L/4)) / (self.E*self.Ix)
        theta_y = (self.M*(3*self.L/4)) / (self.J*self.G)
        theta_z = (self.M*(3*self.L/4)) / (self.E*self.Iz)

        return [delta_x, delta_y, delta_z, theta_x, theta_y, theta_z]
    
    def fixed_free_75_percent_from_end_point(self):
        delta_x = (self.F*(self.L/4)**3) / (3*self.E*self.Iz)
        delta_y = (self.F*(self.L/4)) / ((self.W*self.D)*self.E) + ((self.F*self.W/2)*(self.L/4)**2) / (2*self.E*self.Iz)
        delta_z = (self.F*(self.L/4)**3) / (3*self.E*self.Ix)
        
        theta_x = (self.M*(self.L/4)) / (self.E*self.Ix)
        theta_y = (self.M*(self.L/4)) / (self.J*self.G)
        theta_z = (self.M*(self.L/4)) / (self.E*self.Iz)

        return [delta_x, delta_y, delta_z, theta_x, theta_y, theta_z]
    
if __name__=='__main__':
    deflect = TheoreticalDeflection()
    delta_x, delta_y, delta_z, theta_x, theta_y, theta_z = deflect.fixed_free_end_point()
    print(f"x: {delta_x}, y: {delta_y}, z:{delta_z}")
    print(f"theta_x: {theta_x}, theta_y: {theta_y}, theta_z:{theta_z}")