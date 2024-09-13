class Timoshenko_Beam:
    def __init__(self):
        self.E = 71E9
        self.v = 0.33
        self.G = self.E/(2*(1+self.v))
        self.k = 5/6 ### Shear correction factor
        self.L = 400E-3
        self.W = 20E-3
        self.D = 20E-3
        self.Iz = 13333.3E-12
        self.Ix = 13333.3E-12
        self.J = (self.W*self.D) * (self.W**2 + self.D**2) / 12
        self.F = 1
        self.M = 1

    def fixed_free_end_point(self):
        c11 = (self.F*self.L**3) / (3*self.E*self.Iz)
        c22 = (self.F*self.L) / ((self.W*self.D)*self.E)
        c61 = (self.F*self.L**2) / (2*self.E*self.Iz) + self.L / (self.k*self.G*(self.W*self.D))

        return c11, c22, c61
    
    def fixed_fixed_mid_point(self):
        delta_x = (self.F*self.L**3) / (192*self.E*self.Iz)

        return delta_x
    

if __name__=='__main__':
    beam = Timoshenko_Beam()
    c11, c22, c61 = beam.fixed_free_end_point()
    print(f"x: {c11}, y:{c22}, slope_z: {c61}")