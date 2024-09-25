class Euler_Bernoulli_Beam:
    def __init__(self):
        self.E = 71E9
        self.v = 0.33
        self.G = self.E/(2*(1+self.v))
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
        c55 = (self.M*self.L) / (self.J*self.G)
        c61 = (self.F*self.L**2) / (2*self.E*self.Iz)
        c66 = (self.M*self.L) / (self.E*self.Iz)

        return c11, c22, c55, c61, c66
    
    def fixed_fixed_mid_point(self):
        delta_x = (self.F*self.L**3) / (192*self.E*self.Iz)

        return delta_x
    

if __name__=='__main__':
    beam = Euler_Bernoulli_Beam()
    c11, c22, c55, c61, c66 = beam.fixed_free_end_point()
    print(f"c11: {c11}, c22:{c22}, c55:{c55}, c61: {c61}, c66: {c66}")