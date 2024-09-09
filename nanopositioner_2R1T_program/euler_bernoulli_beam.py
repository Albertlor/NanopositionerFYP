class Euler_Bernoulli_Beam:
    def __init__(self):
        self.E = 71E9
        self.v = 0.33
        self.G = self.E/(2*(1+self.v))
        self.L = 400E-3
        self.W = 2E-3
        self.D = 20E-3
        self.Iz = 1333.33E-12
        self.Ix = 1.33333E-12
        self.J = (self.W*self.D) * (self.W**2 + self.D**2) / 12
        self.F = 1
        self.M = 1

    def fixed_free_end_point(self):
        delta_x = (self.F*self.L**3) / (3*self.E*self.Iz)

        return delta_x
    

if __name__=='__main__':
    beam = Euler_Bernoulli_Beam()
    delta_x = round(beam.fixed_free_end_point(), 9)
    print(f"x: {delta_x}")