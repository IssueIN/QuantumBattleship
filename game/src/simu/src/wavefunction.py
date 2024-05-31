import numpy as np
from simu.src.solver import TDSESolver

class Wavefunction:
    def __init__(self, x, y, wf_xy0, V_xy, hbar, m):
        self.x = x
        self.y = y
        self.X, self.Y = np.meshgrid(x, y)
        self.wf_xy0 = wf_xy0
        self.V_xy = V_xy
        self.hbar = hbar
        self.m = m
        self.Nx = len(x)
        self.Ny = len(y)

        self.solver_x = [TDSESolver(y, wf_xy0[i, :], V_xy[i, :], hbar, m) for i in range(self.Nx)]
        self.solver_y = [TDSESolver(x, wf_xy0[:, j], V_xy[:, j], hbar, m) for j in range(self.Ny)]
    
    def solve(self, dt, Nsteps=1):
        for solver in self.solver_x:
            solver.solve(dt/2, Nsteps)
        for solver in self.solver_y:
            solver.solve(dt, Nsteps)
        for solver in self.solver_x:
            solver.solve(dt/2, Nsteps)

    # def get_wf_xy(self):
    #     wf_x_part = np.array([solver.wf_x for solver in self.solver_x])
    #     wf_y_part = np.array([solver.wf_x for solver in self.solver_y])
    #     return wf_x_part, wf_y_part
    def get_wf_xy(self):
        wf_xy = np.zeros((self.Nx, self.Ny), dtype=complex)
        for i in range(self.Nx):
            wf_xy[i, :] = self.solver_x[i].wf_x
        for j in range(self.Ny):
            wf_xy[:, j] *= self.solver_y[j].wf_x
        return wf_xy


