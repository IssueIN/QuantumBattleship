import numpy as np
import matplotlib.pyplot as plt
from src.solver import TDSESolver

class Wavefunction:
    def __init__(self, x, y, wf_xy0, V_xy, hbar, m):
        self.x = x
        self.y = y
        self.X, self.Y = np.meshgrid(x, y)
        self.wf_xy0 = wf_xy0

        self.potentials = [V_xy]
        self.V_xy = V_xy

        self.hbar = hbar
        self.m = m
        self.Nx = len(x)
        self.Ny = len(y)

        self.Nx = len(x)
        self.dx = self.x[1] - self.x[0]
        self.dkx = 2 * np.pi / (self.Nx * self.dx)
        self.Ny = len(y)
        self.dy = self.y[1] - self.y[0]
        self.dky = 2 * np.pi / (self.Ny * self.dy)

        self.solver_x = [TDSESolver(y, wf_xy0[i, :], V_xy[i, :], hbar, m) for i in range(self.Nx)]
        self.solver_y = [TDSESolver(x, wf_xy0[:, j], V_xy[:, j], hbar, m) for j in range(self.Ny)]
    
    def solve(self, dt, Nsteps=1):
        for solver in self.solver_x:
            solver.solve(dt/2, Nsteps)
        for solver in self.solver_y:
            solver.solve(dt, Nsteps)
        for solver in self.solver_x:
            solver.solve(dt/2, Nsteps)
    
    def norm(self, wf):
        # return np.sqrt((abs(wf) ** bn2).sum() * 2 * np.pi / (self.dx * self.dy))
        return np.sqrt((abs(wf) ** 2).sum() * self.dx * self.dy)
    
    def get_wf_xy(self):
        wf_xy = np.zeros((self.Nx, self.Ny), dtype=complex)
        for i in range(self.Nx):
            wf_xy[i, :] = self.solver_x[i].wf_x
        for j in range(self.Ny):
            wf_xy[:, j] *= self.solver_y[j].wf_x
        wf_xy /= self.norm(wf_xy)
        return wf_xy
    
    def measure(self, wf, Area):
        x_min, x_max, y_min, y_max = Area
        mask_x = (self.x >= x_min) & (self.x <= x_max)
        mask_y = (self.y >= y_min) & (self.y <= y_max)
        mask = np.outer(mask_x, mask_y)
        wf_measured = np.copy(wf)
        wf_measured[~mask] = 0
        wf_measured /= self.norm(wf_measured)
        return wf_measured

    def appendV(self, V_new):
        """
        Adds a new potential to the existing potential.

        Parameters:
        -----------
        V_new : tuple
            A tuple in the form (x_min, x_max, y_min, y_max, potential_value) specifying
            the new potential to be added.
        """
        x_min, x_max, y_min, y_max, potential_value = V_new
        mask_x = (self.x >= x_min) & (self.x <= x_max)
        mask_y = (self.y >= y_min) & (self.y <= y_max)
        mask = np.outer(mask_x, mask_y)
        V_add = np.zeros_like(self.V_xy)
        V_add[mask] = potential_value

        self.potentials.append(V_add)
        self.V_xy += V_add
        self.solver_x = [TDSESolver(self.y, self.wf_xy0[i, :], self.V_xy[i, :], self.hbar, self.m) for i in range(self.Nx)]
        self.solver_y = [TDSESolver(self.x, self.wf_xy0[:, j], self.V_xy[:, j], self.hbar, self.m) for j in range(self.Ny)]


    def deleteV(self, index):
        if 0 <= index < len(self.potentials):
            self.V_xy -= self.potentials[index]
            del self.potentials[index]
            self.solver_x = [TDSESolver(self.y, self.wf_xy0[i, :], self.V_xy[i, :], self.hbar, self.m) for i in range(self.Nx)]
            self.solver_y = [TDSESolver(self.x, self.wf_xy0[:, j], self.V_xy[:, j], self.hbar, self.m) for j in range(self.Ny)]  

    def V_map(self):
        """
        Plots the potential values as a heatmap.
        """
        total_potential = sum(self.potentials)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(total_potential, extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
                        origin='lower', aspect='auto', cmap='viridis')
        fig.colorbar(cax, ax=ax, label='Potential')
        ax.set_title('Heatmap of the Potential')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        return fig



