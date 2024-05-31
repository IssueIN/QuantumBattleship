import numpy as np
from scipy import fftpack

class TDSESolver:
    def __init__(self, x, wf_x0, V_x, hbar, m, k=None):
        self.x, wf_x0, self.V_x = map(np.asarray, (x, wf_x0, V_x))

        self.hbar = hbar
        self.m = m
        self.N = len(x)
        self.dx = self.x[1] - self.x[0]
        self.dk = 2 * np.pi / (self.N * self.dx)

        if k is None:
            self.k0 = -0.5 * self.N * self.dk
        else:
            self.k0 = k
        self.k = self.k0 + self.dk * np.arange(self.N)

        self.wf_x = wf_x0
        self._fft()

        # Vars
        self.dt_ = None
        self.wf_x_evolve_half = None
        self.wf_x_evolve = None
        self.wf_k_evolve = None

    def _mod_wf_x(self, wf_x):
        """
        Handling boundary condition of fft
        """
        self.mod_wf_x = (wf_x * np.exp(-1j * self.k[0] * self.x) * self.dx / np.sqrt(2 * np.pi))
        self.mod_wf_x /= self._norm(self.mod_wf_x)
        self._fft()

    def _get_wf_x(self):
        return (self.mod_wf_x * np.exp(1j * self.k[0] * self.x) * np.sqrt(2 * np.pi) / self.dx)

    def _mod_wf_k(self, wf_k):
        self.mod_wf_k = wf_k * np.exp(1j * self.x[0] * self.dk * np.arange(self.N))
        self._ifft()
        self._fft()

    def _get_wf_k(self):
        return self.mod_wf_k * np.exp(-1j * self.x[0] * self.dk * np.arange(self.N))

    def _fft(self):
        self.mod_wf_k = fftpack.fft(self.mod_wf_x)

    def _ifft(self):
        self.mod_wf_x = fftpack.ifft(self.mod_wf_k)

    def _norm(self, wf):
        return np.sqrt((abs(wf) ** 2).sum() * 2 * np.pi / self.dx)

    def _evolve(self, dt):
        if dt != self.dt_:
            self.dt_ = dt
            self.wf_x_evolve_half = np.exp(-0.5 * 1j * self.V_x / self.hbar * self.dt_)
            self.wf_x_evolve = self.wf_x_evolve_half * self.wf_x_evolve_half
            self.wf_k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * (self.k * self.k) * self.dt_)

    def _get_dt(self):
        return self.dt_

    wf_x = property(_get_wf_x, _mod_wf_x)
    wf_k = property(_get_wf_k, _mod_wf_k)
    dt = property(_get_dt, _evolve)

    def solve(self, dt, Nsteps=1):
        self.dt = dt
        if Nsteps >= 0:
            self.mod_wf_x *= self.wf_x_evolve_half
        for i in range(Nsteps - 1):
            self._fft()
            self.mod_wf_k *= self.wf_k_evolve
            self._ifft()
            self.mod_wf_x *= self.wf_x_evolve
        self._fft()
        self.mod_wf_k *= self.wf_k_evolve
        self._ifft()
        self.mod_wf_x *= self.wf_x_evolve_half
        self._fft()
        self.mod_wf_x /= self._norm(self.mod_wf_x)
        self._ifft()    
