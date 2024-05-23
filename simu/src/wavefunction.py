import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Wavefunction:
    def __init__(self, x, wf_x0, V_x, hbar, m, k=None):
        self.x, self.wf_x0, self.V_x = map(np.asarray, (x, wf_x0, V_x))

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

    def SSFM(self, dt, M):
        u = self.wf_x0

        expV = np.exp(-1j * self.V_x * dt / (2 * self.hbar))
        expK = np.exp(-1j * (self.hbar * self.k**2) * dt / (2 * self.m))

        results = [np.abs(u)**2]

        for _ in range(M):
            u = expV * u

            u_hat = fftpack.fft(u)

            u_hat = expK * u_hat

            u = fftpack.ifft(u_hat)

            u = expV * u

            results.append(np.abs(u)**2)

        return results

    
if __name__ == "__main__":
    # Define parameters
    hbar = 1.0
    m = 1.0
    x = np.linspace(-10, 10, 1024)
    wf_x0 = np.sin(np.pi * (x + 10) / 20)  # Initial wave function for infinite square well
    V_x = np.zeros_like(x)  # Infinite square well potential

    # Create wavefunction instance
    wf = Wavefunction(x, wf_x0, V_x, hbar, m)

    # Time step and number of steps
    dt = 0.01
    M = 100

    # Perform SSFM
    wf_results = wf.SSFM(dt, M)

    # Plot the result in animation
    fig, ax = plt.subplots()
    line, = ax.plot(x, wf_results[0])
    ax.set_ylim(0, 1)
    ax.set_xlim(-10, 10)

    def update(frame):
        line.set_ydata(wf_results[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(M+1), interval=50, blit=True)
    plt.show()
        





    
