import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.wavefunction import Wavefunction

# Example usage
if __name__ == "__main__":
    # Define parameters
    hbar = 1.0
    m = 1.0
    L = 30  # Width of the infinite square well
    N = 1024  # Number of spatial points
    x = np.linspace(-L / 2, L / 2, N)

    # Initial Gaussian wave packet
    x0 = -5  # Initial position
    sigma = 1.0  # Width of the wave packet
    k0 = -5.0  # Initial momentum
    wf_x0 = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x)

    # Infinite square well with a step potential inside
    V_x = np.zeros_like(x)
    V_x[x > 0] = 10  # Step potential of height 10 for x > 0
    V_x[x > 10] = 1e6
    V_x[x < -10] = 1e6

    # Create wavefunction instance
    wf = Wavefunction(x, wf_x0, V_x, hbar, m)

    # Time step and number of steps
    dt = 0.001
    M = 2000

    # Perform SSFM
    wf_results = []
    for i in range(M):
        wf.solve(dt, 1)
        wf_results.append(np.abs(wf.wf_x) ** 2)

    # Plot the result in animation
    fig, ax = plt.subplots()
    line, = ax.plot(x, wf_results[0])
    ax.set_ylim(0, np.max(wf_results))
    ax.set_xlim(-L / 2, L / 2)
    ax.set_xlabel('x')
    ax.set_ylabel('|ψ(x)|^2')

    def update(frame):
        line.set_ydata(wf_results[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(0, M, 10), interval=50, blit=True)
    plt.show()
