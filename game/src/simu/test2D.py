import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
from src.wavefunction import Wavefunction

# Example use - 2D simulation
if __name__ == "__main__":
    # Define parameters
    hbar = 1.0
    m = 1.0
    Lx, Ly = 30, 30  # Width of the infinite square well
    Nx, Ny = 128, 128  # Number of spatial points
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)

    # Initial Gaussian wave packet
    x0, y0 = -5, -5  # Initial position
    sigma = 1.0  # Width of the wave packet
    k0x, k0y = -10.0, -15.0  # Initial momentum
    X, Y = np.meshgrid(x, y)
    wf_xy0 = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2)) * np.exp(1j * (k0x * X + k0y * Y))

    # Infinite square well with a step potential inside
    V_xy = np.zeros_like(X)
    V_xy[X > 0] = 100  # Step potential of height 10 for x > 0
    V_xy[X > 10] = 1e10
    V_xy[X < -10] = 1e10
    V_xy[Y > 10] = 1e10
    V_xy[Y < -10] = 1e10

    # Create wavefunction instance
    wf2d = Wavefunction(x, y, wf_xy0, V_xy, hbar, m)

    # Time step and number of steps
    dt = 0.001
    M = 2000

    # Perform SSFM
    wf_results = []
    for i in range(M):
        wf2d.solve(dt, 1)
        wf_xy = wf2d.get_wf_xy()
        wf_results.append(np.abs(wf_xy) ** 2)
        # wf_x_part, wf_y_part = wf2d.get_wf_xy()
        # wf_combined = np.outer(wf_x_part, wf_y_part)
        # wf_combined = np.abs(wf_combined) ** 2 
        # wf_results.append(wf_combined)
        # # wf_combined = np.outer(wf_x_part, wf_y_part)
        # wf_combined = np.abs(wf_x_part) ** 2 + np.abs(wf_y_part.T) ** 2 
        # wf_results.append(wf_combined ** 2)

    # Plot the result in 3D animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    Z = wf_results[0]

    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_zlim(0, np.max(wf_results))

    def update(frame):
        ax.clear()
        Z = wf_results[frame]
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_zlim(0, np.max(wf_results))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('|ψ(x, y)|^2')
        return surf,

    ani = animation.FuncAnimation(fig, update, frames=range(0, M, 10), interval=50, blit=False)
    plt.show()