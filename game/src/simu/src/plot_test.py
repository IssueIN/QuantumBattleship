import numpy as np
import matplotlib.pyplot as plt

def plot_potential_heatmap_example():
    # Define the grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create multiple potentials
    V1 = np.zeros_like(X)
    V1[(X >= -2) & (X <= 2) & (Y >= -2) & (Y <= 2)] = 10
    
    V2 = np.zeros_like(X)
    V2[(X >= -4) & (X <= -3) & (Y >= -4) & (Y <= -3)] = 20
    
    V3 = np.zeros_like(X)
    V3[(X >= 3) & (X <= 4) & (Y >= 3) & (Y <= 4)] = 15
    
    # Sum the potentials
    total_potential = V1 + V2 + V3
    
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(total_potential, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Potential')
    plt.title('Heatmap of the Potential')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_potential_heatmap_example()
