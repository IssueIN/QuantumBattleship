# export PYTHONPATH=/path/to/quantum-battleship/game/src:$PYTHONPATH

import numpy as np
import random
from simu.src.wavefunction import Wavefunction
from simu.src.solver import TDSESolver

ParNum = 2  # one player has one particle
# coordinate, radius and velocity of each particle: {x-cood,y-cood,velocity,direction}
ParCord = [[0] * 4 for _ in range(ParNum)]  

TrapNum = 6  # each player has 3 traps
# coordinate and radius of trap  : {x-cood,y-cood,radius}
TrapCord = [[0] * 3 for _ in range(TrapNum)] 

# var for potential energy
PotentialNum = 2

RAND_MAX = 32767
class QuantumBattleship:
  def __init__(self):
    self.ParCord = np.zeros((2, 4))
    self.TrapCord = np.zeros((1, 3))
    
  def SetPar(self, number, Cord):
    """
    Set particle coordinates (x,y) and initial momentum (velocity, direction)
    Args:
      number (int): the number of the particle to be set, 0 for player 0/ 1 for player 1
      Cord (1D array of size 4): the initial conditions (x,y,v,d) of the particle
    """
    self.ParCord[number][0] = Cord[0]  # x-coord
    self.ParCord[number][1] = Cord[1]  # y-coord
    self.ParCord[number][2] = Cord[2]  # velocity
    self.ParCord[number][3] = Cord[3]  # direction


  # Set the trap on the map
  def SetTrap(self, number, trapCord):
    """
    Set the trap on the map, calls Catch to see if the particle is caught
    Args:
      number (int): the number of the trap to be set, 0 for player 0/ 1 for player 1
      trapCord (1D array of size 3): (x,y) + radius of the trap
    """
    self.TrapCord[number][0] = trapCord[0]
    self.TrapCord[number][1] = trapCord[1]
    self.TrapCord[number][2] = trapCord[2]
    
    return self.Catch(ParCord[number], trapCord)


  def RandNum(self):
    """
    Generate a random number to be used in the principle of uncertainty 
    returns a random float number for determining the probability of catching a particle
    """
    random.seed()
    random_fl = float(random.randint(0, RAND_MAX)) / RAND_MAX
    return random_fl


  # Calculate probability
  def Schro(self, Cord, Loc):
    """_summary_

    Args:
        Cord (1D array of size 4): coordinates of the particle
        Loc (1D array of size 2): coordinates of the location to calculate the probability

    Returns:
        float: the probability of a particle appearing in the (x,y) position
    """
    x = np.linspace(Cord[0], Cord[1], 100)
    y = np.linspace(Loc[0], Loc[1], 100)
    wf_xy0 = np.ones((100, 100))  # initial wavefunction (placeholder)
    V_xy = np.zeros((100, 100))   # potential (placeholder)
    hbar = 1.0
    m = 1.0

    wf = Wavefunction(x, y, wf_xy0, V_xy, hbar, m)
    wf.solve(0.01, 100)
    probability = np.abs(wf.get_wf_xy()) ** 2

    return probability
  

  # Check if the particle is caught by the trap
  def Catch(self, PCord, TCord):
    # generates a random number for the catching process
    if self.RanNum() <= self.Schro(PCord, TCord):
      return True
    else:
      return False

  # Detection on particle
  def Detect(self, Area):
    """Detect and see if the particle is within the detection range

    Args:
      Area (2D array of int, x1,x2,y1,y2): the area of detection
      
    Returns:
      probability (2D array of float): a 2D array of probabilities indicating the likelihood of the particle being at each point in the detection area
    """
    # go through each point in area and calculate the probability of the particle being there
    # returns a 2D array of probabilities
    probability = np.zeros((len(Area), len(Area[0])))
    for i in range(len(Area)):
      for j in range(len(Area[0])):
        probability[i][j] = self.Schro(ParCord, [Area[i][0], Area[i][1], Area[j][0], Area[j][1]])
    
    return probability



  # Movement of particles after being detected
  # to be resolved
  def Reaction(self, PNum, PCord, Detect):
      self.ParCord[PNum][0] = 0
      self.ParCord[PNum][1] = 0
      self.ParCord[PNum][2] = 0
      self.ParCord[PNum][3] = 0  # Setting new coordinate of the particle

  # to be resolved
  def SetPotential(self, PointCord, Potential):
    return 0

  # test to init game
  def init_game(self):
    # Set initial particle positions and traps
    self.SetPar(0, [2, 3, 1, 0])  # Particle 0 for Player 0
    self.SetPar(1, [6, 7, 1, 0])  # Particle 1 for Player 1
    self.SetTrap(0, [5, 5, 0])    # Trap

    # Perform detection
    probabilities = self.Detect([0, 10], True)
    print("Detection Probabilities:\n", probabilities)

    # Check if a particle is caught
    is_caught_0 = self.Catch(self.ParCord[0], self.TrapCord[0])
    is_caught_1 = self.Catch(self.ParCord[1], self.TrapCord[0])
    print("Is Particle 0 Caught: ", is_caught_0)
    print("Is Particle 1 Caught: ", is_caught_1)

# Uncomment the following lines to run the game initialization example
game = QuantumBattleship()
game.init_game()