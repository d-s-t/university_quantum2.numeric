from utils import const, flatten
from astropy.units.quantity import Quantity
import numpy as np
from typing import Callable
from classes import Particle
from tqdm import tqdm
from sys import float_info  

class Numerov:
    """
    Numerov model for quantum mechanics for 2 particles for equations of the form:
    -ħ²/2μ ∇²u + V(r)u = Eu
    where it gets the form:
    d²u/dr² + [(E- V(r))(2μ/ħ²) - (l(l+1)/r²)]u = 0
    and we define the W function as:
    W(r) = (E - V(r))(2μ/ħ²) - l(l+1)/r²
    """
    def __init__(self,
                 p: Particle,
                 V: Callable[[np.ndarray[Quantity["fm"]]], np.ndarray[Quantity["MeV"]]],
                 n: int,
                 l: np.ndarray[int]=None):
        """
        p: Particle
            Particle to be used in the model
        V: Callable[[np.ndarray[Quantity["fm"]]], np.ndarray[Quantity["MeV"]]]
            Potential energy function
        n: int
            Principal quantum number
        l: np.ndarray[int] (optional)
            Azimuthal quantum number
            Defaults to np.arange(n, dtype=int)
        """
        self.particle = p
        self.V = V
        self.n = n
        self.l = l or np.arange(n, dtype=int)

    def W(self,
          E: np.ndarray[Quantity["MeV"]],
          r: np.ndarray[Quantity["fm"]]
          ) -> np.ndarray[Quantity["fm-2"]]:
        """
        the W function for the given energy and distance

        E: np.ndarray[Quantity["MeV"]]
            Energy of the particle
            shape: (M,)
        r: np.ndarray[Quantity["fm"]]
            Array of distances
            shape: (N,)
        
        returns: np.ndarray[Quantity["fm-2"]]
            W function
            shape: (N, M, L)
            where L is the number of angular momentum quantum numbers
        """
        r = r[:, np.newaxis]
        return (2 * self.particle.m * (E-self.V(r))[:, :, np.newaxis] / const.hbar**2 - (self.l * (self.l+1) / r**2)[:, np.newaxis, :]).to('fm-2')


    def ure(self,
            r: np.ndarray[float],
            E: np.ndarray[float],
            progress_bar: bool = True
            ) -> Quantity["fm"]:
        """
        Numerov method to solve the equation for the wave function

        r: np.ndarray[float]
            Array of distances in fm
            shape: (N,)
        E: np.ndarray[float]
            Array of energies in MeV
            shape: (M,)
        progress_bar: bool
            Whether to show a progress bar or not
        """
        r = r * self.particle.a_B
        dr = r[1] - r[0]
        u = np.zeros((*r.shape, *E.shape)) * dr.unit
        E = E * self.particle.R_y
        u[1,:] = dr**(self.l+1)
        w = dr**2 * self.W(E, r) / 12
        interations = tqdm(range(2, len(r))) if progress_bar else range(2, len(r))
        for i in interations:
            u[i, :] = ((2 - w[i-1,:] * 10) * u[i-1,:] - (1 + w[i-2,:]) * u[i-2,:]) / (1 + w[i,:])
        return u
    
    def find_root(self, E_max: float, E_min: float,
                  r: np.ndarray[float],
                  N: int = 10,
                  ) -> np.ndarray[float]:
        """
        find the eneergy where the wave function u(R,E) = 0

        E_max: float
            Maximum energy to search for the root as a multiple of the Rydberg energy
        E_min: float
            Minimum energy to search for the root as a multiple of the Rydberg energy
        r: np.ndarray[float]
            Array of distances in fm
            shape: (M,)
        N: int (optional)
            Number of devisions to make in the search for the root
        """
        E_bounds = np.linspace(E_min, E_max, N + 1)
        u = self.ure(r, E_bounds, progress_bar=False)[-1,:]
        u_min, u_max = u[0], u[-1]
        print(f"diff={abs(E_max - E_min):.3e}, {u_min=:.3e}, {u_max=:.3e}", end='\r', flush=True)
        if abs(E_max - E_min) <= float_info.epsilon:
            print()
            return [E_min] if abs(u_min) < abs(u_max) else [E_max]
        return flatten([self.find_root(E_bounds[i+1], E_bounds[i], r, N) for i in range(N) if u[i]*u[i+1]<0])
        
    
    def etta(self, E: float):
        """
        relative error of the energy
        """
        return abs(1 + self.n**2 * E)


    @property
    def a_B(self):
        return self.particle.a_B
    
    @property
    def R_y(self):
        return self.particle.R_y
    

    def __str__(self):
        return f"particle {self.particle.name} in state |n={self.n},l={self.l}⟩"
