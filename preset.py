from classes import Attom, Particle
from utils import units

H1 = Attom('H1', Z=1, A=1, R=0.88 * units.fm)
H2 = Attom('H2', Z=1, A=2, R=2.14 * units.fm)
He4 = Attom('He', Z=2, A=4, R=1.97 * units.fm)
K = Particle('K', 493.7 * units.MeVc2, -1)
