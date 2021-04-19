import numpy as np
from torch import device, cuda

np.set_printoptions(precision=2)

run_device = device("cuda" if cuda.is_available() else "cpu")
