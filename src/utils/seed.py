import random
import numpy as np

def seed(seed: int = 42):
	np.random.seed(seed)
	random.seed(seed)