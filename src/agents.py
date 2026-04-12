import numpy as np
from mesa import Agent


class Individual(Agent):
    def __init__(self, model, sex, traits, preferences=None):
        super().__init__(model)
        self.sex = sex
        decimals = int(getattr(model, "trait_precision_decimals", 5))
        self.traits = np.round(np.asarray(traits, dtype=float), decimals)
        self.preferences = (
            None
            if preferences is None
            else np.round(np.asarray(preferences, dtype=float), decimals)
        )
        self.energy = getattr(model, "initial_energy", 1.0)
        self.age = 0
        self.reproduction_cooldown = 0

