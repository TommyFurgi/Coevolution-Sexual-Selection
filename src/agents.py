from mesa import Agent


class Individual(Agent):
    def __init__(self, model, sex, traits, preferences=None):
        super().__init__(model)
        self.sex = sex
        self.traits = traits
        self.preferences = preferences
        self.energy = getattr(model, "initial_energy", 1.0)
        self.age = 0
        self.reproduction_cooldown = 0

