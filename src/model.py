import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace
from src.agents import Individual


class ReproductionModel(Model):
    def __init__(self,
                 population_size=50,
                 trait_dim=3,
                 mutation_std=0.05,
                 n_food=30,
                 move_cost=0.01,
                 food_energy=0.4,
                 female_reproduction_cost=0.7,
                 male_reproduction_cost=0.15,
                 food_regrowth_per_step=3,
                 **kwargs):
        
        super().__init__()

        self.population_size = population_size
        self.trait_dim = trait_dim
        self.mutation_std = mutation_std
        self.move_cost = move_cost
        self.food_energy = food_energy
        self.female_reproduction_cost = female_reproduction_cost
        self.male_reproduction_cost = male_reproduction_cost
        self.food_regrowth_per_step = food_regrowth_per_step

        self.initial_energy = 0.6
        self.max_energy = 2.0
        self.low_energy_threshold = 0.7 # minimum energy until agent seek for food
        self.step_towards_food = 0.03
        self.step_towards_mate = 0.03
        self.random_step_std = 0.02
        self.eat_radius = 0.02
        self.reproduction_energy_threshold = 0.9 # minimum energy until agent seek to reproduce
        self.reproduction_distance = 0.03
        self.female_reproduction_cooldown = 5
        self.max_age = 100

        self.space = ContinuousSpace(1.0, 1.0, torus=True)

        self.individuals = []

        self.food = []
        for _ in range(n_food):
            fx, fy = np.random.rand(2)
            self.food.append({"x": float(fx), "y": float(fy), "energy": self.food_energy})

        for i in range(population_size):

            sex = "Male" if i < population_size / 2 else "Female"
            traits = np.random.rand(trait_dim)

            if sex == "Female":
                preferences = np.random.rand(trait_dim)
            else:
                preferences = None

            agent = Individual(self, sex, traits, preferences)
            pos = np.random.rand(2)
            self.space.place_agent(agent, pos)
            self.individuals.append(agent)

        reporters = {}
        for i in range(self.trait_dim):
            name = f"Mean_Trait_{i+1}"
            reporters[name] = (lambda index: lambda m: m.mean_trait(index))(i)

        self.datacollector = DataCollector(
            model_reporters={
                # Dynamic traits
                **{f"Mean_Trait_{i+1}": (lambda index: lambda m: m.mean_trait(index))(i) 
                for i in range(self.trait_dim)},

                "Avg_Age": lambda m: np.mean([a.age for a in m.individuals]) if m.individuals else 0,
                "Population_Count": lambda m: len(m.individuals),
                "Total_Food": lambda m: len(m.food)
            }
        )

    def mean_trait(self, index):
        males = [a for a in self.individuals if a.sex == "Male"]
        if len(males) == 0:
            return 0
        return np.mean([a.traits[index] for a in males])

    # -----------------------
    # Simulation step
    # -----------------------

    def step(self):

        survivors = []
        children = []

        males = [
            a for a in self.individuals
            if a.sex == "Male" and a.energy >= self.reproduction_energy_threshold
        ]

        females = [
            a for a in self.individuals
            if a.sex == "Female"
            and a.energy >= self.reproduction_energy_threshold
            and getattr(a, "reproduction_cooldown", 0) == 0
        ]

        for agent in self.individuals:

            if agent.pos is None:
                continue

            self._update_agent_energy_and_age(agent)

            pos = np.array(agent.pos, dtype=float)
            step_vec = self._compute_movement(agent)
            new_pos = np.clip(pos + step_vec, 0.0, 1.0)
            self.space.move_agent(agent, tuple(new_pos))
            agent.energy -= self.move_cost

            self._handle_eating(agent, new_pos)
            self._handle_death(agent, survivors, new_pos)

        self.individuals = survivors

        self._reproduction_phase(children, females, males)
        self.individuals.extend(children)

        self._regrow_food()

        self.datacollector.collect(self)


# ============================================================
# Agent utilities
# ============================================================

    def _update_agent_energy_and_age(self, agent):
        agent.age += 1

        if hasattr(agent, "reproduction_cooldown") and agent.reproduction_cooldown > 0:
            agent.reproduction_cooldown -= 1
            if agent.reproduction_cooldown < 0:
                agent.reproduction_cooldown = 0


    # ------------------------------------------------------------

    def _compute_movement(self, agent):
        pos = np.array(agent.pos, dtype=float)

        # 1) Food seeking
        if self.food and agent.energy < self.low_energy_threshold:
            food_positions = np.array([[f["x"], f["y"]] for f in self.food])
            dists = np.linalg.norm(food_positions - pos, axis=1)

            nearest_idx = int(np.argmin(dists))
            target = food_positions[nearest_idx]

            direction = target - pos
            dist = np.linalg.norm(direction)

            if dist > 0:
                return direction / dist * self.step_towards_food
            return np.zeros(2)

        # 2) Female mate seeking
        if agent.sex == "Female" and agent.energy >= self.reproduction_energy_threshold:

            candidate_males = [
                m for m in self.individuals
                if m.sex == "Male" and m.pos is not None
            ]

            if candidate_males:

                if agent.preferences is not None:
                    pref_vec = np.array(agent.preferences, dtype=float)
                    male_traits = np.array([m.traits for m in candidate_males], dtype=float)

                    trait_dists = np.linalg.norm(male_traits - pref_vec, axis=1)
                    best_idx = int(np.argmin(trait_dists))

                else:
                    male_positions = np.array(
                        [np.array(m.pos, dtype=float) for m in candidate_males]
                    )
                    spatial_dists = np.linalg.norm(male_positions - pos, axis=1)
                    best_idx = int(np.argmin(spatial_dists))

                target_pos = np.array(candidate_males[best_idx].pos, dtype=float)

                direction = target_pos - pos
                dist = np.linalg.norm(direction)

                if dist > 0:
                    return direction / dist * self.step_towards_mate
                return np.zeros(2)

            return np.random.normal(0, self.random_step_std, 2)

        return np.random.normal(0, self.random_step_std, 2)


    # ------------------------------------------------------------

    def _handle_eating(self, agent, new_pos):
        eaten_indices = []

        for i, food in enumerate(self.food):
            fx, fy = food["x"], food["y"]

            if np.linalg.norm(new_pos - np.array([fx, fy])) <= self.eat_radius:
                agent.energy = min(agent.energy + food["energy"], self.max_energy)
                eaten_indices.append(i)

        for i in sorted(eaten_indices, reverse=True):
            del self.food[i]


    # ------------------------------------------------------------

    def _handle_death(self, agent, survivors, new_pos):
        if agent.energy > 0 and agent.age < self.max_age:
            survivors.append(agent)
        else:
            self.space.remove_agent(agent)


    # ------------------------------------------------------------

    def _reproduction_phase(self, children, females, males):
        for female in females:

            if female.pos is None or not males:
                continue

            f_pos = np.array(female.pos, dtype=float)

            male_positions = np.array(
                [np.array(m.pos, dtype=float) for m in males if m.pos is not None]
            )

            if male_positions.size == 0:
                continue

            dists = np.linalg.norm(male_positions - f_pos, axis=1)

            nearest_idx = int(np.argmin(dists))
            nearest_male = males[nearest_idx]

            if dists[nearest_idx] > self.reproduction_distance:
                continue

            if (female.energy - self.female_reproduction_cost <= 0 or
                    nearest_male.energy - self.male_reproduction_cost <= 0):
                continue

            female.energy -= self.female_reproduction_cost
            nearest_male.energy -= self.male_reproduction_cost

            female.reproduction_cooldown = self.female_reproduction_cooldown

            traits = (female.traits + nearest_male.traits) / 2.0
            traits += np.random.normal(0, self.mutation_std, self.trait_dim)
            traits = np.clip(traits, 0.0, 1.0)

            child_sex = "Male" if np.random.rand() < 0.5 else "Female"

            if child_sex == "Female":
                if female.preferences is not None:
                    preferences = female.preferences.copy()
                else:
                    preferences = np.random.rand(self.trait_dim)

                preferences += np.random.normal(0, self.mutation_std, self.trait_dim)
                preferences = np.clip(preferences, 0.0, 1.0)
            else:
                preferences = None

            child = Individual(self, child_sex, traits, preferences)

            m_pos = np.array(nearest_male.pos, dtype=float)
            base_pos = (f_pos + m_pos) / 2.0

            pos = base_pos + np.random.normal(0, 0.01, 2)
            pos = np.clip(pos, 0.0, 1.0)

            self.space.place_agent(child, tuple(pos))
            children.append(child)


    # ------------------------------------------------------------

    def _regrow_food(self):
        for _ in range(self.food_regrowth_per_step):
            fx, fy = np.random.rand(2)
            self.food.append({
                "x": float(fx),
                "y": float(fy),
                "energy": self.food_energy
            })