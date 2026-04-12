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
                 mate_perception_radius=0.15,
                 mating_energy_buffer=0.05,
                 male_ornament_cost_coeff=0.012,
                 max_age=200,
                 mortality_start_age=100,
                 initial_energy=0.85,
                 eat_radius=0.028,
                 trait_precision_decimals=4,
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
        self.mate_perception_radius = mate_perception_radius
        self.mating_energy_buffer = mating_energy_buffer
        self.male_ornament_cost_coeff = male_ornament_cost_coeff
        self.max_age = max_age
        self.mortality_start_age = min(mortality_start_age, max(0, max_age - 1))
        self.initial_energy = initial_energy
        self.eat_radius = eat_radius
        self.trait_precision_decimals = max(0, int(trait_precision_decimals))

        self.max_energy = 2.0
        self.step_towards_food = 0.03
        self.step_towards_mate = 0.03
        self.random_step_std = 0.02
        self.reproduction_distance = 0.03
        self.female_reproduction_cooldown = 5

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

        def _male_mean(i):
            return lambda m: m.mean_trait_male(i)

        def _female_mean(i):
            return lambda m: m.mean_trait_female(i)

        def _male_var(i):
            return lambda m: m.var_trait_male(i)

        def _female_var(i):
            return lambda m: m.var_trait_female(i)

        def _female_pref_mean(i):
            return lambda m: m.mean_preference_female(i)

        reporters = {
            **{f"Mean_Male_Trait_{i+1}": _male_mean(i) for i in range(self.trait_dim)},
            **{f"Mean_Female_Trait_{i+1}": _female_mean(i) for i in range(self.trait_dim)},
            **{f"Var_Male_Trait_{i+1}": _male_var(i) for i in range(self.trait_dim)},
            **{f"Var_Female_Trait_{i+1}": _female_var(i) for i in range(self.trait_dim)},
            **{f"Mean_Female_Pref_{i+1}": _female_pref_mean(i) for i in range(self.trait_dim)},
            "Avg_Age": lambda m: np.mean([a.age for a in m.individuals]) if m.individuals else 0,
            "Population_Count": lambda m: len(m.individuals),
            "Total_Food": lambda m: len(m.food),
        }

        self.datacollector = DataCollector(model_reporters=reporters)

    def _energy_reserve_threshold(self, agent):
        """Below this energy (for sex), agent prioritizes foraging over mating."""
        if agent.sex == "Female":
            return self.female_reproduction_cost + self.mating_energy_buffer
        return self.male_reproduction_cost + self.mating_energy_buffer

    def _needs_food(self, agent):
        return bool(self.food) and agent.energy < self._energy_reserve_threshold(agent)

    def _female_seeks_mate(self, agent):
        return (
            agent.sex == "Female"
            and not self._needs_food(agent)
            and agent.energy > self.female_reproduction_cost
            and getattr(agent, "reproduction_cooldown", 0) == 0
        )

    def mean_trait_male(self, index):
        males = [a for a in self.individuals if a.sex == "Male"]
        if not males:
            return 0.0
        return float(np.mean([a.traits[index] for a in males]))

    def mean_trait_female(self, index):
        females = [a for a in self.individuals if a.sex == "Female"]
        if not females:
            return 0.0
        return float(np.mean([a.traits[index] for a in females]))

    def var_trait_male(self, index):
        males = [a for a in self.individuals if a.sex == "Male"]
        if len(males) < 2:
            return 0.0
        return float(np.var([a.traits[index] for a in males]))

    def var_trait_female(self, index):
        females = [a for a in self.individuals if a.sex == "Female"]
        if len(females) < 2:
            return 0.0
        return float(np.var([a.traits[index] for a in females]))

    def mean_preference_female(self, index):
        females = [a for a in self.individuals if a.sex == "Female" and a.preferences is not None]
        if not females:
            return 0.0
        return float(np.mean([a.preferences[index] for a in females]))

    def _age_mortality_probability(self, age):
        """Per-step death probability from age alone; 0 below mortality_start_age,
        linear from 0 to almost 1 between mortality_start_age and max_age (exclusive)."""
        if age < self.mortality_start_age:
            return 0.0
        if age >= self.max_age:
            return 1.0
        span = self.max_age - self.mortality_start_age
        if span <= 0:
            return 1.0
        return (age - self.mortality_start_age) / float(span)

    def _torus_delta(self, from_pos, to_pos):
        """Shortest displacement from from_pos to to_pos (minimal image if torus)."""
        p = np.asarray(from_pos, dtype=float)
        q = np.asarray(to_pos, dtype=float)
        d = q - p
        if self.space.torus:
            w, h = self.space.width, self.space.height
            d = d.copy()
            d[0] -= w * np.rint(d[0] / w)
            d[1] -= h * np.rint(d[1] / h)
        return d

    def _torus_distance(self, a, b):
        return float(np.linalg.norm(self._torus_delta(a, b)))

    def _best_preferred_male(self, female, males, f_pos):
        """Same rule for movement and mating: minimize trait distance to preferences;
        ties broken by shorter torus distance."""
        if not males:
            return None
        f_pos = np.asarray(f_pos, dtype=float)
        spatial = np.array([self._torus_distance(f_pos, m.pos) for m in males])
        if female.preferences is not None:
            pref = np.asarray(female.preferences, dtype=float)
            trait_dists = np.array(
                [float(np.linalg.norm(m.traits - pref)) for m in males]
            )
            order = np.lexsort((spatial, trait_dists))
            return males[int(order[0])]
        return males[int(np.argmin(spatial))]

    # -----------------------
    # Simulation step
    # -----------------------

    def step(self):

        survivors = []
        children = []

        for agent in self.individuals:

            if agent.pos is None:
                continue

            self._update_agent_energy_and_age(agent)

            pos = np.array(agent.pos, dtype=float)
            step_vec = self._compute_movement(agent)
            new_pos = pos + step_vec
            self.space.move_agent(agent, (float(new_pos[0]), float(new_pos[1])))
            new_pos = np.array(agent.pos, dtype=float)
            agent.energy -= self.move_cost

            self._handle_eating(agent, new_pos)

            if agent.sex == "Male" and self.male_ornament_cost_coeff > 0:
                cost = self.male_ornament_cost_coeff * float(np.linalg.norm(agent.traits))
                agent.energy = max(0.0, agent.energy - cost)

            self._handle_death(agent, survivors, new_pos)

        self.individuals = survivors

        males = [
            a
            for a in self.individuals
            if a.sex == "Male"
            and not self._needs_food(a)
            and a.energy > self.male_reproduction_cost
            and a.pos is not None
        ]
        females = [
            a
            for a in self.individuals
            if self._female_seeks_mate(a) and a.pos is not None
        ]

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

        # 1) Food seeking — reserve depends on sex (no dead band vs mating)
        if self._needs_food(agent):
            best_i = None
            best_d = np.inf
            for i, f in enumerate(self.food):
                d = self._torus_distance(pos, (f["x"], f["y"]))
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is None:
                return np.random.normal(0, self.random_step_std, 2)
            direction = self._torus_delta(pos, (self.food[best_i]["x"], self.food[best_i]["y"]))
            dist = float(np.linalg.norm(direction))
            if dist > 0:
                return direction / dist * self.step_towards_food
            return np.zeros(2)

        # 2) Female mate seeking — only when reserves allow reproduction; males in perception radius
        if self._female_seeks_mate(agent):

            candidate_males = [
                m
                for m in self.individuals
                if m.sex == "Male"
                and m.pos is not None
                and self._torus_distance(pos, m.pos) <= self.mate_perception_radius
            ]

            if candidate_males:
                target_male = self._best_preferred_male(agent, candidate_males, pos)
                if target_male is None:
                    return np.random.normal(0, self.random_step_std, 2)
                direction = self._torus_delta(pos, target_male.pos)
                dist = float(np.linalg.norm(direction))
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

            if self._torus_distance(new_pos, (fx, fy)) <= self.eat_radius:
                agent.energy = min(agent.energy + food["energy"], self.max_energy)
                eaten_indices.append(i)

        for i in sorted(eaten_indices, reverse=True):
            del self.food[i]


    # ------------------------------------------------------------

    def _handle_death(self, agent, survivors, new_pos):
        if agent.energy <= 0:
            self.space.remove_agent(agent)
            return
        if agent.age >= self.max_age:
            self.space.remove_agent(agent)
            return
        p_die = self._age_mortality_probability(agent.age)
        if p_die > 0 and self.random.random() < p_die:
            self.space.remove_agent(agent)
            return
        survivors.append(agent)


    # ------------------------------------------------------------

    def _reproduction_phase(self, children, females, males):
        for female in females:

            if female.pos is None or not males:
                continue

            f_pos = np.array(female.pos, dtype=float)

            nearby = [
                m
                for m in males
                if m.pos is not None
                and self._torus_distance(f_pos, m.pos) <= self.reproduction_distance
                and m.energy > self.male_reproduction_cost
            ]
            if not nearby:
                continue

            if female.energy <= self.female_reproduction_cost:
                continue

            chosen_male = self._best_preferred_male(female, nearby, f_pos)
            if chosen_male is None:
                continue

            female.energy -= self.female_reproduction_cost
            chosen_male.energy -= self.male_reproduction_cost

            female.reproduction_cooldown = self.female_reproduction_cooldown

            traits = (female.traits + chosen_male.traits) / 2.0
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

            m_pos = np.array(chosen_male.pos, dtype=float)
            base_pos = f_pos + 0.5 * self._torus_delta(f_pos, m_pos)

            raw_pos = base_pos + np.random.normal(0, 0.01, 2)
            pos = self.space.torus_adj((float(raw_pos[0]), float(raw_pos[1])))

            self.space.place_agent(child, pos)
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