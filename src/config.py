import solara

model_params = {
    # Evolution Parameters
    "mutation_std": {
        "type": "SliderFloat",
        "value": 0.05,
        "label": "Mutation Rate (Std Dev)",
        "min": 0.0,
        "max": 0.3,
        "step": 0.01,
    },
    "trait_dim": {
        "type": "SliderInt",
        "value": 3,
        "label": "Trait Dimensions",
        "min": 1,
        "max": 10,
        "step": 1,
    },

    # Energy Economy
    "move_cost": {
        "type": "SliderFloat",
        "value": 0.01,
        "label": "Movement Energy Cost",
        "min": 0.0,
        "max": 0.1,
        "step": 0.001,
    },
    "food_energy": {
        "type": "SliderFloat",
        "value": 0.4,
        "label": "Energy per Food Unit",
        "min": 0.1,
        "max": 2.0,
        "step": 0.05,
    },

    # Reproduction
    "female_reproduction_cost": {
        "type": "SliderFloat",
        "value": 0.7,
        "label": "Female Reproduction Cost",
        "min": 0.1,
        "max": 1.5,
        "step": 0.05,
    },
    "male_reproduction_cost": {
        "type": "SliderFloat",
        "value": 0.15,
        "label": "Male Reproduction Cost",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },

    # Population & Environment
    "population_size": {
        "type": "SliderInt",
        "value": 50,
        "label": "Initial Population Size",
        "min": 2,
        "max": 300,
        "step": 2,
    },
    "n_food": {
        "type": "SliderInt",
        "value": 30,
        "label": "Initial Food Count",
        "min": 0,
        "max": 200,
        "step": 5,
    },
    "food_regrowth_per_step": {
        "type": "SliderInt",
        "value": 3,
        "label": "Food Regrowth Rate",
        "min": 0,
        "max": 50,
        "step": 1,
    },
}