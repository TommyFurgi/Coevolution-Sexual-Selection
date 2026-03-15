# Coevolution & Sexual Selection Simulation

An agent-based simulation built with the **Mesa** framework and **Solara** visualization. This model explores the evolutionary dynamics between male traits and female preferences in a resource-constrained 2D environment.

## Overview

In this simulation, individuals (agents) must manage their energy by finding food while seeking reproductive partners. The population evolves over time as traits are inherited and mutated across generations.

### Key Mechanisms:
* **Energy Management:** Movement and reproduction consume energy. Finding food is essential for survival.
* **Mate Choice:** Females evaluate males based on a multi-dimensional trait vector.
* **Heredity:** Offspring inherit traits from both parents with a configurable degree of mutation.
* **Continuous Space:** Agents operate in a continuous 2D torus environment.

## Installation

Run the following command in the project directory to install all necessary dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Simulation

To launch the interactive dashboard, use the Solara server:

```bash
python -m solara run src/server.py
```

Once the server is running, open the address displayed in your terminal (typically `http://127.0.0.1:8765`).

## Configuration & Parameters

The simulation parameters can be adjusted in real-time through the web interface. These settings directly influence the evolutionary pressure and survival rates of the population:

| Parameter | Description |
| :--- | :--- |
| **Population Size** | The initial number of individuals spawned at the start of the simulation. |
| **Mutation Rate** | The standard deviation of genetic changes; higher values lead to more diverse traits in offspring. |
| **Movement Cost** | The amount of energy consumed by an agent for every step taken in the environment. |
| **Food Energy** | The amount of energy an agent recovers upon consuming a unit of food. |
| **Reproduction Cost** | The energy penalty applied to parents (higher for females) after successful breeding. |
| **Food Regrowth** | The rate at which new food units appear in the environment each step. |