from src.model import ReproductionModel
import altair as alt
import pandas as pd
import solara
from mesa.visualization import SolaraViz
from mesa.visualization.components import make_altair_plot_component
import numpy as np
from config import model_params


def _build_tooltip(df: pd.DataFrame):
    base_fields = [
        "sex:N",
        "energy:Q",
        "age:Q",
        "trait_1:Q",
        "trait_2:Q",
        "trait_3:Q",
        "pref_1:Q",
        "pref_2:Q",
        "pref_3:Q",
    ]

    tooltip = []
    for f in base_fields:
        tooltip.append(f)

    return tooltip

def _agent_records(model: ReproductionModel) -> pd.DataFrame:
    rows = []
    for a in getattr(model, "individuals", []):
        if a.pos is None:
            continue
        x, y = a.pos
        traits = list(a.traits)
        prefs = list(a.preferences) if a.preferences is not None else [None] * len(traits)
        rows.append(
            {
                "x": float(x),
                "y": float(y),
                "sex": a.sex,
                "energy": float(getattr(a, "energy", 0.0)),
                "age": float(getattr(a, "age", 0.0)),
                "trait_1": traits[0],
                "trait_2": traits[1] if len(traits) > 1 else np.nan,
                "trait_3": traits[2] if len(traits) > 2 else np.nan,
                "pref_1": prefs[0] if len(prefs) > 0 else np.nan,
                "pref_2": prefs[1] if len(prefs) > 1 else np.nan,
                "pref_3": prefs[2] if len(prefs) > 2 else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "x",
                "y",
                "sex",
                "age",
                "energy",
                "trait_1",
                "trait_2",
                "trait_3",
                "pref_1",
                "pref_2",
                "pref_3",
            ]
        )
    return pd.DataFrame(rows)


def _food_records(model: ReproductionModel) -> pd.DataFrame:
    rows = []
    for f in getattr(model, "food", []):
        rows.append({"x": f["x"], "y": f["y"], "energy": f["energy"]})
    if not rows:
        return pd.DataFrame(columns=["x", "y", "energy"])
    return pd.DataFrame(rows)


def agent_map_component(model: ReproductionModel):
    df_agents = _agent_records(model)
    df_food = _food_records(model)

    if df_agents.empty and df_food.empty:
        return solara.Text("No agents to display")

    charts = []

    if not df_food.empty:
        food_chart = (
            alt.Chart(df_food)
            .mark_square(size=50, opacity=0.7, color="green")
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(title="X")),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(title="Y")),
                tooltip=["energy:Q"],
            )
        )
        charts.append(food_chart)

    if not df_agents.empty:
        agent_chart = (
            alt.Chart(df_agents)
            .mark_point(filled=True, size=400)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1], padding=10), axis=alt.Axis(title="X")),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1], padding=10), axis=alt.Axis(title="Y")),
                color=alt.Color("sex:N", title="Gender"),
                tooltip=_build_tooltip(df_agents)
            )
        )
        charts.append(agent_chart)

    chart = alt.layer(*charts).properties(width=1000, height=600)

    return solara.FigureAltair(chart)


def model_params_component(model: ReproductionModel):
    rows = []
    for k, v in model.__dict__.items():
        if isinstance(v, (int, float, str, bool)):
            rows.append({"Parameter": k, "Value": v})

    df = pd.DataFrame(rows).sort_values("Parameter").reset_index(drop=True)

    if df.empty:
        return solara.Text("No parameters found")

    with solara.Div(style={"height": "900px", "width": "600px", "overflow": "auto"}):
        solara.DataFrame(
            df,
            items_per_page=30
        )

def safe_altair_component(m, cols):
    return make_altair_plot_component(cols)[0](m)

@solara.component
def dashboard(m):
    with solara.Column(style={"gap": "20px", "padding": "10px"}):
        
        with solara.Card("Evolutionary Traits"):
            df = m.datacollector.get_model_vars_dataframe()
            if not df.empty:
                trait_cols = [c for c in df.columns if c.startswith("Mean_Trait_")]
                safe_altair_component(m, trait_cols)
            else:
                solara.Text("Waiting for data...")

        with solara.Card("Population Metrics"):
            df = m.datacollector.get_model_vars_dataframe()
            if not df.empty:
                safe_altair_component(m, ["Population_Count", "Avg_Age"])
            else:
                solara.Text("Waiting for data...")

        with solara.Card("Environment & Food"):
            df = m.datacollector.get_model_vars_dataframe()
            if not df.empty:
                safe_altair_component(m, ["Total_Food"])
            else:
                solara.Text("Waiting for data...")

@solara.component
def Page():
    initial_values = {k: v["value"] if isinstance(v, dict) else v 
                     for k, v in model_params.items()}
    
    model_state = solara.use_reactive(ReproductionModel(**initial_values))

    components = [
        (agent_map_component, 0),
        (model_params_component, 1),
        (dashboard, 2),
    ]

    return SolaraViz(
        model=model_state.value,
        components=components,
        model_params=model_params,
        name="Reproduction Model",
    )
