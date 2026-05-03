from src.model import ReproductionModel
import altair as alt
import pandas as pd
import solara
from mesa.visualization import SolaraViz
from mesa.visualization.components import make_altair_plot_component
import numpy as np
from config import model_params

alt.data_transformers.disable_max_rows()


def _tooltip_common_traits():
    """Fields shown for every agent in the map tooltip."""
    return [
        "sex:N",
        "energy:Q",
        "age:Q",
        "trait_1:Q",
        "trait_2:Q",
        "trait_3:Q",
    ]


def _tooltip_female_prefs():
    """Preferences only in tooltip for females (males use a separate chart layer)."""
    return _tooltip_common_traits() + ["pref_1:Q", "pref_2:Q", "pref_3:Q"]

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

    color_scale = alt.Scale(
        domain=["Male", "Female", "Food"],
        range=["#4c78a8", "#e45756", "#2ca02c"],
    )
    _fixed = alt.Scale(domain=[0, 1], padding=10, nice=False, clamp=True)
    color_enc = alt.Color("category:N", title="Legend", scale=color_scale)
    x_enc = alt.X("x:Q", scale=_fixed, axis=alt.Axis(title="X"))
    y_enc = alt.Y("y:Q", scale=_fixed, axis=alt.Axis(title="Y"))

    charts = []

    if not df_food.empty:
        df_food = df_food.copy()
        df_food["category"] = "Food"
        charts.append(
            alt.Chart(df_food)
            .mark_square(size=50, opacity=0.7)
            .encode(x=x_enc, y=y_enc, color=color_enc, tooltip=["energy:Q"])
        )

    if not df_agents.empty:
        male_df = df_agents[df_agents["sex"] == "Male"].copy()
        female_df = df_agents[df_agents["sex"] == "Female"].copy()
        male_df["category"] = "Male"
        female_df["category"] = "Female"

        agent_layers = []
        if not male_df.empty:
            agent_layers.append(
                alt.Chart(male_df)
                .mark_point(filled=True, size=400)
                .encode(
                    x=x_enc,
                    y=y_enc,
                    color=color_enc,
                    tooltip=_tooltip_common_traits(),
                )
            )
        if not female_df.empty:
            agent_layers.append(
                alt.Chart(female_df)
                .mark_point(filled=True, size=400)
                .encode(
                    x=x_enc,
                    y=y_enc,
                    color=color_enc,
                    tooltip=_tooltip_female_prefs(),
                )
            )
        if agent_layers:
            charts.append(alt.layer(*agent_layers))

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

_FULL_CARD = {"width": "100%", "boxSizing": "border-box"}
_W, _H = 1100, 380


def _chart_card(m, title, cols, height=_H):
    with solara.Card(title, style=_FULL_CARD):
        def _size(chart):
            return chart.properties(width=_W, height=height)
        df = m.datacollector.get_model_vars_dataframe()
        if not df.empty:
            make_altair_plot_component(cols, post_process=_size)[0](m)
        else:
            solara.Text("Waiting for data...")


@solara.component
def dashboard(m):
    with solara.Column(style={"width": "100%", "boxSizing": "border-box", "padding": "8px", "gap": "16px"}):
        _chart_card(m, "Male Traits (mean)",
                    [f"Mean_Male_Trait_{i+1}" for i in range(m.trait_dim)])
        _chart_card(m, "Female Traits (mean)",
                    [f"Mean_Female_Trait_{i+1}" for i in range(m.trait_dim)])
        _chart_card(m, "Female Mate Preferences (mean)",
                    [f"Mean_Female_Pref_{i+1}" for i in range(m.trait_dim)])
        _chart_card(m, "Trait–Preference Alignment (gap → 0 = coevolution)",
                    [f"Trait_Pref_Gap_{i+1}" for i in range(m.trait_dim)])
        _chart_card(m, "Sex Ratio", ["Male_Count", "Female_Count"])
        _chart_card(m, "Mean Energy by Sex", ["Mean_Male_Energy", "Mean_Female_Energy"])
        _chart_card(m, "Population Metrics", ["Population_Count", "Avg_Age"])
        _chart_card(m, "Environment & Food", ["Total_Food"])

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
