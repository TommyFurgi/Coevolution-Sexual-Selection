from src.model import ReproductionModel
import altair as alt
import pandas as pd
import solara
from mesa.visualization import SolaraViz
from mesa.visualization.components import make_altair_plot_component
import numpy as np
from config import model_params


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
        sex_scale = alt.Scale(domain=["Male", "Female"])
        x_enc = alt.X("x:Q", scale=alt.Scale(domain=[0, 1], padding=10), axis=alt.Axis(title="X"))
        y_enc = alt.Y("y:Q", scale=alt.Scale(domain=[0, 1], padding=10), axis=alt.Axis(title="Y"))
        color_enc = alt.Color("sex:N", title="Gender", scale=sex_scale)

        male_df = df_agents[df_agents["sex"] == "Male"]
        female_df = df_agents[df_agents["sex"] == "Female"]

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

def safe_altair_component(m, cols, width=720, height=380):
    def _size(chart):
        return chart.properties(width=width, height=height)

    return make_altair_plot_component(cols, post_process=_size)[0](m)


_CARD_FLEX = {"flex": "1 1 380px", "minWidth": "min(100%, 300px)", "maxWidth": "100%"}
_TRAITS_CARD_FLEX = {"flex": "2 1 560px", "minWidth": "min(100%, 420px)", "maxWidth": "100%"}
_ROW_STYLE = {
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "16px",
    "width": "100%",
    "alignItems": "stretch",
    "justifyContent": "stretch",
}


@solara.component
def dashboard(m):
    with solara.Column(style={"width": "100%", "boxSizing": "border-box", "padding": "8px"}):
        with solara.Row(style=_ROW_STYLE):
            with solara.Card("Evolutionary Traits", style=_TRAITS_CARD_FLEX):
                df = m.datacollector.get_model_vars_dataframe()
                if not df.empty:
                    trait_cols = [
                        c
                        for c in df.columns
                        if c.startswith(
                            (
                                "Mean_Male_Trait_",
                                "Mean_Female_Trait_",
                                "Var_Male_Trait_",
                                "Var_Female_Trait_",
                                "Mean_Female_Pref_",
                            )
                        )
                    ]
                    safe_altair_component(m, trait_cols, width=1100, height=420)
                else:
                    solara.Text("Waiting for data...")

            with solara.Card("Population Metrics", style=_CARD_FLEX):
                df = m.datacollector.get_model_vars_dataframe()
                if not df.empty:
                    safe_altair_component(m, ["Population_Count", "Avg_Age"], width=1100, height=420)
                else:
                    solara.Text("Waiting for data...")

            with solara.Card("Environment & Food", style=_CARD_FLEX):
                df = m.datacollector.get_model_vars_dataframe()
                if not df.empty:
                    safe_altair_component(m, ["Total_Food"], width=1100, height=420)
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
