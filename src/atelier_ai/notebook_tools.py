"""Shared notebook figure presentation and portable PNG export."""

from pathlib import Path
import plotly.express as px

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#285A89",
    "#B67824",
    "#687B38",
    "#9B596D",
    "#78629A",
]


def show_figure(figure, project_root, name, *, height=520):
    directory = Path(project_root) / "artifacts/figures"
    directory.mkdir(parents=True, exist_ok=True)
    figure.update_layout(
        template="plotly_white",
        colorway=px.defaults.color_discrete_sequence,
        width=1100,
        height=height,
        font={"family": "Arial", "size": 13},
        title={"font": {"size": 18}, "x": 0.02, "xanchor": "left"},
        margin={"l": 100, "r": 50, "t": 100, "b": 100},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    if figure.layout.yaxis.title.text == "count":
        figure.update_yaxes(title_text="Observed customers")
    figure.write_image(str(directory / f"{name}.png"), scale=1.25)
    figure.show(renderer="plotly_mimetype+png", config={"displaylogo": False})
