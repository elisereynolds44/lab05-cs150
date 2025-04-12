from dash import html, dcc
import utils.dash_reusable_components as drc

layout = html.Div(
    id="body",
    children=[
        html.Div(
            className="banner",
            children=[
                html.Div(
                    className="container scalable",
                    children=[
                        html.H2(
                            id="banner-title",
                            style={"fontFamily": "'Playfair Display', sans-serif"},
                            children=[
                                html.A(
                                    "Arsenal Match Win Predictor",
                                    style={"text-decoration": "none", "color": "inherit"},
                                )
                            ],
                        ),
                        html.H4(
                            "CS 150 - Elise Reynolds",
                            style={"color": "black", "marginTop": "0.5rem", "fontWeight": "normal"},
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="app-container",
            style={"display": "flex"},
            children=[
                html.Div(
                    id="left-column",
                    style={"width": "30%", "paddingRight": "20px"},
                    children=[
                        drc.Card(
                            id="button-card",
                            children=[
                                drc.NamedSlider(
                                    name="Threshold",
                                    id="slider-threshold",
                                    min=0,
                                    max=1,
                                    step=0.01,
                                    value=0.5,
                                    marks={i / 10: str(i / 10) for i in range(11)},
                                ),
                                html.Button("Reset Threshold", id="button-zero-threshold"),
                            ],
                        ),
                        drc.Card(
                            id="last-card",
                            children=[
                                drc.NamedDropdown(
                                    name="Kernel",
                                    id="dropdown-svm-parameter-kernel",
                                    options=[
                                        {"label": "Radial basis function (RBF)", "value": "rbf"},
                                        {"label": "Linear", "value": "linear"},
                                        {"label": "Polynomial", "value": "poly"},
                                        {"label": "Sigmoid", "value": "sigmoid"},
                                    ],
                                    value="rbf",
                                    clearable=False,
                                    searchable=False,
                                ),
                                drc.NamedSlider(
                                    name="Cost (C)",
                                    id="slider-svm-parameter-C-power",
                                    min=-2,
                                    max=4,
                                    value=0,
                                    marks={i: f"{10 ** i}" for i in range(-2, 5)},
                                ),
                                drc.FormattedSlider(
                                    id="slider-svm-parameter-C-coef", min=1, max=9, value=1
                                ),
                                drc.NamedSlider(
                                    name="Degree",
                                    id="slider-svm-parameter-degree",
                                    min=2,
                                    max=10,
                                    value=3,
                                    step=1,
                                    marks={str(i): str(i) for i in range(2, 11, 2)},
                                ),
                                drc.NamedSlider(
                                    name="Gamma",
                                    id="slider-svm-parameter-gamma-power",
                                    min=-5,
                                    max=0,
                                    value=-1,
                                    marks={i: f"{10 ** i}" for i in range(-5, 1)},
                                ),
                                drc.FormattedSlider(
                                    id="slider-svm-parameter-gamma-coef", min=1, max=9, value=5
                                ),
                                html.Div(
                                    id="shrinking-container",
                                    children=[
                                        html.P(children="Shrinking"),
                                        dcc.RadioItems(
                                            id="radio-svm-parameter-shrinking",
                                            labelStyle={
                                                "margin-right": "7px",
                                                "display": "inline-block",
                                            },
                                            options=[
                                                {"label": " Enabled", "value": "True"},
                                                {"label": " Disabled", "value": "False"},
                                            ],
                                            value="True",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="right-column",
                    style={"width": "70%"},
                    children=[
                        html.Div(
                            id="div-graphs",
                            children=[
                                html.Div(
                                    id="svm-graph-container",
                                    children=dcc.Loading(
                                        className="graph-wrapper",
                                        children=dcc.Graph(id="graph-sklearn-svm"),
                                    ),
                                ),
                                html.Div(
                                    id="graphs-container",
                                    children=[
                                        dcc.Loading(
                                            className="graph-wrapper",
                                            children=dcc.Graph(id="graph-line-roc-curve"),
                                        ),
                                        dcc.Loading(
                                            className="graph-wrapper",
                                            children=html.Div(id="confusion-matrix-table"),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
)