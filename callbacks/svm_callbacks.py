import time
from dash import Input, Output, State, dcc, html
import numpy as np
from sklearn.svm import SVC

from utils.data import generate_arsenal_data
import utils.figures as figs

def register_callbacks(app, figs, generate_arsenal_data):
    @app.callback(
        Output("slider-svm-parameter-gamma-coef", "marks"),
        [Input("slider-svm-parameter-gamma-power", "value")],
    )
    def update_slider_svm_parameter_gamma_coef(power):
        scale = 10 ** power
        return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


    @app.callback(
        Output("slider-svm-parameter-C-coef", "marks"),
        [Input("slider-svm-parameter-C-power", "value")],
    )
    def update_slider_svm_parameter_C_coef(power):
        scale = 10 ** power
        return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


    @app.callback(
        Output("slider-threshold", "value"),
        [Input("button-zero-threshold", "n_clicks")],
        [State("graph-sklearn-svm", "figure")],
    )
    def reset_threshold_center(n_clicks, figure):
        # check if z_data is a list of ndarray
        # if so, convert it to Numppu before calling min max
        try:
            if n_clicks and figure and "data" in figure and len(figure["data"]) > 0:
                z_data = figure["data"][0].get("z")

                if z_data is not None and isinstance(z_data, (list, np.ndarray)):
                    Z = np.array(z_data, dtype=np.float64)
                    return -Z.min() / (Z.max() - Z.min())
        except Exception as e:
            print("Threshold reset error")

        return 0.5


    # Disable Sliders if kernel not in the given list
    @app.callback(
        Output("slider-svm-parameter-degree", "disabled"),
        [Input("dropdown-svm-parameter-kernel", "value")],
    )
    def disable_slider_param_degree(kernel):
        return kernel != "poly"


    @app.callback(
        Output("slider-svm-parameter-gamma-coef", "disabled"),
        [Input("dropdown-svm-parameter-kernel", "value")],
    )
    def disable_slider_param_gamma_coef(kernel):
        return kernel not in ["rbf", "poly", "sigmoid"]


    @app.callback(
        Output("slider-svm-parameter-gamma-power", "disabled"),
        [Input("dropdown-svm-parameter-kernel", "value")],
    )
    def disable_slider_param_gamma_power(kernel):
        return kernel not in ["rbf", "poly", "sigmoid"]


    @app.callback(
        Output("svm-graph-container", "children"),
        Output("graph-line-roc-curve", "figure"),
        Output("confusion-matrix-table", "children"),
        [
            Input("dropdown-svm-parameter-kernel", "value"),
            Input("slider-svm-parameter-degree", "value"),
            Input("slider-svm-parameter-C-coef", "value"),
            Input("slider-svm-parameter-C-power", "value"),
            Input("slider-svm-parameter-gamma-coef", "value"),
            Input("slider-svm-parameter-gamma-power", "value"),
            Input("radio-svm-parameter-shrinking", "value"),
            Input("slider-threshold", "value"),
        ],
    )

    def update_svm_graph(
        kernel,
        degree,
        C_coef,
        C_power,
        gamma_coef,
        gamma_power,
        shrinking,
        threshold,
    ):
        t_start = time.time()
        h = 0.3  # step size in the mesh


        X_train, X_test, y_train, y_test = generate_arsenal_data()

        x_min = X_train[:, 0].min() - 0.5
        x_max = X_train[:, 0].max() + 0.5
        y_min = X_train[:, 1].min() - 0.5
        y_max = X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        C = C_coef * 10 ** C_power
        gamma = gamma_coef * 10 ** gamma_power

        if shrinking == "True":
            flag = True
        else:
            flag = False

        # Train SVM
        clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=flag)
        clf.fit(X_train, y_train)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        prediction_figure = figs.serve_prediction_plot(
            model=clf,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            Z=Z,
            xx=xx,
            yy=yy,
            mesh_step=h,
            threshold=threshold,
        )

        roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test, y_test=y_test)

        confusion_figure = figs.serve_table_confusion_matrix(
            model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
        )

        return (
            dcc.Graph(id="graph-sklearn-svm", figure=prediction_figure),
            roc_figure,
            confusion_figure,
        )

