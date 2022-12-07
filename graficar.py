import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

## Procesado y modelado
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.api import add_constant
from statsmodels.api import OLS
import statsmodels.formula.api as sm


class grafica:
    def __init__(
        self,
        var_x="Variable x",
        var_y="Variable y",
        title="",
        label_x="",
        label_y="",
        x_limit=(),
        y_limit=(),
        finish="False",
        save=False,
        label="",
        color="",
    ):
        self.var_x = var_x
        self.var_y = var_y
        self.title = title
        self.label_x = label_x
        self.label_y = label_y
        self.save = save
        self.finish = finish
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.label = label
        self.color = color

    def scatter(self):

        plt.scatter(self.var_x, self.var_y, label=self.label, color=self.color)
        self.__conditions__()
        self.__finish__()

    def line(self):
        plt.plot(self.var_x, self.var_y, label=self.label, color=self.color)
        self.__conditions__()
        self.__finish__()

    def fill_line(self):
        plt.plot(self.var_x, self.var_y, color="black")
        plt.fill_between(self.var_x, self.var_y, label=self.label, color=self.color)
        self.__conditions__()
        self.__finish__()

    def linear_regression(self):

        datos = pd.DataFrame({"Potencia": self.var_x, "Error (%)": self.var_y})
        datos.head(3)

        # División de los datos en train y test
        # ==============================================================================
        X = datos[["Potencia"]]
        y = datos["Error (%)"]

        X_train, X_test, y_train, y_test = train_test_split(
            X.values.reshape(-1, 1),
            y.values.reshape(-1, 1),
            train_size=0.8,
            random_state=1234,
            shuffle=True,
        )

        # Creación del modelo
        # ==============================================================================
        modelo = LinearRegression()
        modelo.fit(X=X_train.reshape(-1, 1), y=y_train)

        # Información del modelo
        # ==============================================================================
        print("Intercept:", modelo.intercept_)
        print(
            "Coeficiente:",
            list(
                zip(
                    X.columns,
                    modelo.coef_.flatten(),
                )
            ),
        )
        print("Coeficiente de determinación R^2:", modelo.score(X, y))

        # Error de test del modelo
        # ==============================================================================
        predicciones = modelo.predict(X=X_test)
        print(
            predicciones[
                0:3,
            ]
        )

        rmse = mean_squared_error(y_true=y_test, y_pred=predicciones, squared=False)
        print("")
        print(f"El error (rmse) de test es: {rmse}")

        # Creación del modelo utilizando matrices como en scikitlearn
        # ==============================================================================
        # A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
        X_train = add_constant(X_train, prepend=True)
        modelo = OLS(
            endog=y_train,
            exog=X_train,
        )
        modelo = modelo.fit()
        print(modelo.summary())

        # Intervalos de confianza para los coeficientes del modelo
        # ==============================================================================
        modelo.conf_int(alpha=0.05)

        # Predicciones con intervalo de confianza del 95%
        # ==============================================================================
        predicciones = modelo.get_prediction(exog=X_train).summary_frame(alpha=0.05)
        predicciones.head(4)

        # Predicciones con intervalo de confianza del 95%
        # ==============================================================================
        predicciones = modelo.get_prediction(exog=X_train).summary_frame(alpha=0.05)
        predicciones["x"] = X_train[:, 1]
        predicciones["y"] = y_train
        predicciones = predicciones.sort_values("x")

        # Gráfico del modelo
        # ==============================================================================
        fig, ax = plt.subplots(figsize=(6, 3.84))

        ax.scatter(predicciones["x"], predicciones["y"], marker="o", color="gray")
        ax.plot(predicciones["x"], predicciones["mean"], linestyle="-", label="OLS")
        ax.plot(
            predicciones["x"],
            predicciones["mean_ci_lower"],
            linestyle="--",
            color="red",
            label="95% CI",
        )
        ax.plot(
            predicciones["x"],
            predicciones["mean_ci_upper"],
            linestyle="--",
            color="red",
        )
        ax.fill_between(
            predicciones["x"],
            predicciones["mean_ci_lower"],
            predicciones["mean_ci_upper"],
            alpha=0.1,
        )
        self.__conditions__()
        self.__finish__()

    def __conditions__(self):
        plt.ylabel(self.label_y)
        plt.xlabel(self.label_x)
        plt.title(self.title)
        plt.legend()
        plt.grid(axis="y", color="gray", linestyle="dashed")
        plt.xlim(self.x_limit)
        plt.ylim(self.y_limit)

    def __save__(self):
        plt.savefig(self.title + ".png", dpi=800)
        plt.close()

    def __finish__(self):
        if self.save == True:
            self.__save__()
        if self.finish == True:
            plt.show()
