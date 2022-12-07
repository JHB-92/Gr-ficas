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
import statsmodels.api as sm
from statsmodels.api import add_constant
from statsmodels.api import OLS
import statsmodels.formula.api as sm

from graficar import grafica
import json

f = open("points.json", encoding="utf8")
graficas = json.load(f)
array_graphics = ["line", "scatter", "fill_line", "linear_regression"]
control = False

# FALTA DATOS
datos = pd.DataFrame(
    {
        "Potencia": [
            1625,
            2205,
            2865,
            4831,
            8251,
            12115,
            16735,
            21247,
            26658,
            35795,
            44752,
            53226,
            63727,
            76968,
            87538,
            96418,
            104353,
            113913,
            124521,
            130745,
            2908,
            3892,
            6153,
            9224,
            13374,
            17106,
            21607,
            32361,
            42412,
            52325,
            62856,
            78422,
            100985,
            128013,
            158121,
            180725,
            209178,
            224695,
            251122,
            280837,
            443,
            2317,
            3908,
            6391,
            8598,
            11378,
            11426,
            19723,
            28958,
            40665,
            53788,
            89213,
            110941,
            122686,
            143985,
            167432,
            197030,
            224111,
            240373,
            284247,
            1026,
            2054,
            2946,
            4480,
            7373,
            9778,
            16875,
            23811,
            30405,
            42870,
            68520,
            98666,
            119382,
            138462,
            158594,
            180824,
            201008,
            224159,
            250876,
            283611,
            10555,
            12096,
            16302,
            22514,
            29751,
            41060,
            46196,
            56137,
            67244,
            82674,
            106384,
            127508,
            138458,
            157470,
            171124,
            196315,
            220167,
            240275,
            253474,
            293406,
        ],
        "Error (%)": [
            17.6,
            54.2,
            28.4,
            35.5,
            35.0,
            36.9,
            84.0,
            -8.9,
            25.1,
            23.4,
            12.3,
            27.2,
            18.4,
            13.9,
            15.8,
            16.3,
            16.2,
            15.4,
            12.9,
            10.1,
            16.3,
            41.6,
            15.0,
            15.7,
            44.5,
            65.8,
            41.2,
            39.4,
            18.0,
            -1.1,
            -21.5,
            22.0,
            15.3,
            23.4,
            13.3,
            21.5,
            30.0,
            11.1,
            22.0,
            19.8,
            -40.7,
            -23.9,
            34.6,
            35.7,
            63.7,
            59.9,
            4.3,
            51.2,
            43.7,
            38.4,
            22.2,
            18.9,
            11.4,
            31.0,
            28.9,
            15.2,
            15.5,
            31.2,
            20.9,
            13.1,
            1.5,
            3.1,
            30.0,
            25.7,
            36.4,
            40.4,
            55.2,
            19.3,
            15.2,
            25.1,
            52.0,
            -1.6,
            43.9,
            -7.0,
            16.0,
            25.9,
            -9.4,
            8.6,
            12.9,
            24.7,
            17.1,
            100.3,
            18.7,
            38.6,
            12.7,
            77.9,
            43.2,
            7.1,
            24.2,
            20.5,
            9.4,
            61.1,
            26.0,
            22.5,
            23.3,
            17.1,
            -15.4,
            29.1,
            27.7,
            27.4,
        ],
    }
)

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

label_x = "Potencia (W)"
label_y = "% Desviación"
title = "Regresión Lineal Error (%)"
plt.ylabel(label_y)
plt.xlabel(label_x)
plt.title(title)
plt.legend()
plt.grid(axis="y", color="gray", linestyle="dashed")
plt.xlim(0, 310000)
plt.ylim(-50, 50)
plt.show()
plt.close()
