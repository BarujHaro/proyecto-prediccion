# %%
#Librerias
#Para procesamiento de datos
import pandas as pd
import numpy as np
import os
#Modelado
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
import seaborn as sns
from imblearn.over_sampling import SMOTE


# %%

df1 = pd.read_csv("Dataset.csv")
# Exclude the non-numeric column 'status' from the correlation calculation
numeric_df = df1.drop(columns=['status'])


Y1 = df1["status"].str.strip().map({"alive": 1, "failed": 0})

# %%
columns_to_drop = [
    "anno",
    "gastos operativos totales", "pasivos totales", "ingresos totales", 
    "ganancia antes de intereses", "activos totales", "ventas netas", 
    "EBITDA", "bienes vendidos"
]
df = df1.drop(columns=columns_to_drop)

X = df.drop(columns=["status"])
Y = df["status"].str.strip().map({"alive": 1, "failed": 0})

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)


smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)


model = RandomForestClassifier(
    n_estimators=50,              # Aumentar el número de árboles
    max_depth=10,                 # Reducir la profundidad máxima
    min_samples_split=8,         # Relajar la división de nodos
    min_samples_leaf=3,           # Relajar el número de muestras por hoja
    class_weight='balanced',      # Balancear clases
    random_state=42
)

def obtener_parametros_modelo(modelo):
    params = modelo.get_params()
    
    # Parámetros clave que quieres mostrar (personaliza según necesites)
    parametros_clave = {
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'class_weight': str(params['class_weight']),
        'random_state': params['random_state']
    }
    
    return parametros_clave

model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, Y_pred)



# %%

f1_per_class = f1_score(Y_test, Y_pred, average=None)

def obtener_metricas():
    return {
        'MAE': mean_absolute_error(Y_test, Y_pred),
        'F1-Score (Failed)': f1_score(Y_test, Y_pred, average=None)[0],
        'F1-Score (Alive)': f1_score(Y_test, Y_pred, average=None)[1]
    }


def prediccion(datos_usuario):
    new_data = pd.DataFrame([datos_usuario])
    pred = model.predict(new_data)
    return pred




importancias = model.feature_importances_
importancias_dict = dict(zip(X.columns, importancias))
promedios_alive = X[Y == 1].mean()

def explicar_fallo(datos_usuario):
    pred = model.predict(pd.DataFrame([datos_usuario]))[0]
    if pred == 1:
        return "La empresa se considera sana"
    
    explicaciones = []
    importancias = model.feature_importances_
    importantes = sorted(zip(X.columns, importancias), key=lambda x: x[1], reverse=True)

    promedios_alive = X[Y == 1].mean()

    for feature, importancia in importantes[:3]:  # Top 3 causas
        valor_usuario = datos_usuario[feature]
        promedio_sano = promedios_alive[feature]
        
        if valor_usuario < promedio_sano * 0.5:
            explicaciones.append(f"- El valor de {feature} es muy bajo.\n")
        

    if not explicaciones:
        explicaciones.append("Los valores ingresados no coinciden con el perfil de empresas exitosas.\n")

    mensaje = "La empresa podría fallar por las siguientes razones:\n" + "\n".join(explicaciones)
    return mensaje



