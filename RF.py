# %%
#Librerias
#Para procesamiento de datos
import pandas as pd
import numpy as np
import os
#Modelado
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
import seaborn as sns
from imblearn.over_sampling import SMOTE
#Graficas
import matplotlib.pyplot as plt

#Visualizacion
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# %%

df1 = pd.read_csv("Dataset.csv")
# Exclude the non-numeric column 'status' from the correlation calculation
numeric_df = df1.drop(columns=['status'])


Y1 = df1["status"].str.strip().map({"alive": 1, "failed": 0})

print("Distribución de clases en el dataset:")
print(Y1.value_counts())
print("\nEtiquetas: 1 = alive ✅, 0 = failed ❌")

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

print(mae)

# %%

f1_per_class = f1_score(Y_test, Y_pred, average=None)

def obtener_metricas():
    return {
        'MAE': mean_absolute_error(Y_test, Y_pred),
        'F1-Score (Failed)': f1_score(Y_test, Y_pred, average=None)[0],
        'F1-Score (Alive)': f1_score(Y_test, Y_pred, average=None)[1]
    }
# %%
#Se crea la matriz de confusion
cm = confusion_matrix(Y_test, Y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="viridis")  

plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Real")
plt.title("Matriz de Confusión")
ruta_matriz=os.path.join('static', 'cm.png')
plt.savefig(ruta_matriz)
plt.close()

#Esta de esta forma debido a la disparidad de los datos "failed", son muy pocos





def prediccion(datos_usuario):
    new_data = pd.DataFrame([datos_usuario])
    pred = model.predict(new_data)
    return pred

# %%
# Encontrar el árbol con menos nodos y menor profundidad
min_nodos = float('inf')
mejor_arbol = None

for i, arbol in enumerate(model.estimators_):
    nodos = arbol.tree_.node_count
    profundidad = arbol.tree_.max_depth
  
    
    # Guardar el árbol más pequeño
    if nodos < min_nodos:
        min_nodos = nodos
        mejor_arbol = arbol

# %%
from sklearn.tree import export_graphviz

# Exportar el árbol en formato DOT
export_graphviz(mejor_arbol, 
                out_file="arbol_completo.dot",
                feature_names=X.columns,
                filled=True, rounded=True)

#Usar dot -Tpng arbol_completo.dot -o arbol_completo.png


# %%
plt.figure(figsize=(30, 20)) 
plot_tree(mejor_arbol, 
          feature_names=X.columns, 
          class_names=["failed", "alive"], 
          filled=True, 
          rounded=True, 
          fontsize=10,
          max_depth=3)  # Muestra solo los primeros 3 niveles
plt.title("Árbol con Menos Nodos (Limitado a 3 niveles)", fontsize=18)
ruta_imagen = os.path.join('static', 'arbol.png')
plt.savefig(ruta_imagen, bbox_inches='tight')
plt.close()





casos = {
    "Empresa en riesgo (Failed)": {
        'activos': 1200,
        'depreciacion': 300,
        'inventario': 500,
        'ingresos netos': -100,
        'total de cuentas por cobrar': 200,
        'valor del mercado': 100,
        'deuda a largo plazo': 900,
        'beneficio bruto': 2000,
        'pasivos corrientes totales': 8000,
        'ganancias retenidas': -15000
    },
    "Empresa sana (Alive)": {
        'activos': 1000000,
        'depreciacion': 50000,
        'inventario': 250000,
        'ingresos netos': 150000,
        'total de cuentas por cobrar': 300000,
        'valor del mercado': 2000000,
        'deuda a largo plazo': 250000,
        'beneficio bruto': 800000,
        'pasivos corrientes totales': 300000,
        'ganancias retenidas': 500000
    },
    "Caso dudoso": {
        'activos': 1500,
        'depreciacion': 100,
        'inventario': 4000,
        'ingresos netos': 250,
        'total de cuentas por cobrar': 300,
        'valor del mercado': 1800,
        'deuda a largo plazo': 350,
        'beneficio bruto': 600,
        'pasivos corrientes totales': 280,
        'ganancias retenidas': 500
    },
    "Caso realista":{
        'activos': 350.0,  # 350,000 USD
        'depreciacion': 45.0,
        'inventario': 120.0,
        'ingresos netos': -85.0,  # Pérdida de 85,000 USD
        'total de cuentas por cobrar': 90.0,
        'valor del mercado': 280.0,
        'deuda a largo plazo': 320.0,  # Deuda > Activos
        'beneficio bruto': 95.0,
        'pasivos corrientes totales': 180.0,
        'ganancias retenidas': -150.0  # Pérdidas acumuladas
    },
    #El caso donde tiene todos datos uniformes falla debido a que el modelo no fue entrenado para estos casos
    "Caso 1":{
    'activos': 1,  
    'depreciacion': 1,
    'inventario': 1,
    'ingresos netos': 1,  
    'total de cuentas por cobrar': 1,
    'valor del mercado': 1,
    'deuda a largo plazo': 1,  
    'beneficio bruto': 1,
    'pasivos corrientes totales': 1,
    'ganancias retenidas': 1  
    }
}

# Ejecutar cada caso
for nombre, datos in casos.items():
    resultado = prediccion(datos)
    estado = "Alive ✅" if resultado[0] == 1 else "Failed ❌"
    print(f"{nombre} → Predicción: {estado}")

