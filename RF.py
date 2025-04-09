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

df1 = pd.read_csv("Dataset1.csv")
# Exclude the non-numeric column 'status' from the correlation calculation
numeric_df = df1.drop(columns=['status'])


# %%
columns_to_drop = [
    "gastos operativos totales", "pasivos totales", "ingresos totales", 
    "ganancia antes de intereses", "activos totales", "ventas netas", 
    "EBITDA", "bienes vendidos"
]
df = df1.drop(columns=columns_to_drop)
# Usar df_temp sin afectar df1



# %%
#Se asignan las variables independientes y dependientes
#Independientes, contiene todo excepto "status"
X = df.drop(columns=["status"])  
#Dependiente, contiene status, se convierte a variables numericas
Y = df["status"].str.strip().map({"alive": 1, "failed": 0})

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


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

model.fit(X_train_res, Y_train_res)


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


# %%
data_1 = {
    'anno': 2008,
    'activos': 800000,
    'depreciacion': 50000,
    'inventario': 200000,  
    'ingresos netos': 30000,
    'total de cuentas por cobrar': 200000,
    'valor del mercado': 1000000,
    'deuda a largo plazo': 150000,
    'beneficio bruto': 400000,
    'pasivos corrientes totales': 100000, 
    'ganancias retenidas': 250000
}


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



