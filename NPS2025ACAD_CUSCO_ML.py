# %% [markdown]
# # **Modelo de Machine Learning de los factores de Detracción en el Campus Cusco**
#
# Este cuaderno amplía `NPS2025ACAD_CUSCO.ipynb`. Allí se ajustó una Regresión Logística por bloque de
# preguntas; aquí se comparan varios modelos de Machine Learning con validación cruzada estratificada,
# se elige el mejor por bloque y se explica qué factores pesan más en que un estudiante del campus Cusco
# sea **DETRACTOR** (DETRACTOR = 1).
#
# Bloques analizados (los mismos de `NPS2025ACAD_CUSCO.ipynb`):
# 1. Satisfacción general por dimensión (SATIF_ACAD, SATISF_SERV, ...).
# 2. P4: subcomponentes de la satisfacción académica.
# 3. P26: comunicación e imagen de la universidad.
#
# Funciona en Google Colab (lee `DATANPS2025.csv` y `utilidadesML.py` de `MyDrive/Colab Notebooks`)
# y también en local (los busca en la carpeta del script o del cuaderno).

# %% [markdown]
# # **Importar paquetes**

# %%
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             roc_auc_score, RocCurveDisplay)

warnings.filterwarnings('ignore')
SEMILLA = 42

# %% [markdown]
# # **Preparación del acceso al Drive**

# %%
try:
    from google.colab import drive
    drive.mount('/content/drive')
    RUTA_DATOS = '/content/drive/MyDrive/Colab Notebooks'
except ImportError:
    # Ejecución local: los archivos están junto al script / cuaderno
    RUTA_DATOS = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Agregar la carpeta de utilidadesML al path de Python
sys.path.append(RUTA_DATOS)

import utilidadesML as ml

# %% [markdown]
# # **Lectura de los datos**

# %%
estudiantes = pd.read_csv(os.path.join(RUTA_DATOS, 'DATANPS2025.csv'), sep=";")

# %% [markdown]
# # **Filtrado del Campus Cusco**

# %%
estudiantes_CUSCO = estudiantes[estudiantes['X_CAMPUS'] == 'CZC'].copy()
print(f"Estudiantes del campus Cusco: {len(estudiantes_CUSCO)}")
print(estudiantes_CUSCO['DETRACTOR'].value_counts().rename({0: 'No detractor', 1: 'Detractor'}))
print(f"Proporción de detractores: {estudiantes_CUSCO['DETRACTOR'].mean():.1%}")

# %% [markdown]
# **Las clases están desbalanceadas (≈ 1 detractor por cada 4 no detractores).** Por eso:
# - se usa `class_weight='balanced'` en los modelos que lo permiten,
# - la partición entrenamiento/prueba es **estratificada**,
# - además de la exactitud (accuracy) se miran **ROC-AUC**, **recall** y **F1 de la clase detractor**,
#   que son las métricas que importan cuando queremos encontrar a los detractores.

# %% [markdown]
# # **Funciones del análisis**
#
# Se definen una sola vez y se aplican a cada bloque de preguntas.

# %%
def candidatos_modelo():
    """Modelos a comparar. Todos manejan el desbalance de clases."""
    return {
        'Regresión Logística': make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='liblinear', max_iter=500, random_state=50, class_weight='balanced')),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=SEMILLA,
            class_weight='balanced', n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEMILLA,
            class_weight='balanced'),
    }


def refinar_variables(data, columnas, titulo_bloque):
    """Pruebas t de diferencia de medias (utilidadesML) para quedarnos con las variables significativas."""
    X = data[columnas]
    XCopia = X.copy()
    for column in X.columns:
        title = f"Relación de {column} vs Detracción (DETRACTOR=1)"
        ml.analisis_variable_violin(data, "DETRACTOR", column, title)
        ml.prueba_t(data, 'DETRACTOR', column, 1, 0, XCopia)
    print(f"[{titulo_bloque}] Variables que pasan la prueba t: {list(XCopia.columns)}")
    return XCopia


def comparar_modelos(X_train, y_train):
    """Validación cruzada estratificada de 5 particiones sobre el conjunto de entrenamiento."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    filas = []
    for nombre, modelo in candidatos_modelo().items():
        res = cross_validate(modelo, X_train, y_train, cv=cv,
                             scoring=['roc_auc', 'recall', 'f1', 'accuracy'])
        filas.append({'Modelo': nombre,
                      'ROC-AUC': res['test_roc_auc'].mean(),
                      'Recall detractor': res['test_recall'].mean(),
                      'F1 detractor': res['test_f1'].mean(),
                      'Accuracy': res['test_accuracy'].mean()})
    tabla = pd.DataFrame(filas).set_index('Modelo').sort_values('ROC-AUC', ascending=False)
    print(tabla.round(3))
    return tabla


def evaluar_modelo(modelo, nombre, X_test, y_test):
    """Matriz de confusión, reporte de clasificación y curva ROC en el conjunto de prueba."""
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    print(f'Modelo elegido: {nombre}')
    print(f'Accuracy en prueba: {accuracy_score(y_test, y_pred):.3f}')
    print(f'ROC-AUC en prueba: {roc_auc_score(y_test, y_prob):.3f}')
    print(f'Reporte de clasificación:\n{classification_report(y_test, y_pred)}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Matriz de Confusión - {nombre}')
    axes[0].set_xlabel('Predicción')
    axes[0].set_ylabel('Valor Real')
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=nombre)
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='grey')
    axes[1].set_title('Curva ROC')
    plt.tight_layout()
    plt.show()


def importancia_variables(modelo, nombre, X_test, y_test):
    """Importancia por permutación: cuánto cae el ROC-AUC al desordenar cada variable."""
    imp = permutation_importance(modelo, X_test, y_test, scoring='roc_auc',
                                 n_repeats=20, random_state=SEMILLA, n_jobs=-1)
    importancias = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    print(importancias.sort_values(ascending=False).round(4))

    plt.figure(figsize=(10, max(4, 0.45 * len(importancias))))
    bars = plt.barh(importancias.index, importancias.values)
    for bar in bars:
        plt.annotate(f'{bar.get_width():.3f}', xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                     xytext=(3, 0), textcoords="offset points", va='center', fontsize=8)
    plt.xlabel('Caída media del ROC-AUC')
    plt.title(f'Importancia por permutación - {nombre}')
    plt.tight_layout()
    plt.show()
    return importancias.sort_values(ascending=False)


def analisis_shap(modelo, nombre, X_train, X_test):
    """Valores SHAP sobre la probabilidad de ser detractor."""
    import shap
    fondo = shap.kmeans(X_train, 50)
    muestra = X_test.sample(min(200, len(X_test)), random_state=SEMILLA)
    explainer = shap.KernelExplainer(lambda d: modelo.predict_proba(d)[:, 1], fondo)
    shap_values = explainer.shap_values(muestra, silent=True)

    plt.title(f'Importancia de las Características (SHAP) - {nombre}')
    shap.summary_plot(shap_values, muestra, feature_names=muestra.columns, plot_type="bar")
    plt.title(f'Importancia de las Características (SHAP) - {nombre}')
    shap.summary_plot(shap_values, muestra, feature_names=muestra.columns)


def modelar_bloque(data, columnas, titulo_bloque):
    """Flujo completo de un bloque: prueba t -> comparación -> evaluación -> explicación."""
    XCopia = refinar_variables(data, columnas, titulo_bloque)
    y = data['DETRACTOR']
    X_train, X_test, y_train, y_test = train_test_split(
        XCopia, y, test_size=0.20, random_state=SEMILLA, stratify=y)

    tabla = comparar_modelos(X_train, y_train)
    nombre = tabla.index[0]
    modelo = candidatos_modelo()[nombre].fit(X_train, y_train)

    evaluar_modelo(modelo, nombre, X_test, y_test)
    importancias = importancia_variables(modelo, nombre, X_test, y_test)
    analisis_shap(modelo, nombre, X_train, X_test)
    return {'tabla': tabla, 'modelo': nombre, 'importancias': importancias}


def matriz_correlacion(data, columnas, tam):
    plt.figure(figsize=(tam, tam))
    sns.heatmap(data[['DETRACTOR'] + columnas].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.show()


resultados = {}

# %% [markdown]
# # **1. Satisfacción general por dimensión**

# %%
cols_general = ['SATIF_ACAD', 'SATISF_SERV', 'SATISF_EXP', 'SATISF_REG_ACAD', 'SATISF_ACT_SOST',
                'SATISF_APY_LOGRO_ACAD', 'SATISF_INF_TEC', 'SATISF_CAMP_VIRT', 'SATISF_COM_IMAG']
matriz_correlacion(estudiantes_CUSCO, cols_general, 10)

# %% [markdown]
# **Pruebas t, comparación de modelos, evaluación e importancia de variables**
#
# En el gráfico SHAP de calor: **si ves puntos rojos en el lado izquierdo y azules en el derecho, valores
# altos de la característica disminuyen la probabilidad de ser detractor y valores bajos la aumentan.**

# %%
resultados['Satisfacción general'] = modelar_bloque(estudiantes_CUSCO, cols_general, 'Satisfacción general')

# %% [markdown]
# # **2. Factores de la Satisfacción Académica**
# **P3. ¿QUÉ TAN SATISFECHO TE ENCUENTRAS CON EL ÁREA ACADÉMICA DE LA UNIVERSIDAD CONTINENTAL?**

# %%
cols_academica = ['P4_1_1', 'P4_1_2', 'P4_1_3', 'P4_1_4', 'P4_1_5', 'P4_1_6', 'P4_1_7', 'P4_1_8',
                  'P4_2_1', 'P4_2_2', 'P4_3_1', 'P4_3_2', 'P4_3_3']
matriz_correlacion(estudiantes_CUSCO, cols_academica, 14)

# %%
resultados['Satisfacción académica (P4)'] = modelar_bloque(estudiantes_CUSCO, cols_academica,
                                                           'Satisfacción académica (P4)')

# %% [markdown]
# # **3. Factores de la Satisfacción con la Imagen y Comunicación de la universidad**
# **P26. ¿QUÉ TAN SATISFECHO TE ENCUENTRAS CON CADA UNO DE LOS ASPECTOS DE LA COMUNICACIÓN E IMAGEN DE LA UNIVERSIDAD CONTINENTAL?**

# %%
cols_imagen = ['P26_1', 'P26_2', 'P26_3', 'P26_4', 'P26_5', 'P26_6', 'P26_7']
matriz_correlacion(estudiantes_CUSCO, cols_imagen, 8)

# %%
resultados['Comunicación e imagen (P26)'] = modelar_bloque(estudiantes_CUSCO, cols_imagen,
                                                           'Comunicación e imagen (P26)')

# %% [markdown]
# # **Resumen de resultados**
#
# Mejor modelo de cada bloque (según ROC-AUC en validación cruzada) y los 3 factores con mayor
# importancia por permutación.

# %%
resumen = pd.DataFrame([
    {'Bloque': bloque,
     'Mejor modelo': r['modelo'],
     'ROC-AUC (CV)': round(r['tabla'].loc[r['modelo'], 'ROC-AUC'], 3),
     'Recall detractor (CV)': round(r['tabla'].loc[r['modelo'], 'Recall detractor'], 3),
     'Top 3 factores': ', '.join(r['importancias'].index[:3])}
    for bloque, r in resultados.items()
]).set_index('Bloque')
print(resumen.to_string())
