import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Cargar los datos
data = pd.read_csv('data.csv')

# Exploración inicial
print("Primeras filas del dataset:")
print(data.head())
print("\nInformación del dataset:")
print(data.info())
print("\nEstadísticas descriptivas:")
print(data.describe())
print("\nDistribución de diagnósticos:")
print(data['diagnosis'].value_counts())

# Preprocesamiento
# Convertir diagnóstico a binario (M=1, B=0)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Eliminar columna id y la columna con valores nulos
data = data.drop(columns=['id', 'Unnamed: 32'])

# Verificar valores nulos
print("\nValores nulos por columna:")
print(data.isnull().sum())

# Separar características y variable objetivo
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Fórmula de regresión logística
print("\nFÓRMULA DE REGRESIÓN LOGÍSTICA:")
print("P(Y=1|X) = 1 / (1 + e^(-(β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ)))")
print("Donde:")
print("- P(Y=1|X) es la probabilidad de que el tumor sea maligno")
print("- β₀ es el intercepto")
print("- β₁, β₂, ..., βₚ son los coeficientes de las características")
print("- X₁, X₂, ..., Xₚ son los valores de las características")

# Crear y entrenar el modelo
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.4f}")

# Coeficientes del modelo
print("\nCoeficientes del modelo:")
for i, coef in enumerate(model.coef_[0]):
    print(f"{X.columns[i]}: {coef:.4f}")

print(f"Intercepto: {model.intercept_[0]:.4f}")

# 1. Gráfico de barras - Distribución de diagnósticos
plt.figure(figsize=(10, 6))
sns.countplot(x=data['diagnosis'].map({1: 'Maligno', 0: 'Benigno'}))
plt.title('Distribución de Diagnósticos de Cáncer de Mama')
plt.xlabel('Diagnóstico')
plt.ylabel('Cantidad')
plt.show()
print("Gráfico 1: Distribución de diagnósticos - Muestra el balance entre casos malignos y benignos en el dataset")

# 2. Gráfico circular - Proporción de diagnósticos
plt.figure(figsize=(8, 8))
diagnosis_counts = data['diagnosis'].value_counts()
plt.pie(diagnosis_counts, labels=['Benigno', 'Maligno'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Proporción de Diagnósticos de Cáncer de Mama')
plt.show()
print("Gráfico 2: Proporción de diagnósticos - Visualización circular de la distribución de casos")

# 3. Gráfico estadístico - Matriz de correlación
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación (Primeras 10 Características)')
plt.tight_layout()
plt.show()
print("Gráfico 3: Matriz de correlación - Muestra las relaciones entre las diferentes características del dataset")

# 4. Gráfico de regresión lineal - Relación entre dos variables importantes
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['radius_mean'], y=data['texture_mean'], hue=data['diagnosis'].map({1: 'Maligno', 0: 'Benigno'}))
plt.title('Relación entre Radio Medio y Textura Media')
plt.xlabel('Radio Medio')
plt.ylabel('Textura Media')
plt.legend(title='Diagnóstico')
plt.show()
print("Gráfico 4: Regresión lineal visual - Muestra la relación entre dos variables importantes y cómo se separan los diagnósticos")

# 5. Gráfico 3D - PCA para visualización en 3 dimensiones
# Reducir a 3 componentes principales para visualización 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Crear gráfico 3D interactivo con Plotly
fig = px.scatter_3d(
    x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
    color=y.map({1: 'Maligno', 0: 'Benigno'}),
    labels={'color': 'Diagnóstico'},
    title='Visualización 3D - Componentes Principales'
)
fig.update_traces(marker=dict(size=3))
fig.show()
print("Gráfico 5: Visualización 3D con PCA - Muestra la separación de clases en un espacio tridimensional reducido")

# 6. Gráfico de importancia de características
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_[0])
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Top 10 Características más Importantes para el Modelo')
plt.xlabel('Importancia Absoluta')
plt.ylabel('Característica')
plt.show()
print("Gráfico 6: Importancia de características - Muestra las variables más influyentes en el modelo de regresión logística")

# 7. Gráfico de matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno'])
plt.title('Matriz de Confusión')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.show()
print("Gráfico 7: Matriz de confusión - Muestra el rendimiento del modelo en términos de verdaderos/falsos positivos/negativos")

# [Todo el código anterior permanece igual hasta la sección de evaluación]

# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluar el modelo con métricas adicionales
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score, precision_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nPrecisión del modelo: {accuracy:.4f}")
print(f"Precisión (Precision): {precision:.4f}")
print(f"Recuperación (Recall): {recall:.4f}")
print(f"Puntuación F1: {f1:.4f}")

# Coeficientes del modelo
print("\nCoeficientes del modelo:")
for i, coef in enumerate(model.coef_[0]):
    print(f"{X.columns[i]}: {coef:.4f}")

print(f"Intercepto: {model.intercept_[0]:.4f}")

# [Los gráficos 1-7 permanecen igual]

# 8. Gráfico de curvas de probabilidad y métricas
plt.figure(figsize=(16, 6))

# Subplot 1: Distribución de probabilidades
plt.subplot(1, 3, 1)
plt.hist(y_pred_prob[y_test == 0], bins=20, alpha=0.7, label='Benigno', color='green')
plt.hist(y_pred_prob[y_test == 1], bins=20, alpha=0.7, label='Maligno', color='red')
plt.xlabel('Probabilidad Predicha de ser Maligno')
plt.ylabel('Frecuencia')
plt.title('Distribución de Probabilidades Predichas')
plt.legend()

# Subplot 2: Curva ROC
plt.subplot(1, 3, 2)
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")

# Subplot 3: Métricas del modelo
plt.subplot(1, 3, 3)
metrics = ['Precisión', 'Recall', 'F1-Score', 'Precision']
values = [accuracy, recall, f1, precision]
colors = ['blue', 'green', 'red', 'orange']

bars = plt.bar(metrics, values, color=colors, alpha=0.7)
plt.ylim(0, 1)
plt.title('Métricas del Modelo')
plt.ylabel('Valor')

# Agregar valores en las barras
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
print("Gráfico 8: Curvas de probabilidad y métricas - Izquierda: Distribución de probabilidades; Centro: Curva ROC; Derecha: Métricas del modelo")

# 9. Gráfico de comparación de métricas para ambas clases
from sklearn.metrics import precision_recall_fscore_support

# Calcular métricas para cada clase
precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(10, 6))
x = np.arange(2)
width = 0.25

plt.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
plt.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)

plt.xlabel('Clase')
plt.ylabel('Valor')
plt.title('Métricas por Clase')
plt.xticks(x, ['Benigno (0)', 'Maligno (1)'])
plt.legend()
plt.ylim(0, 1.1)

# Agregar valores en las barras
for i in range(2):
    plt.text(i - width, precision_per_class[i] + 0.02, f'{precision_per_class[i]:.3f}', ha='center')
    plt.text(i, recall_per_class[i] + 0.02, f'{recall_per_class[i]:.3f}', ha='center')
    plt.text(i + width, f1_per_class[i] + 0.02, f'{f1_per_class[i]:.3f}', ha='center')

plt.tight_layout()
plt.show()
print("Gráfico 9: Métricas por clase - Muestra Precision, Recall y F1-Score para cada clase (Benigno y Maligno)")

# Reporte de clasificación detallado
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN DETALLADO CON MÉTRICAS ADICIONALES")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))

# Resumen del modelo con métricas adicionales
print("\n" + "="*60)
print("RESUMEN COMPLETO DEL MODELO DE REGRESIÓN LOGÍSTICA")
print("="*60)
print(f"Precisión (Accuracy): {accuracy:.4f}")
print(f"Precisión (Precision - Clase Maligna): {precision:.4f}")
print(f"Recuperación (Recall - Clase Maligna): {recall:.4f}")
print(f"Puntuación F1 (Clase Maligna): {f1:.4f}")
print(f"Número de características: {X.shape[1]}")
print(f"Tamaño del dataset de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del dataset de prueba: {X_test.shape[0]} muestras")
print(f"Coeficientes estimados: {len(model.coef_[0])}")
print(f"Intercepto: {model.intercept_[0]:.4f}")

# Análisis interpretativo de las métricas
print("\n" + "="*60)
print("INTERPRETACIÓN DE LAS MÉTRICAS")
print("="*60)
print("• Precisión (Accuracy): Proporción de predicciones correctas")
print("• Precision: De los predichos como malignos, cuántos realmente lo son")
print("• Recall: De los realmente malignos, cuántos fueron detectados")
print("• F1-Score: Media armónica entre Precision y Recall")

if recall > 0.9:
    print("✓ Excelente Recall: El modelo detecta bien los casos malignos")
elif recall > 0.7:
    print("✓ Buen Recall: El modelo detecta razonablemente los casos malignos")
else:
    print("⚠ Recall mejorable: Podrían escaparse algunos casos malignos")

if precision > 0.9:
    print("✓ Excelente Precision: Pocos falsos positivos")
elif precision > 0.7:
    print("✓ Buena Precision: Falsos positivos controlados")
else:
    print("⚠ Precision mejorable: Muchos falsos positivos")
# Reporte de clasificación detallado
print("\n" + "="*50)
print("REPORTE DE CLASIFICACIÓN DETALLADO")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))

# Resumen del modelo
print("\n" + "="*50)
print("RESUMEN DEL MODELO DE REGRESIÓN LOGÍSTICA")
print("="*50)
print(f"Precisión: {accuracy:.4f}")
print(f"Número de características: {X.shape[1]}")
print(f"Tamaño del dataset de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del dataset de prueba: {X_test.shape[0]} muestras")
print(f"Coeficientes estimados: {len(model.coef_[0])}")
print(f"Intercepto: {model.intercept_[0]:.4f}")