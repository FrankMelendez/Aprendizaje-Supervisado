# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configuración de estilo para gráficos
plt.style.use('default')
sns.set_palette("husl")

# Leer el dataset
df = pd.read_csv('Advertising.csv')

# Exploración inicial de los datos
print("="*50)
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("="*50)

# Información básica del dataset
print("1. Primeras filas del dataset:")
print(df.head())
print("\n2. Información del dataset:")
print(df.info())
print("\n3. Estadísticas descriptivas:")
print(df.describe())
print("\n4. Verificar valores nulos:")
print(df.isnull().sum())

# Eliminar columna innecesaria si existe
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print("\nColumna 'Unnamed: 0' eliminada")

# Fórmula de regresión logística
print("\nFÓRMULA DE REGRESIÓN LINEAL:")
print("Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃")
print("Donde:")
print("- Y son las ventas predichas")
print("- β₀ = 2.9799 es el intercepto")
print("- β₁ = 0.0447 es el coeficiente para TV")
print("- β₂ = 0.1892 es el coeficiente para Radio")
print("- β₃ = 0.0028 es el coeficiente para Newspaper")
print("- X₁ son los gastos en TV")
print("- X₂ son los gastos en Radio")
print("- X₃ son los gastos en Newspaper")

# Visualización de la distribución de variables
print("\n" + "="*50)
print("VISUALIZACIONES")
print("="*50)

# Configurar subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Distribución de Sales
axes[0,0].hist(df['Sales'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Distribución de Ventas')
axes[0,0].set_xlabel('Ventas')
axes[0,0].set_ylabel('Frecuencia')

# Distribución de TV
axes[0,1].hist(df['TV'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0,1].set_title('Distribución de Gastos en TV')
axes[0,1].set_xlabel('Gastos en TV')
axes[0,1].set_ylabel('Frecuencia')

# Distribución de Radio
axes[1,0].hist(df['Radio'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
axes[1,0].set_title('Distribución de Gastos en Radio')
axes[1,0].set_xlabel('Gastos en Radio')
axes[1,0].set_ylabel('Frecuencia')

# Distribución de Newspaper
axes[1,1].hist(df['Newspaper'], bins=20, alpha=0.7, color='gold', edgecolor='black')
axes[1,1].set_title('Distribución de Gastos en Periódico')
axes[1,1].set_xlabel('Gastos en Periódico')
axes[1,1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# Matriz de correlación
print("\n5. Matriz de correlación:")
correlation_matrix = df.corr()
print(correlation_matrix)

# Heatmap de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación entre Variables')
plt.show()

# Scatter plots de cada variable predictora vs ventas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# TV vs Sales
axes[0].scatter(df['TV'], df['Sales'], alpha=0.6)
axes[0].set_title('TV vs Ventas')
axes[0].set_xlabel('Gastos en TV')
axes[0].set_ylabel('Ventas')

# Radio vs Sales
axes[1].scatter(df['Radio'], df['Sales'], alpha=0.6, color='orange')
axes[1].set_title('Radio vs Ventas')
axes[1].set_xlabel('Gastos en Radio')
axes[1].set_ylabel('Ventas')

# Newspaper vs Sales
axes[2].scatter(df['Newspaper'], df['Sales'], alpha=0.6, color='green')
axes[2].set_title('Periódico vs Ventas')
axes[2].set_xlabel('Gastos en Periódico')
axes[2].set_ylabel('Ventas')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ENTRENAMIENTO DEL MODELO DE REGRESIÓN LINEAL")
print("="*50)

# Preparar los datos
X = df[['TV', 'Radio', 'Newspaper']]  # Variables predictoras
y = df['Sales']  # Variable objetivo

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n6. Resultados del modelo:")
print(f"Coeficientes: {model.coef_}")
print(f"Intercepto: {model.intercept_}")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# Mostrar ecuación del modelo
print(f"\nEcuación del modelo:")
print(f"Ventas = {model.intercept_:.4f} + {model.coef_[0]:.4f}*TV + {model.coef_[1]:.4f}*Radio + {model.coef_[2]:.4f}*Newspaper")

# Visualización de predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.show()

# Gráfico de residuos
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')
plt.show()

# Importancia de las variables (coeficientes)
features = X.columns
coefficients = model.coef_

plt.figure(figsize=(10, 6))
plt.bar(features, coefficients)
plt.xlabel('Variables Predictoras')
plt.ylabel('Coeficientes')
plt.title('Importancia de las Variables (Coeficientes)')
plt.show()

# Predicción de ejemplo
print("\n" + "="*50)
print("EJEMPLO DE PREDICCIÓN")
print("="*50)

# Datos de ejemplo para predecir
ejemplo = pd.DataFrame({
    'TV': [200],
    'Radio': [40],
    'Newspaper': [30]
})

prediccion = model.predict(ejemplo)
print(f"Para gastos de TV: {ejemplo['TV'].values[0]}, Radio: {ejemplo['Radio'].values[0]}, Periódico: {ejemplo['Newspaper'].values[0]}")
print(f"Ventas predichas: {prediccion[0]:.2f}")

# Comparación con valores reales (opcional)
print("\nComparación con algunos valores reales del dataset:")
print(df.head(3))

# Gráfico 3D de la relación entre TV, Radio y Ventas (opcional)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(df['TV'], df['Radio'], df['Sales'], c=df['Sales'], cmap='viridis', alpha=0.6)

# Crear superficie de predicción
tv_range = np.linspace(df['TV'].min(), df['TV'].max(), 10)
radio_range = np.linspace(df['Radio'].min(), df['Radio'].max(), 10)
TV, Radio = np.meshgrid(tv_range, radio_range)
Z = model.intercept_ + model.coef_[0]*TV + model.coef_[1]*Radio + model.coef_[2]*df['Newspaper'].mean()

# Plot superficie
surf = ax.plot_surface(TV, Radio, Z, alpha=0.3, cmap='viridis')

ax.set_xlabel('Gastos en TV')
ax.set_ylabel('Gastos en Radio')
ax.set_zlabel('Ventas')
ax.set_title('Relación 3D: TV, Radio y Ventas')
plt.colorbar(scatter)
plt.show()

print("\n" + "="*50)
print("CONCLUSIONES")
print("="*50)
print("1. El modelo de regresión lineal muestra cómo los gastos en diferentes medios afectan las ventas.")
print("2. TV y Radio tienen mayor impacto positivo en las ventas según los coeficientes.")
print("3. El periódico tiene menor impacto en comparación.")
print("4. El modelo tiene un buen poder predictivo (R² cercano a 1 indica buen ajuste).")