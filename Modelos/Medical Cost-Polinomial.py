# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Cargar el dataset
df = pd.read_csv('insurance.csv')

# Exploración inicial de los datos
print("="*60)
print("ANÁLISIS EXPLORATORIO DE DATOS - SEGUROS")
print("="*60)

print("1. Primeras filas del dataset:")
print(df.head())
print("\n2. Información del dataset:")
print(df.info())
print("\n3. Estadísticas descriptivas:")
print(df.describe())
print("\n4. Verificar valores nulos:")
print(df.isnull().sum())

# Análisis de variables categóricas
print("\n5. Variables categóricas:")
print("Sex:", df['sex'].unique())
print("Smoker:", df['smoker'].unique())
print("Region:", df['region'].unique())

# Preprocesamiento: Convertir variables categóricas a numéricas
df_processed = df.copy()
df_processed['sex'] = df_processed['sex'].map({'male': 0, 'female': 1})
df_processed['smoker'] = df_processed['smoker'].map({'no': 0, 'yes': 1})
df_processed = pd.get_dummies(df_processed, columns=['region'], drop_first=True)

print("\n6. Dataset después del preprocesamiento:")
print(df_processed.head())

# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(15, 10))

# Distribución de charges
plt.subplot(2, 3, 1)
plt.hist(df['charges'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribución de Cargos (Charges)')
plt.xlabel('Cargos')
plt.ylabel('Frecuencia')

# Distribución de age
plt.subplot(2, 3, 2)
plt.hist(df['age'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')

# Distribución de bmi
plt.subplot(2, 3, 3)
plt.hist(df['bmi'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
plt.title('Distribución de BMI')
plt.xlabel('BMI')
plt.ylabel('Frecuencia')

# Distribución de children
plt.subplot(2, 3, 4)
plt.hist(df['children'], bins=6, alpha=0.7, color='gold', edgecolor='black')
plt.title('Distribución de Hijos')
plt.xlabel('Número de Hijos')
plt.ylabel('Frecuencia')

# Boxplot de charges por smoker
plt.subplot(2, 3, 5)
df.boxplot(column='charges', by='smoker')
plt.title('Cargos por Estado de Fumador')
plt.suptitle('')
plt.xlabel('Fumador')
plt.ylabel('Cargos')

# Boxplot de charges por sex
plt.subplot(2, 3, 6)
df.boxplot(column='charges', by='sex')
plt.title('Cargos por Género')
plt.suptitle('')
plt.xlabel('Género')
plt.ylabel('Cargos')

plt.tight_layout()
plt.show()

# Matriz de correlación
correlation_matrix = df_processed.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# Scatter plots de variables importantes vs charges
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Age vs Charges
axes[0,0].scatter(df['age'], df['charges'], alpha=0.6, c=df['smoker'].map({'yes': 'red', 'no': 'blue'}), s=20)
axes[0,0].set_title('Edad vs Cargos (Rojo: Fumador, Azul: No fumador)')
axes[0,0].set_xlabel('Edad')
axes[0,0].set_ylabel('Cargos')

# BMI vs Charges
axes[0,1].scatter(df['bmi'], df['charges'], alpha=0.6, c=df['smoker'].map({'yes': 'red', 'no': 'blue'}), s=20)
axes[0,1].set_title('BMI vs Cargos (Rojo: Fumador, Azul: No fumador)')
axes[0,1].set_xlabel('BMI')
axes[0,1].set_ylabel('Cargos')

# Children vs Charges
axes[1,0].scatter(df['children'], df['charges'], alpha=0.6, c=df['smoker'].map({'yes': 'red', 'no': 'blue'}), s=20)
axes[1,0].set_title('Hijos vs Cargos (Rojo: Fumador, Azul: No fumador)')
axes[1,0].set_xlabel('Número de Hijos')
axes[1,0].set_ylabel('Cargos')

# Charges distribution by smoker
smoker_charges = df.groupby('smoker')['charges'].mean()
axes[1,1].bar(smoker_charges.index, smoker_charges.values, color=['blue', 'red'])
axes[1,1].set_title('Cargos Promedio por Estado de Fumador')
axes[1,1].set_xlabel('Fumador')
axes[1,1].set_ylabel('Cargos Promedio')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("MODELO DE REGRESIÓN POLINOMIAL")
print("="*60)

# Preparar los datos
X = df_processed.drop('charges', axis=1)
y = df_processed['charges']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")

# Crear pipeline para regresión polinomial
def train_polynomial_regression(degree=2):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    # Hacer predicciones
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Calcular métricas
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)

    return pipeline, y_pred_test, (mse_train, rmse_train, r2_train, mse_test, rmse_test, r2_test)

# Probar diferentes grados del polinomio
degrees = [1, 2, 3, 4]
results = {}

for degree in degrees:
    print(f"\nEntrenando modelo con grado {degree}...")
    model, y_pred, metrics = train_polynomial_regression(degree)
    results[degree] = {
        'model': model,
        'y_pred': y_pred,
        'metrics': metrics
    }

    mse_train, rmse_train, r2_train, mse_test, rmse_test, r2_test = metrics
    print(f"Grado {degree} - MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, R²: {r2_test:.4f}")

# Fórmula de regresión logística
print("\nFÓRMULA DE REGRESIÓN POLINOMIAL:")
print("Y = β₀ + Σ(βᵢ × Xᵢ) + Σ(βᵢⱼ × Xᵢ × Xⱼ) + Σ(βᵢⱼₖ × Xᵢ × Xⱼ × Xₖ) + ...")
print("Donde:")
print("- Y son los cargos médicos predichos")
print("- β₀ es el intercepto (término constante)")
print("- βᵢ son los coeficientes para términos lineales")
print("- βᵢⱼ son los coeficientes para términos de interacción de segundo orden")
print("- βᵢⱼₖ son los coeficientes para términos de interacción de tercer orden")
print("- Xᵢ son las características originales (age, sex, bmi, children, smoker, region)")
print("- Los términos polinomiales incluyen todas las combinaciones hasta el grado especificado")
print(f"\nPara el modelo de grado {best_degree}:")
print(f"- Número de características originales: {X.shape[1]}")
print(f"- Número de características polinomiales: {len(feature_names)}")
print(f"- Intercepto (β₀): {best_model.named_steps['regressor'].intercept_:.4f}")
print(f"- Coeficientes β: {len(coefficients)} valores")
print(f"- R² en prueba: {r2_test:.4f}")

# Seleccionar el mejor modelo basado en R²
best_degree = max(results.keys(), key=lambda x: results[x]['metrics'][5])
best_model = results[best_degree]['model']
best_y_pred = results[best_degree]['y_pred']
best_metrics = results[best_degree]['metrics']

mse_train, rmse_train, r2_train, mse_test, rmse_test, r2_test = best_metrics

print("\n" + "="*60)
print("MEJOR MODELO DE REGRESIÓN POLINOMIAL")
print("="*60)
print(f"Grado óptimo del polinomio: {best_degree}")
print(f"Error Cuadrático Medio (MSE) - Entrenamiento: {mse_train:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE) - Entrenamiento: {rmse_train:.2f}")
print(f"Coeficiente de Determinación (R²) - Entrenamiento: {r2_train:.4f}")
print(f"Error Cuadrático Medio (MSE) - Prueba: {mse_test:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE) - Prueba: {rmse_test:.2f}")
print(f"Coeficiente de Determinación (R²) - Prueba: {r2_test:.4f}")

# Visualización de resultados
plt.figure(figsize=(15, 10))

# Gráfico 1: Predicciones vs Valores reales
plt.subplot(2, 2, 1)
plt.scatter(y_test, best_y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title(f'Predicciones vs Valores Reales (Grado {best_degree})\nR² = {r2_test:.4f}')

# Gráfico 2: Residuos
residuals = y_test - best_y_pred
plt.subplot(2, 2, 2)
plt.scatter(best_y_pred, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')

# Gráfico 3: Distribución de residuos
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')

# Gráfico 4: Comparación de R² por grado
r2_scores = [results[deg]['metrics'][5] for deg in degrees]
plt.subplot(2, 2, 4)
plt.plot(degrees, r2_scores, marker='o', linestyle='-', color='purple')
plt.xlabel('Grado del Polinomio')
plt.ylabel('R² Score')
plt.title('R² por Grado del Polinomio')
plt.grid(True)

plt.tight_layout()
plt.show()

# Importancia de las características (coeficientes del modelo)
poly_features = best_model.named_steps['poly']
feature_names = poly_features.get_feature_names_out(X.columns)
coefficients = best_model.named_steps['regressor'].coef_

# Crear DataFrame con importancia de características
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False).head(15)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'], feature_importance['Absolute_Coefficient'])
plt.xlabel('Importancia Absoluta (Valor absoluto del coeficiente)')
plt.ylabel('Característica')
plt.title('Top 15 Características más Importantes')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Predicción de ejemplo
print("\n" + "="*60)
print("EJEMPLO DE PREDICCIÓN")
print("="*60)

# Crear datos de ejemplo
ejemplo = pd.DataFrame({
    'age': [40],
    'sex': [1],  # female
    'bmi': [30.0],
    'children': [2],
    'smoker': [0],  # no
    'region_northwest': [0],
    'region_southeast': [1],
    'region_southwest': [0]
})

# Asegurarse de que el orden de columnas sea el mismo
ejemplo = ejemplo[X.columns]

prediccion = best_model.predict(ejemplo)
print("Datos de ejemplo:")
print(f"Edad: {ejemplo['age'].values[0]}")
print(f"Sexo: {'Femenino' if ejemplo['sex'].values[0] == 1 else 'Masculino'}")
print(f"BMI: {ejemplo['bmi'].values[0]}")
print(f"Hijos: {ejemplo['children'].values[0]}")
print(f"Fumador: {'Sí' if ejemplo['smoker'].values[0] == 1 else 'No'}")
print(f"Región: Sureste")
print(f"\nCargo predicho: ${prediccion[0]:.2f}")

# Comparación con valores reales
print("\nComparación con algunos valores reales del dataset:")
print(df[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']].head(3))

# Análisis de sobreajuste
print("\n" + "="*60)
print("ANÁLISIS DE SOBREAJUSTE")
print("="*60)

train_r2_scores = [results[deg]['metrics'][2] for deg in degrees]
test_r2_scores = [results[deg]['metrics'][5] for deg in degrees]

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_r2_scores, marker='o', label='Entrenamiento', linewidth=2)
plt.plot(degrees, test_r2_scores, marker='o', label='Prueba', linewidth=2)
plt.xlabel('Grado del Polinomio')
plt.ylabel('R² Score')
plt.title('R² en Entrenamiento vs Prueba por Grado del Polinomio')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar diferencias entre train y test
for degree in degrees:
    train_r2 = results[degree]['metrics'][2]
    test_r2 = results[degree]['metrics'][5]
    diff = train_r2 - test_r2
    print(f"Grado {degree}: Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}, Diferencia = {diff:.4f}")

print("\n" + "="*60)
print("CONCLUSIONES")
print("="*60)
print("1. El modelo de regresión polinomial puede capturar relaciones no lineales en los datos.")
print("2. El grado óptimo del polinomio balancea entre complejidad y generalización.")
print("3. Variables como 'smoker' y 'age' tienen alta importancia predictiva.")
print("4. Es importante monitorear el sobreajuste al aumentar el grado del polinomio.")
print("5. El modelo puede ser útil para predecir costos de seguros médicos basado en características demográficas.")