Caso de estudio — Grupo 10

Asignatura: Estadística Aplicada
Universidad: Universidad de Santiago de Chile
Año: 2026

Descripción

Proyecto de predicción del retraso de llegada de vuelos comerciales durante 2024, utilizando modelos de regresión lineal, regularización, GAM, árboles y ensambles.

Datos

Se utiliza flight_data_2024_sample.csv, que contiene inicialmente 10.000 observaciones. Tras excluir vuelos cancelados y desviados, se emplean 9.836 registros.

Fuente: Flight Data 2024 — Kaggle.

Requisitos

R y RStudio.

Paquetes: car, dplyr, ggplot2, caret, lmtest, sandwich, glmnet, mgcv, rpart, rpart.plot, randomForest y gbm.

Instrucciones de reproducción

Descargar el repositorio.

Abrir RStudio y establecer como carpeta de trabajo la ubicación del script y del CSV.

Instalar los paquetes requeridos que no estén disponibles.

Verificar que el script tenga FOLDS_POR_FECHA <- TRUE.

Ejecutar el script completo, desde el inicio hasta el final.

El análisis utiliza la semilla 123, una partición temporal de entrenamiento y test, y cinco folds de validación por fechas completas.

Resultados

El script genera la carpeta salidas_informe_fechas, que contiene métricas de validación y test, hiperparámetros, gráficos y archivos de reproducibilidad.

El modelo final seleccionado es GAM.

Uso de IA

Se utilizó ChatGPT (OpenAI) como apoyo para la revisión del código, la redacción y la elaboración de figuras. Los cálculos estadísticos se realizaron en R.
