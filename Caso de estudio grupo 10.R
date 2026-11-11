# ============================================================
# CONFIGURACION DEL EXPERIMENTO
# ============================================================
# FALSE: folds por observaciones cronologicas (reproduce la tabla historica).
# TRUE: folds por fechas completas (protocolo recomendado sin fechas compartidas).
# Reentrene todos los modelos al cambiar este parametro: no reutilice
# las metricas exportadas con otro valor.
FOLDS_POR_FECHA <- TRUE
stopifnot(is.logical(FOLDS_POR_FECHA),length(FOLDS_POR_FECHA)==1L,
          !is.na(FOLDS_POR_FECHA))

# ============================================================
# 0. INSTALACION DE PAQUETES
# ============================================================

# install.packages("car")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("caret")
# install.packages("lmtest")
# install.packages("sandwich")
# install.packages("glmnet")
# install.packages("mgcv")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("randomForest")
# install.packages("gbm")


# ============================================================
# 1. PAQUETES
# ============================================================

library(car)
library(dplyr)
library(ggplot2)
library(caret)
library(lmtest)
library(sandwich)
library(glmnet)
library(mgcv)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)


# ============================================================
# 2. IMPORTACION DE DATOS
# ============================================================
# Detectar ubicación del script si se ejecuta mediante Rscript.
# En RStudio, abrir proyecto/carpeta de trabajo que contenga script y CSV.
argumento_archivo <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(argumento_archivo) > 0L) {
  directorio_script <- dirname(normalizePath(sub("^--file=", "", argumento_archivo[[1L]])))
  setwd(directorio_script)
}
set.seed(123)
opciones_csv <- c("flight_data_2024_sample.csv",
                   "flight_data_2024_sample (1) (1).csv")
rutas_existentes <- opciones_csv[file.exists(opciones_csv)]
if (length(rutas_existentes) == 0L) {
  stop("No se encontró el CSV en el directorio de trabajo: ", getwd(),
       ". Ubique el CSV junto al script o establezca setwd() antes de ejecutar.")
}
ruta_csv <- rutas_existentes[[1L]]
message("Archivo CSV: ", normalizePath(ruta_csv))

datos <- read.csv(
  ruta_csv
)


# ============================================================
# 3. REVISION GENERAL
# ============================================================

dim(datos)
names(datos)
glimpse(datos)


# ============================================================
# 4. REVISION DE VUELOS CANCELADOS Y DESVIADOS
# ============================================================

table(datos$cancelled)
table(datos$diverted)


# ============================================================
# 5. FILTRAR VUELOS VALIDOS
# ============================================================

datos_modelo <- datos %>%
  filter(
    cancelled == 0,
    diverted == 0
  )

dim(datos_modelo)


# ============================================================
# 6. SELECCION DE VARIABLES
# ============================================================

datos_modelo <- datos_modelo %>%
  select(
    fl_date,
    arr_delay,
    month,
    day_of_month,
    day_of_week,
    crs_dep_time,
    crs_arr_time,
    dep_delay,
    taxi_out,
    crs_elapsed_time,
    distance
  )

dim(datos_modelo)


# ============================================================
# 7. TIPO DE VARIABLES
# ============================================================

datos_modelo <- datos_modelo %>%
  mutate(
    
    # Fecha
    fl_date = as.Date(fl_date),
    
    # Copia numerica del mes para GAM
    month_num = as.numeric(month),
    
    # Variables categoricas
    month = as.factor(month),
    day_of_week = as.factor(day_of_week)
  )

str(datos_modelo)


# ============================================================
# 8. REVISION DE VALORES FALTANTES
# ============================================================

colSums(is.na(datos_modelo))

round(
  colMeans(is.na(datos_modelo)) * 100,
  2
)


# ============================================================
# 9. AUDITORIA DE LA VARIABLE RESPUESTA
# ============================================================

summary(datos_modelo$arr_delay)

sum(is.na(datos_modelo$arr_delay))


# ============================================================
# 10. DISTRIBUCION DE LA VARIABLE RESPUESTA
# ============================================================

ggplot(
  datos_modelo,
  aes(x = arr_delay)
) +
  geom_histogram(
    bins = 50
  ) +
  labs(
    title = "Distribución del retraso de llegada",
    x = "Retraso de llegada (minutos)",
    y = "Frecuencia"
  ) +
  theme_minimal()


# ============================================================
# 11. ORDEN CRONOLOGICO
# ============================================================

datos_modelo <- datos_modelo %>%
  arrange(fl_date)

range(datos_modelo$fl_date)


# ============================================================
# 12. PARTICION TEMPORAL TRAIN / TEST
# ============================================================

# Se dejan las fechas mas recientes exclusivamente para test.

fechas <- sort(
  unique(datos_modelo$fl_date)
)

fecha_corte <- fechas[
  floor(length(fechas) * 0.80)
]

fecha_corte


# TRAIN: periodo antiguo

train <- datos_modelo %>%
  filter(
    fl_date <= fecha_corte
  )


# TEST: periodo mas reciente

test <- datos_modelo %>%
  filter(
    fl_date > fecha_corte
  )


dim(train)
dim(test)

range(train$fl_date)
range(test$fl_date)


# ============================================================
# 13. REVISION DE LA PARTICION
# ============================================================

nrow(datos_modelo)

nrow(train)
nrow(test)

nrow(train) / nrow(datos_modelo)
nrow(test) / nrow(datos_modelo)


# ============================================================
# 14. DISTRIBUCION DE LA RESPUESTA EN TRAIN
# ============================================================

summary(train$arr_delay)

ggplot(
  train,
  aes(x = arr_delay)
) +
  geom_histogram(
    bins = 50
  ) +
  labs(
    title = "Distribución del retraso de llegada en train",
    x = "Retraso de llegada (minutos)",
    y = "Frecuencia"
  ) +
  theme_minimal()


# ============================================================
# 15. PREPARAR VALIDACION TEMPORAL
# ============================================================

train <- train %>%
  arrange(fl_date)

n <- nrow(train)


# ============================================================
# 16. FOLDS TEMPORALES - PARAMETRO FOLDS_POR_FECHA
# ============================================================

if (FOLDS_POR_FECHA) {
  # Ventanas sobre dias: ningun dia se comparte entre train y validacion.
  fechas_cv <- sort(unique(train$fl_date))
  cortes <- caret::createTimeSlices(
    fechas_cv,
    initialWindow = floor(length(fechas_cv) * 0.50),
    horizon = floor(length(fechas_cv) * 0.10),
    fixedWindow = FALSE,
    skip = floor(length(fechas_cv) * 0.10) - 1
  )
  folds_temporales <- list(
    train = lapply(cortes$train, function(i) which(train$fl_date %in% fechas_cv[i])),
    test  = lapply(cortes$test,  function(i) which(train$fl_date %in% fechas_cv[i]))
  )
} else {
  # Ventanas por filas ordenadas cronologicamente, como en el registro
  # original del equipo: pueden coincidir dias a ambos lados del corte.
  filas_cv <- seq_len(nrow(train))
  cortes <- caret::createTimeSlices(
    filas_cv,
    initialWindow = floor(length(filas_cv) * 0.50),
    horizon = floor(length(filas_cv) * 0.10),
    fixedWindow = FALSE,
    skip = floor(length(filas_cv) * 0.10) - 1
  )
  folds_temporales <- list(train=cortes$train,test=cortes$test)
}
message('Protocolo de folds: ', if (FOLDS_POR_FECHA) 'FECHAS COMPLETAS' else 'FILAS CRONOLOGICAS')
stopifnot(length(folds_temporales$train)==length(folds_temporales$test),
          length(folds_temporales$train)==5L)


# ============================================================
# 17. CONTROL DE VALIDACION
# ============================================================

control_cv <- caret::trainControl(
  method = "cv",
  index = folds_temporales$train,
  indexOut = folds_temporales$test,
  savePredictions = "final"
)


# ============================================================
# 18. REVISION DE LOS FOLDS
# ============================================================

length(folds_temporales$train)

sapply(
  folds_temporales$train,
  length
)

sapply(
  folds_temporales$test,
  length
)
FOLDS_POR_FECHA
sapply(folds_temporales$train, length)
sapply(folds_temporales$test, length)
sapply(seq_along(folds_temporales$train), function(i) {
  max(train$fl_date[folds_temporales$train[[i]]]) <
    min(train$fl_date[folds_temporales$test[[i]]])
})
# ============================================================
# 19. BASELINE CON VALIDACION TEMPORAL
# ============================================================

resultados_baseline_folds <- data.frame(
  Fold = 1:length(folds_temporales$train),
  RMSE = NA,
  MAE = NA,
  R2 = NA
)


for(i in 1:length(folds_temporales$train)) {
  
  idx_train <- folds_temporales$train[[i]]
  idx_valid <- folds_temporales$test[[i]]
  
  
  # Media calculada solamente con el pasado
  
  media_fold <- mean(
    train$arr_delay[idx_train]
  )
  
  
  # Prediccion baseline
  
  pred_fold <- rep(
    media_fold,
    length(idx_valid)
  )
  
  
  # Valores reales
  
  obs_fold <- train$arr_delay[idx_valid]
  
  
  # RMSE
  
  resultados_baseline_folds$RMSE[i] <- sqrt(
    mean(
      (obs_fold - pred_fold)^2
    )
  )
  
  
  # MAE
  
  resultados_baseline_folds$MAE[i] <- mean(
    abs(
      obs_fold - pred_fold
    )
  )
  
  
  # R2
  
  resultados_baseline_folds$R2[i] <- 1 -
    sum((obs_fold - pred_fold)^2) /
    sum((obs_fold - mean(obs_fold))^2)
}


resultados_baseline_folds


# ============================================================
# 20. METRICAS PROMEDIO BASELINE
# ============================================================

metricas_baseline <- data.frame(
  
  RMSE = mean(
    resultados_baseline_folds$RMSE
  ),
  
  MAE = mean(
    resultados_baseline_folds$MAE
  ),
  
  R2 = mean(
    resultados_baseline_folds$R2
  ),
  
  RMSESD = sd(
    resultados_baseline_folds$RMSE
  ),
  
  MAESD = sd(
    resultados_baseline_folds$MAE
  ),
  
  R2SD = sd(
    resultados_baseline_folds$R2
  )
)

metricas_baseline


# ============================================================
# 21. OLS INICIAL
# ============================================================

modelo_lm <- lm(
  arr_delay ~
    month +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time +
    distance,
  data = train
)

summary(modelo_lm)


# ============================================================
# 22. MULTICOLINEALIDAD OLS INICIAL
# ============================================================

vif(modelo_lm)


# ============================================================
# 23. RELACION DISTANCE Y CRS_ELAPSED_TIME
# ============================================================

cor(
  train$distance,
  train$crs_elapsed_time
)


# ============================================================
# 24. OLS DEFINITIVO SIN DISTANCE
# ============================================================

# Se elimina distance del OLS debido a la alta relacion
# con crs_elapsed_time.

modelo_lm_2 <- lm(
  arr_delay ~
    month +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time,
  data = train
)

summary(modelo_lm_2)


# ============================================================
# 25. MULTICOLINEALIDAD DEL OLS DEFINITIVO
# ============================================================

vif(modelo_lm_2)


# ============================================================
# 26. DIAGNOSTICO DE RESIDUOS
# ============================================================

dir.create('salidas_informe', recursive=TRUE, showWarnings=FALSE)
png('salidas_informe/diagnosticos_OLS.png', width=1600, height=1400, res=180)
par(mfrow = c(2, 2))
plot(modelo_lm_2)
dev.off()



# ============================================================
# 27. PRUEBA DE HETEROCEDASTICIDAD
# ============================================================

bptest(modelo_lm_2)


# ============================================================
# 28. ERRORES ESTANDAR ROBUSTOS HC3
# ============================================================

resultado_robusto <- coeftest(
  modelo_lm_2,
  vcov = vcovHC(
    modelo_lm_2,
    type = "HC3"
  )
)

resultado_robusto


# ============================================================
# 29. INTERVALOS DE CONFIANZA ROBUSTOS 95%
# ============================================================

intervalos_robustos <- coefci(
  modelo_lm_2,
  vcov. = vcovHC(
    modelo_lm_2,
    type = "HC3"
  ),
  level = 0.95
)

intervalos_robustos


# ============================================================
# 30. VALIDACION TEMPORAL OLS
# ============================================================

set.seed(123)

cv_ols <- caret::train(
  arr_delay ~
    month +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time,
  data = train,
  method = "lm",
  trControl = control_cv,
  metric = "RMSE"
)


# ============================================================
# 31. FUNCION COMUN DE METRICAS
# ============================================================

# Esta funcion se utilizara para que todos los modelos
# tengan exactamente la misma definicion de RMSE, MAE y R2.

calcular_metricas_folds <- function(predicciones) {
  
  predicciones %>%
    group_by(Resample) %>%
    summarise(
      
      RMSE = sqrt(
        mean((obs - pred)^2)
      ),
      
      MAE = mean(
        abs(obs - pred)
      ),
      
      R2 = 1 -
        sum((obs - pred)^2) /
        sum((obs - mean(obs))^2),
      
      .groups = "drop"
    )
}


# ============================================================
# 32. METRICAS OLS POR FOLD
# ============================================================

resultados_ols_folds <- calcular_metricas_folds(
  cv_ols$pred
)

resultados_ols_folds


# ============================================================
# 33. METRICAS PROMEDIO OLS
# ============================================================

metricas_ols <- data.frame(
  
  RMSE = mean(
    resultados_ols_folds$RMSE
  ),
  
  MAE = mean(
    resultados_ols_folds$MAE
  ),
  
  R2 = mean(
    resultados_ols_folds$R2
  ),
  
  RMSESD = sd(
    resultados_ols_folds$RMSE
  ),
  
  MAESD = sd(
    resultados_ols_folds$MAE
  ),
  
  R2SD = sd(
    resultados_ols_folds$R2
  )
)

metricas_ols


# ============================================================
# 34. COMPARACION INICIAL BASELINE VS OLS
# ============================================================

comparacion_inicial <- data.frame(
  
  Modelo = c(
    "Baseline",
    "OLS"
  ),
  
  RMSE = c(
    metricas_baseline$RMSE,
    metricas_ols$RMSE
  ),
  
  MAE = c(
    metricas_baseline$MAE,
    metricas_ols$MAE
  ),
  
  R2 = c(
    metricas_baseline$R2,
    metricas_ols$R2
  ),
  
  RMSESD = c(
    metricas_baseline$RMSESD,
    metricas_ols$RMSESD
  )
)

comparacion_inicial


# ============================================================
# 35. VALORES DE LAMBDA PARA REGULARIZACION
# ============================================================

valores_lambda <- 10^seq(
  from = -3,
  to = 1,
  length.out = 50
)


# ============================================================
# 36. RIDGE
# ============================================================

set.seed(123)

cv_ridge <- caret::train(
  arr_delay ~
    month +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time +
    distance,
  data = train,
  method = "glmnet",
  trControl = control_cv,
  metric = "RMSE",
  tuneGrid = expand.grid(
    alpha = 0,
    lambda = valores_lambda
  )
)

cv_ridge$bestTune


# ============================================================
# 37. LASSO
# ============================================================

set.seed(123)

cv_lasso <- caret::train(
  arr_delay ~
    month +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time +
    distance,
  data = train,
  method = "glmnet",
  trControl = control_cv,
  metric = "RMSE",
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = valores_lambda
  )
)

cv_lasso$bestTune


# ============================================================
# 38. ELASTIC NET
# ============================================================

set.seed(123)

cv_enet <- caret::train(
  arr_delay ~
    month +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time +
    distance,
  data = train,
  method = "glmnet",
  trControl = control_cv,
  metric = "RMSE",
  tuneGrid = expand.grid(
    alpha = c(
      0.25,
      0.50,
      0.75
    ),
    lambda = valores_lambda
  )
)

cv_enet$bestTune


# ============================================================
# 39. METRICAS RIDGE
# ============================================================

resultados_ridge_folds <- calcular_metricas_folds(
  cv_ridge$pred
)

metricas_ridge <- data.frame(
  
  RMSE = mean(resultados_ridge_folds$RMSE),
  MAE = mean(resultados_ridge_folds$MAE),
  R2 = mean(resultados_ridge_folds$R2),
  
  RMSESD = sd(resultados_ridge_folds$RMSE),
  MAESD = sd(resultados_ridge_folds$MAE),
  R2SD = sd(resultados_ridge_folds$R2)
)

metricas_ridge


# ============================================================
# 40. METRICAS LASSO
# ============================================================

resultados_lasso_folds <- calcular_metricas_folds(
  cv_lasso$pred
)

metricas_lasso <- data.frame(
  
  RMSE = mean(resultados_lasso_folds$RMSE),
  MAE = mean(resultados_lasso_folds$MAE),
  R2 = mean(resultados_lasso_folds$R2),
  
  RMSESD = sd(resultados_lasso_folds$RMSE),
  MAESD = sd(resultados_lasso_folds$MAE),
  R2SD = sd(resultados_lasso_folds$R2)
)

metricas_lasso


# ============================================================
# 41. METRICAS ELASTIC NET
# ============================================================

resultados_enet_folds <- calcular_metricas_folds(
  cv_enet$pred
)

metricas_enet <- data.frame(
  
  RMSE = mean(resultados_enet_folds$RMSE),
  MAE = mean(resultados_enet_folds$MAE),
  R2 = mean(resultados_enet_folds$R2),
  
  RMSESD = sd(resultados_enet_folds$RMSE),
  MAESD = sd(resultados_enet_folds$MAE),
  R2SD = sd(resultados_enet_folds$R2)
)

metricas_enet


# ============================================================
# 42. HIPERPARAMETROS SELECCIONADOS
# ============================================================

cv_ridge$bestTune
cv_lasso$bestTune
cv_enet$bestTune


# ============================================================
# 43. VERIFICAR VARIABLE NUMERICA DEL MES
# ============================================================

"month_num" %in% names(train)

summary(train$month_num)


# ============================================================
# 44. FUNCION DE VALIDACION TEMPORAL PARA GAM
# ============================================================

# El mes se utiliza numericamente en el GAM.
# Esto evita niveles categoricos futuros que no existian
# en algunos periodos de entrenamiento.

evaluar_gam <- function(k_valor) {
  
  predicciones <- data.frame()
  
  
  for(i in 1:length(folds_temporales$train)) {
    
    idx_train <- folds_temporales$train[[i]]
    idx_valid <- folds_temporales$test[[i]]
    
    
    modelo_temp <- mgcv::gam(
      arr_delay ~
        day_of_week +
        s(month_num, k = 5) +
        s(day_of_month, k = k_valor) +
        s(crs_dep_time, k = k_valor) +
        s(crs_arr_time, k = k_valor) +
        s(dep_delay, k = k_valor) +
        s(taxi_out, k = k_valor) +
        s(crs_elapsed_time, k = k_valor) +
        s(distance, k = k_valor),
      
      data = train[idx_train, ],
      method = "REML"
    )
    
    
    pred_temp <- predict(
      modelo_temp,
      newdata = train[idx_valid, ]
    )
    
    
    predicciones <- rbind(
      predicciones,
      
      data.frame(
        rowIndex = idx_valid,
        Resample = paste0("Fold", i),
        obs = train$arr_delay[idx_valid],
        pred = pred_temp
      )
    )
  }
  
  
  metricas_folds <- calcular_metricas_folds(
    predicciones
  )
  
  
  resumen <- data.frame(
    
    k = k_valor,
    
    RMSE = mean(
      metricas_folds$RMSE
    ),
    
    MAE = mean(
      metricas_folds$MAE
    ),
    
    R2 = mean(
      metricas_folds$R2
    ),
    
    RMSESD = sd(
      metricas_folds$RMSE
    )
  )
  
  
  return(
    list(
      pred = predicciones,
      folds = metricas_folds,
      resumen = resumen
    )
  )
}


# ============================================================
# 45. PROBAR DISTINTOS VALORES DE K
# ============================================================

gam_k5 <- evaluar_gam(5)

gam_k7 <- evaluar_gam(7)

gam_k9 <- evaluar_gam(9)


resultados_k_gam <- rbind(
  gam_k5$resumen,
  gam_k7$resumen,
  gam_k9$resumen
)

resultados_k_gam


# ============================================================
# 46. SELECCION AUTOMATICA DEL MEJOR K
# ============================================================

mejor_k <- resultados_k_gam$k[
  which.min(
    resultados_k_gam$RMSE
  )
]

mejor_k


# ============================================================
# 47. GUARDAR EL MEJOR GAM DE VALIDACION
# ============================================================

lista_gam <- list(
  "5" = gam_k5,
  "7" = gam_k7,
  "9" = gam_k9
)

resultado_gam <- lista_gam[[as.character(mejor_k)]]

# Predicciones temporales

pred_gam_cv <- resultado_gam$pred


# Resultados por fold

resultados_gam_folds <- resultado_gam$folds

resultados_gam_folds


# ============================================================
# 48. METRICAS PROMEDIO DEL GAM
# ============================================================

metricas_gam <- data.frame(
  
  RMSE = mean(
    resultados_gam_folds$RMSE
  ),
  
  MAE = mean(
    resultados_gam_folds$MAE
  ),
  
  R2 = mean(
    resultados_gam_folds$R2
  ),
  
  RMSESD = sd(
    resultados_gam_folds$RMSE
  ),
  
  MAESD = sd(
    resultados_gam_folds$MAE
  ),
  
  R2SD = sd(
    resultados_gam_folds$R2
  )
)

metricas_gam


# ============================================================
# 49. GAM DEFINITIVO
# ============================================================

# Una vez seleccionado k mediante validacion temporal,
# se ajusta el GAM sobre todo el conjunto train.

modelo_gam <- mgcv::gam(
  arr_delay ~
    day_of_week +
    s(month_num, k = 5) +
    s(day_of_month, k = mejor_k) +
    s(crs_dep_time, k = mejor_k) +
    s(crs_arr_time, k = mejor_k) +
    s(dep_delay, k = mejor_k) +
    s(taxi_out, k = mejor_k) +
    s(crs_elapsed_time, k = mejor_k) +
    s(distance, k = mejor_k),
  
  data = train,
  method = "REML"
)


# ============================================================
# 50. INTERPRETACION DEL GAM
# ============================================================

summary(modelo_gam)

png('salidas_informe/gam_diagnostico_todos.png', width=2200, height=1500, res=150)
plot(modelo_gam, pages=1, residuals=TRUE)
dev.off()
# ============================================================
# 51. INCERTIDUMBRE E INTERVALOS DE PREDICCION DEL GAM
# ============================================================

# Se obtienen las predicciones y su error estandar

pred_gam_intervalo <- predict(
  modelo_gam,
  newdata = train,
  type = "response",
  se.fit = TRUE
)


# Varianza residual estimada por el GAM

varianza_residual_gam <- summary(
  modelo_gam
)$scale


# Error asociado a una prediccion individual
# Se considera:
# 52. incertidumbre de la media estimada
# 53. variabilidad residual de una observacion futura

error_prediccion_gam <- sqrt(
  pred_gam_intervalo$se.fit^2 +
    varianza_residual_gam
)


# ============================================================
# 54. INTERVALO DE PREDICCION APROXIMADO AL 95%
# ============================================================

intervalos_pred_gam <- data.frame(
  
  Observado = train$arr_delay,
  
  Prediccion = pred_gam_intervalo$fit,
  
  Limite_Inferior =
    pred_gam_intervalo$fit -
    1.96 * error_prediccion_gam,
  
  Limite_Superior =
    pred_gam_intervalo$fit +
    1.96 * error_prediccion_gam
)


# Revisar primeras observaciones

head(intervalos_pred_gam)
# ============================================================
# 55. COBERTURA DE LOS INTERVALOS
# ============================================================

dentro_intervalo <- (
  intervalos_pred_gam$Observado >=
    intervalos_pred_gam$Limite_Inferior
) & (
  intervalos_pred_gam$Observado <=
    intervalos_pred_gam$Limite_Superior
)


cobertura_train_aparente <- mean(
  dentro_intervalo
)

# cobertura_train_aparente: diagnóstico IN-SAMPLE, no validar como test
# ============================================================
# 56. AMPLITUD PROMEDIO DEL INTERVALO
# ============================================================

amplitud_train_aparente <- mean(
  intervalos_pred_gam$Limite_Superior -
    intervalos_pred_gam$Limite_Inferior
)

# amplitud_train_aparente: diagnóstico IN-SAMPLE, no validar como test
# ============================================================
# 57. ARBOL DE REGRESION CON VALIDACION TEMPORAL
# ============================================================

set.seed(123)

cv_arbol <- caret::train(
  arr_delay ~
    month_num +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time +
    distance,
  
  data = train,
  
  method = "rpart",
  
  trControl = control_cv,
  
  metric = "RMSE",
  
  tuneLength = 15
)


# ============================================================
# 58. CP SELECCIONADO AUTOMATICAMENTE
# ============================================================

cv_arbol$bestTune

cv_arbol$results


# ============================================================
# 59. METRICAS TEMPORALES DEL ARBOL
# ============================================================

resultados_arbol_folds <- calcular_metricas_folds(
  cv_arbol$pred
)

resultados_arbol_folds


metricas_arbol <- data.frame(
  
  RMSE = mean(
    resultados_arbol_folds$RMSE
  ),
  
  MAE = mean(
    resultados_arbol_folds$MAE
  ),
  
  R2 = mean(
    resultados_arbol_folds$R2
  ),
  
  RMSESD = sd(
    resultados_arbol_folds$RMSE
  ),
  
  MAESD = sd(
    resultados_arbol_folds$MAE
  ),
  
  R2SD = sd(
    resultados_arbol_folds$R2
  )
)

metricas_arbol


# ============================================================
# 60. ARBOL DEFINITIVO
# ============================================================

modelo_arbol <- rpart(
  arr_delay ~
    month_num +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time +
    distance,
  
  data = train,
  
  method = "anova",
  
  control = rpart.control(
    cp = cv_arbol$bestTune$cp
  )
)


# ============================================================
# 61. VISUALIZACION DEL ARBOL
# ============================================================

png('salidas_informe/arbol.png', width=1700, height=1100, res=160)
rpart.plot(
  modelo_arbol,
  type = 2,
  extra = 101,
  fallen.leaves = TRUE
)
dev.off()

print(modelo_arbol)

modelo_arbol$variable.importance
# ============================================================
# 62. RANDOM FOREST CON VALIDACION TEMPORAL
# ============================================================

set.seed(123)

cv_rf <- caret::train(
  arr_delay ~
    month_num +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time +
    distance,
  
  data = train,
  
  method = "rf",
  
  trControl = control_cv,
  
  metric = "RMSE",
  
  tuneGrid = data.frame(
    mtry = 1:9  # todos los predictores; evita límite arbitrario en mtry=8
  ),
  
  ntree = 500,
  
  importance = TRUE
)


# ============================================================
# 63. SELECCION AUTOMATICA DE MTRY
# ============================================================

cv_rf$bestTune

# Ver resultados de todos los mtry probados

cv_rf$results[
  order(cv_rf$results$RMSE),
]


# ============================================================
# 64. METRICAS TEMPORALES RANDOM FOREST
# ============================================================

resultados_rf_folds <- calcular_metricas_folds(
  cv_rf$pred
)

resultados_rf_folds


metricas_rf <- data.frame(
  
  RMSE = mean(
    resultados_rf_folds$RMSE
  ),
  
  MAE = mean(
    resultados_rf_folds$MAE
  ),
  
  R2 = mean(
    resultados_rf_folds$R2
  ),
  
  RMSESD = sd(
    resultados_rf_folds$RMSE
  ),
  
  MAESD = sd(
    resultados_rf_folds$MAE
  ),
  
  R2SD = sd(
    resultados_rf_folds$R2
  )
)

metricas_rf


# ============================================================
# 65. RANDOM FOREST DEFINITIVO
# ============================================================

# caret ya ajusta el modelo final utilizando todo train
# con el mtry seleccionado mediante validacion temporal.

modelo_rf <- cv_rf$finalModel

modelo_rf

# ============================================================
# 66. IMPORTANCIA DE VARIABLES RANDOM FOREST
# ============================================================

importancia_rf <- randomForest::importance(
  modelo_rf
)

importancia_rf


png('salidas_informe/rf_importancia_doble.png', width=1700, height=1150, res=160)
varImpPlot(
  modelo_rf
)
dev.off()
# ============================================================
# 67. GRILLA DE HIPERPARAMETROS GRADIENT BOOSTING
# ============================================================

# Se prueban distintas combinaciones de:
# - numero de arboles
# - profundidad
# - shrinkage
# utilizando los mismos folds temporales.

grid_gbm <- expand.grid(
  
  n.trees = c(
    300,
    600,
    900
  ),
  
  interaction.depth = c(
    1,
    2,
    3
  ),
  
  shrinkage = c(
    0.01,
    0.05
  ),
  
  n.minobsinnode = 10
)


# ============================================================
# 68. GRADIENT BOOSTING CON VALIDACION TEMPORAL
# ============================================================

set.seed(123)

cv_gbm <- caret::train(
  
  arr_delay ~
    month_num +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time +
    distance,
  
  data = train,
  
  method = "gbm",
  
  trControl = control_cv,
  
  metric = "RMSE",
  
  tuneGrid = grid_gbm,
  
  verbose = FALSE
)


# ============================================================
# 69. HIPERPARAMETROS SELECCIONADOS
# ============================================================

# Mejor combinacion segun RMSE de validacion temporal

cv_gbm$bestTune


# Ver todas las combinaciones ordenadas
# desde menor a mayor RMSE

resultados_tuning_gbm <- cv_gbm$results[
  order(cv_gbm$results$RMSE),
]

resultados_tuning_gbm


# ============================================================
# 70. METRICAS DEL BOOSTING POR FOLD TEMPORAL
# ============================================================

# savePredictions = "final" guarda las predicciones
# correspondientes a la mejor combinacion seleccionada.

resultados_gbm_folds <- calcular_metricas_folds(
  cv_gbm$pred
)

resultados_gbm_folds


# ============================================================
# 71. METRICAS PROMEDIO DEL BOOSTING
# ============================================================

metricas_gbm <- data.frame(
  
  RMSE = mean(
    resultados_gbm_folds$RMSE
  ),
  
  MAE = mean(
    resultados_gbm_folds$MAE
  ),
  
  R2 = mean(
    resultados_gbm_folds$R2
  ),
  
  RMSESD = sd(
    resultados_gbm_folds$RMSE
  ),
  
  MAESD = sd(
    resultados_gbm_folds$MAE
  ),
  
  R2SD = sd(
    resultados_gbm_folds$R2
  )
)

metricas_gbm


# ============================================================
# 72. MODELO BOOSTING DEFINITIVO
# ============================================================

# caret vuelve a ajustar el modelo utilizando todo train
# con los hiperparametros seleccionados.

modelo_gbm <- cv_gbm$finalModel

modelo_gbm


# ============================================================
# 73. IMPORTANCIA DE VARIABLES BOOSTING
# ============================================================

importancia_gbm <- summary(
  modelo_gbm,
  n.trees = cv_gbm$bestTune$n.trees,
  plotit = FALSE
)

importancia_gbm
# ============================================================
# 74. REFINAMIENTO DE HIPERPARAMETROS BOOSTING
# ============================================================

# Como la mejor combinacion anterior quedo en los limites
# de la grilla, se amplia la busqueda alrededor de esa zona.

grid_gbm_refinado <- expand.grid(
  
  n.trees = c(
    900,
    1200,
    1500
  ),
  
  interaction.depth = c(
    3,
    4
  ),
  
  shrinkage = c(
    0.005,
    0.01,
    0.02
  ),
  
  n.minobsinnode = 10
)


# ============================================================
# 75. BOOSTING REFINADO CON VALIDACION TEMPORAL
# ============================================================

set.seed(123)

cv_gbm_refinado <- caret::train(
  
  arr_delay ~
    month_num +
    day_of_month +
    day_of_week +
    crs_dep_time +
    crs_arr_time +
    dep_delay +
    taxi_out +
    crs_elapsed_time +
    distance,
  
  data = train,
  
  method = "gbm",
  
  trControl = control_cv,
  
  metric = "RMSE",
  
  tuneGrid = grid_gbm_refinado,
  
  verbose = FALSE
)


# ============================================================
# 76. MEJOR COMBINACION REFINADA
# ============================================================

cv_gbm_refinado$bestTune


resultados_tuning_gbm_refinado <- cv_gbm_refinado$results[
  order(cv_gbm_refinado$results$RMSE),
]

resultados_tuning_gbm_refinado


# ============================================================
# 77. METRICAS TEMPORALES DEL BOOSTING REFINADO
# ============================================================

resultados_gbm_folds <- calcular_metricas_folds(
  cv_gbm_refinado$pred
)

resultados_gbm_folds


metricas_gbm <- data.frame(
  
  RMSE = mean(
    resultados_gbm_folds$RMSE
  ),
  
  MAE = mean(
    resultados_gbm_folds$MAE
  ),
  
  R2 = mean(
    resultados_gbm_folds$R2
  ),
  
  RMSESD = sd(
    resultados_gbm_folds$RMSE
  ),
  
  MAESD = sd(
    resultados_gbm_folds$MAE
  ),
  
  R2SD = sd(
    resultados_gbm_folds$R2
  )
)

metricas_gbm


# ============================================================
# 78. BOOSTING DEFINITIVO
# ============================================================

cv_gbm <- cv_gbm_refinado

modelo_gbm <- cv_gbm$finalModel

cv_gbm$bestTune
metricas_gbm
# ============================================================
# 79. PREPARAR PREDICCIONES PARA AVERAGING
# ============================================================

# Revisar que cada observacion aparezca una sola vez
# en los periodos de validacion.

anyDuplicated(cv_ols$pred$rowIndex)
anyDuplicated(pred_gam_cv$rowIndex)
anyDuplicated(cv_rf$pred$rowIndex)
anyDuplicated(cv_gbm$pred$rowIndex)


# ============================================================
# 80. UNIR PREDICCIONES DE LOS MODELOS
# ============================================================

pred_averaging <- cv_ols$pred %>%
  select(
    rowIndex,
    Resample,
    obs,
    pred_ols = pred
  ) %>%
  
  inner_join(
    pred_gam_cv %>%
      select(
        rowIndex,
        pred_gam = pred
      ),
    by = "rowIndex"
  ) %>%
  
  inner_join(
    cv_rf$pred %>%
      select(
        rowIndex,
        pred_rf = pred
      ),
    by = "rowIndex"
  ) %>%
  
  inner_join(
    cv_gbm$pred %>%
      select(
        rowIndex,
        pred_gbm = pred
      ),
    by = "rowIndex"
  )


# Revisar dimensiones

dim(pred_averaging)

head(pred_averaging)


# ============================================================
# 81. CORRELACION ENTRE ERRORES DE LOS MODELOS
# ============================================================

errores_modelos <- pred_averaging %>%
  transmute(
    OLS = obs - pred_ols,
    GAM = obs - pred_gam,
    RF = obs - pred_rf,
    Boosting = obs - pred_gbm
  )

cor(
  errores_modelos
)


# ============================================================
# 82. AVERAGING SIMPLE
# ============================================================

# Se asigna el mismo peso a los cuatro modelos.

pred_averaging <- pred_averaging %>%
  mutate(
    pred = (
      pred_ols +
        pred_gam +
        pred_rf +
        pred_gbm
    ) / 4
  )


# ============================================================
# 83. METRICAS DEL AVERAGING POR FOLD
# ============================================================

resultados_averaging_folds <- calcular_metricas_folds(
  pred_averaging %>%
    select(
      Resample,
      obs,
      pred
    )
)

resultados_averaging_folds


# ============================================================
# 84. METRICAS PROMEDIO DEL AVERAGING
# ============================================================

metricas_averaging <- data.frame(
  
  RMSE = mean(
    resultados_averaging_folds$RMSE
  ),
  
  MAE = mean(
    resultados_averaging_folds$MAE
  ),
  
  R2 = mean(
    resultados_averaging_folds$R2
  ),
  
  RMSESD = sd(
    resultados_averaging_folds$RMSE
  ),
  
  MAESD = sd(
    resultados_averaging_folds$MAE
  ),
  
  R2SD = sd(
    resultados_averaging_folds$R2
  )
)

metricas_averaging
# ============================================================
# 85. TABLA GENERAL DE VALIDACION TEMPORAL
# ============================================================

tabla_modelos <- data.frame(
  
  Modelo = c(
    "Baseline",
    "OLS",
    "Ridge",
    "Lasso",
    "Elastic Net",
    "GAM",
    "Arbol",
    "Random Forest",
    "Boosting",
    "Averaging"
  ),
  
  Familia = c(
    "Baseline",
    "Lineal",
    "Regularizacion",
    "Regularizacion",
    "Regularizacion",
    "Flexible",
    "Arbol",
    "Ensemble",
    "Ensemble",
    "Combinacion"
  ),
  
  RMSE = c(
    metricas_baseline$RMSE,
    metricas_ols$RMSE,
    metricas_ridge$RMSE,
    metricas_lasso$RMSE,
    metricas_enet$RMSE,
    metricas_gam$RMSE,
    metricas_arbol$RMSE,
    metricas_rf$RMSE,
    metricas_gbm$RMSE,
    metricas_averaging$RMSE
  ),
  
  MAE = c(
    metricas_baseline$MAE,
    metricas_ols$MAE,
    metricas_ridge$MAE,
    metricas_lasso$MAE,
    metricas_enet$MAE,
    metricas_gam$MAE,
    metricas_arbol$MAE,
    metricas_rf$MAE,
    metricas_gbm$MAE,
    metricas_averaging$MAE
  ),
  
  R2 = c(
    metricas_baseline$R2,
    metricas_ols$R2,
    metricas_ridge$R2,
    metricas_lasso$R2,
    metricas_enet$R2,
    metricas_gam$R2,
    metricas_arbol$R2,
    metricas_rf$R2,
    metricas_gbm$R2,
    metricas_averaging$R2
  ),
  
  RMSESD = c(
    metricas_baseline$RMSESD,
    metricas_ols$RMSESD,
    metricas_ridge$RMSESD,
    metricas_lasso$RMSESD,
    metricas_enet$RMSESD,
    metricas_gam$RMSESD,
    metricas_arbol$RMSESD,
    metricas_rf$RMSESD,
    metricas_gbm$RMSESD,
    metricas_averaging$RMSESD
  )
)


# ============================================================
# 86. MEJORA RESPECTO AL BASELINE
# ============================================================

tabla_modelos$Mejora_vs_Baseline <- (
  metricas_baseline$RMSE -
    tabla_modelos$RMSE
) / metricas_baseline$RMSE * 100


# ============================================================
# 87. ORDENAR MODELOS POR RMSE
# ============================================================

tabla_modelos <- tabla_modelos %>%
  arrange(RMSE)

tabla_modelos
# ============================================================
# 88. ORDENAR MODELOS POR RMSE REDDONDEADO
# ============================================================
tabla_modelos_redondeada <- tabla_modelos

tabla_modelos_redondeada[
  c(
    "RMSE",
    "MAE",
    "R2",
    "RMSESD",
    "Mejora_vs_Baseline"
  )
] <- round(
  tabla_modelos_redondeada[
    c(
      "RMSE",
      "MAE",
      "R2",
      "RMSESD",
      "Mejora_vs_Baseline"
    )
  ],
  3
)

tabla_modelos_redondeada
# ============================================================
# 89. PREDICCIONES TEMPORALES DEL BASELINE
# ============================================================

# Se guardan las predicciones del baseline para las mismas
# observaciones utilizadas en la validacion temporal.

pred_baseline_cv <- data.frame()

for(i in 1:length(folds_temporales$train)) {
  
  idx_train <- folds_temporales$train[[i]]
  idx_valid <- folds_temporales$test[[i]]
  
  media_fold <- mean(
    train$arr_delay[idx_train]
  )
  
  pred_baseline_cv <- rbind(
    pred_baseline_cv,
    
    data.frame(
      rowIndex = idx_valid,
      Resample = paste0("Fold", i),
      obs = train$arr_delay[idx_valid],
      pred = media_fold
    )
  )
}


# ============================================================
# 90. FUNCION PARA RMSE DEL SUBGRUPO
# ============================================================

# Se analiza el desempeño en vuelos que ya presentan
# al menos 15 minutos de retraso en la salida.

calcular_rmse_subgrupo <- function(
    predicciones,
    nombre_modelo
) {
  
  grupo <- train$dep_delay[
    predicciones$rowIndex
  ] >= 15
  
  data.frame(
    
    Modelo = nombre_modelo,
    
    RMSE_Subgrupo = sqrt(
      mean(
        (
          predicciones$obs[grupo] -
            predicciones$pred[grupo]
        )^2
      )
    ),
    
    N_Subgrupo = sum(grupo)
  )
}


# ============================================================
# 91. RMSE DEL SUBGRUPO PARA TODOS LOS MODELOS
# ============================================================

sub_baseline <- calcular_rmse_subgrupo(
  pred_baseline_cv,
  "Baseline"
)

sub_ols <- calcular_rmse_subgrupo(
  cv_ols$pred,
  "OLS"
)

sub_ridge <- calcular_rmse_subgrupo(
  cv_ridge$pred,
  "Ridge"
)

sub_lasso <- calcular_rmse_subgrupo(
  cv_lasso$pred,
  "Lasso"
)

sub_enet <- calcular_rmse_subgrupo(
  cv_enet$pred,
  "Elastic Net"
)

sub_gam <- calcular_rmse_subgrupo(
  pred_gam_cv,
  "GAM"
)

sub_arbol <- calcular_rmse_subgrupo(
  cv_arbol$pred,
  "Arbol"
)

sub_rf <- calcular_rmse_subgrupo(
  cv_rf$pred,
  "Random Forest"
)

sub_gbm <- calcular_rmse_subgrupo(
  cv_gbm$pred,
  "Boosting"
)

sub_averaging <- calcular_rmse_subgrupo(
  pred_averaging,
  "Averaging"
)


# ============================================================
# 92. TABLA DE RESULTADOS DEL SUBGRUPO
# ============================================================

tabla_subgrupo <- bind_rows(
  sub_baseline,
  sub_ols,
  sub_ridge,
  sub_lasso,
  sub_enet,
  sub_gam,
  sub_arbol,
  sub_rf,
  sub_gbm,
  sub_averaging
)

tabla_subgrupo <- tabla_subgrupo %>%
  arrange(RMSE_Subgrupo)

tabla_subgrupo


# ============================================================
# 93. AGREGAR SUBGRUPO A TABLA GENERAL
# ============================================================

tabla_modelos <- tabla_modelos %>%
  left_join(
    tabla_subgrupo %>%
      select(
        Modelo,
        RMSE_Subgrupo
      ),
    by = "Modelo"
  )


# ============================================================
# 94. TABLA COMPLETA REDONDEADA
# ============================================================

tabla_modelos_redondeada <- tabla_modelos

columnas_numericas <- c(
  "RMSE",
  "MAE",
  "R2",
  "RMSESD",
  "Mejora_vs_Baseline",
  "RMSE_Subgrupo"
)

tabla_modelos_redondeada[
  columnas_numericas
] <- round(
  tabla_modelos_redondeada[
    columnas_numericas
  ],
  3
)

tabla_modelos_redondeada
# ============================================================
# 95. MODELO FINAL
# ============================================================

# El GAM se selecciona como modelo final debido a que obtuvo:
# - menor RMSE de validacion temporal
# - menor RMSE en el subgrupo dep_delay >= 15
# - baja variabilidad entre periodos

modelo_final <- modelo_gam


# ============================================================
# 96. PREDICCIONES FINALES EN TEST
# ============================================================

# El conjunto test corresponde a las fechas mas recientes
# y no fue utilizado para seleccionar ni ajustar hiperparametros.

pred_test_gam <- predict(
  modelo_final,
  newdata = test,
  type = "response"
)


# ============================================================
# 97. METRICAS FINALES DEL GAM EN TEST
# ============================================================

rmse_test_gam <- sqrt(
  mean(
    (test$arr_delay - pred_test_gam)^2
  )
)

mae_test_gam <- mean(
  abs(
    test$arr_delay - pred_test_gam
  )
)

r2_test_gam <- 1 -
  sum(
    (test$arr_delay - pred_test_gam)^2
  ) /
  sum(
    (test$arr_delay - mean(test$arr_delay))^2
  )

rmse_test_gam
mae_test_gam
r2_test_gam


# ============================================================
# 98. BASELINE FINAL EN TEST
# ============================================================

# El baseline utiliza solamente la media del conjunto train.

media_train_final <- mean(
  train$arr_delay
)

pred_test_baseline <- rep(
  media_train_final,
  nrow(test)
)


rmse_test_baseline <- sqrt(
  mean(
    (test$arr_delay - pred_test_baseline)^2
  )
)

mae_test_baseline <- mean(
  abs(
    test$arr_delay - pred_test_baseline
  )
)

r2_test_baseline <- 1 -
  sum(
    (test$arr_delay - pred_test_baseline)^2
  ) /
  sum(
    (test$arr_delay - mean(test$arr_delay))^2
  )


# ============================================================
# 99. TABLA FINAL DE TEST
# ============================================================

tabla_test <- data.frame(
  
  Modelo = c(
    "Baseline",
    "GAM"
  ),
  
  RMSE = c(
    rmse_test_baseline,
    rmse_test_gam
  ),
  
  MAE = c(
    mae_test_baseline,
    mae_test_gam
  ),
  
  R2 = c(
    r2_test_baseline,
    r2_test_gam
  )
)

tabla_test


# ============================================================
# 100. MEJORA FINAL RESPECTO AL BASELINE
# ============================================================

mejora_test <- (
  rmse_test_baseline -
    rmse_test_gam
) / rmse_test_baseline * 100

mejora_test


# ============================================================
# 101. INTERVALOS DE PREDICCION EN TEST
# ============================================================

pred_test_intervalo <- predict(
  modelo_final,
  newdata = test,
  type = "response",
  se.fit = TRUE
)

varianza_residual_gam <- summary(
  modelo_final
)$scale

error_pred_test <- sqrt(
  pred_test_intervalo$se.fit^2 +
    varianza_residual_gam
)

intervalos_test <- data.frame(
  
  Observado = test$arr_delay,
  
  Prediccion = pred_test_intervalo$fit,
  
  Limite_Inferior =
    pred_test_intervalo$fit -
    1.96 * error_pred_test,
  
  Limite_Superior =
    pred_test_intervalo$fit +
    1.96 * error_pred_test
)

head(intervalos_test)


# ============================================================
# 102. COBERTURA DE LOS INTERVALOS EN TEST
# ============================================================

dentro_intervalo_test <- (
  intervalos_test$Observado >=
    intervalos_test$Limite_Inferior
) & (
  intervalos_test$Observado <=
    intervalos_test$Limite_Superior
)

cobertura_test <- mean(
  dentro_intervalo_test
)

cobertura_test


# ============================================================
# 103. AMPLITUD PROMEDIO DE LOS INTERVALOS EN TEST
# ============================================================

amplitud_test <- mean(
  intervalos_test$Limite_Superior -
    intervalos_test$Limite_Inferior
)

amplitud_test

# ============================================================
# ANEXO DE AUDITORIA REPRODUCIBLE Y FIGURAS (NO AJUSTA TEST)
# ============================================================
# El GAM final se selecciono con validacion temporal. Estas salidas
# calculan diagnosticos; no se usan para reentrenar ni elegir modelos.

# Directorio distinto por protocolo, para evitar sobrescribir exportaciones.
salidas_protocolo <- if (FOLDS_POR_FECHA) 'salidas_informe_fechas' else 'salidas_informe_filas'
dir.create(salidas_protocolo, showWarnings = FALSE, recursive = TRUE)
stopifnot(max(train$fl_date) < min(test$fl_date),
          nrow(train) == 7855L, nrow(test) == 1981L,
          length(folds_temporales$train) == 5L)
for (j in seq_along(folds_temporales$train)) {
  tt <- folds_temporales$train[[j]]
  vv <- folds_temporales$test[[j]]
  stopifnot(length(intersect(tt,vv)) == 0L,
            max(tt) < min(vv))
  if (FOLDS_POR_FECHA) {
    stopifnot(max(train$fl_date[tt]) < min(train$fl_date[vv]))
  } else {
    stopifnot(max(train$fl_date[tt]) <= min(train$fl_date[vv]))
  }
}

# Baseline operativo (sin ajuste de parametros), mismo esquema de folds
pred_regla_cv <- do.call(rbind, lapply(seq_along(folds_temporales$test), function(j) {
  ix <- folds_temporales$test[[j]]
  data.frame(rowIndex=ix, Resample=paste0('Fold',j),
             obs=train$arr_delay[ix], pred=train$dep_delay[ix])
}))
resultados_regla_folds <- calcular_metricas_folds(pred_regla_cv)
resumen_regla <- data.frame(
  Modelo='Regla dep_delay', Familia='Regla operacional',
  RMSE=mean(resultados_regla_folds$RMSE),
  MAE=mean(resultados_regla_folds$MAE),
  R2=mean(resultados_regla_folds$R2),
  RMSESD=sd(resultados_regla_folds$RMSE),
  Mejora_vs_Baseline=NA_real_, RMSE_Subgrupo=NA_real_
)
resumen_regla$RMSE_Subgrupo <- sqrt(mean((
  pred_regla_cv$obs[train$dep_delay[pred_regla_cv$rowIndex]>=15] -
  pred_regla_cv$pred[train$dep_delay[pred_regla_cv$rowIndex]>=15]
)^2))

# Recalcular la tabla completa a partir de los objetos de ESTA ejecucion:
# no conservar numeros de versiones previas del script.
tabla_completa <- dplyr::bind_rows(tabla_modelos, resumen_regla)
tabla_completa$Mejora_vs_regla_pct <- 100*(resumen_regla$RMSE - tabla_completa$RMSE)/resumen_regla$RMSE
tabla_completa$Mejora_vs_media_pct <- 100*(metricas_baseline$RMSE - tabla_completa$RMSE)/metricas_baseline$RMSE
tabla_completa <- tabla_completa[order(tabla_completa$RMSE),]
write.csv(tabla_completa, file.path(salidas_protocolo, 'tabla_validacion_completa.csv'), row.names=FALSE)

resultados_por_fold <- list(
  Baseline=resultados_baseline_folds, Regla=resultados_regla_folds,
  OLS=resultados_ols_folds, Ridge=resultados_ridge_folds,
  Lasso=resultados_lasso_folds, Elastic_Net=resultados_enet_folds,
  GAM=resultados_gam_folds, Arbol=resultados_arbol_folds,
  Random_Forest=resultados_rf_folds, Boosting=resultados_gbm_folds,
  Averaging=resultados_averaging_folds
)
for (nombre in names(resultados_por_fold)) {
  write.csv(resultados_por_fold[[nombre]],
            file.path(salidas_protocolo, paste0('folds_', nombre,'.csv')),
            row.names=FALSE)
}

correlaciones <- cor(errores_modelos, use='complete.obs')
write.csv(correlaciones, file.path(salidas_protocolo, 'correlacion_errores_averaging.csv'))
hiperparametros <- data.frame(
  Modelo=c('Ridge','Lasso','Elastic Net','Arbol','GAM','Random Forest','Boosting'),
  alpha=c(0,1,cv_enet$bestTune$alpha,NA,NA,NA,NA),
  lambda=c(cv_ridge$bestTune$lambda,cv_lasso$bestTune$lambda,
           cv_enet$bestTune$lambda,NA,NA,NA,NA),
  cp=c(NA,NA,NA,cv_arbol$bestTune$cp,NA,NA,NA),
  k=c(NA,NA,NA,NA,mejor_k,NA,NA),
  mtry=c(NA,NA,NA,NA,NA,cv_rf$bestTune$mtry,NA),
  arboles=c(NA,NA,NA,NA,NA,500,cv_gbm$bestTune$n.trees),
  profundidad=c(NA,NA,NA,NA,NA,NA,cv_gbm$bestTune$interaction.depth),
  tasa=c(NA,NA,NA,NA,NA,NA,cv_gbm$bestTune$shrinkage),
  minimo_nodo=c(NA,NA,NA,NA,NA,NA,cv_gbm$bestTune$n.minobsinnode)
)
write.csv(hiperparametros,file.path(salidas_protocolo, 'hiperparametros.csv'),row.names=FALSE)
write.csv(tabla_subgrupo,file.path(salidas_protocolo, 'subgrupos_validacion.csv'),row.names=FALSE)

# Test unico del GAM + baseline medio + regla
regla_test <- test$dep_delay
idx_critico_test <- test$dep_delay >= 15
stopifnot(sum(idx_critico_test) == 355L)
cobertura_vuelos <- with(intervalos_test,
  Observado >= Limite_Inferior & Observado <= Limite_Superior)
metricas_test <- data.frame(
  Modelo=c('Baseline media','Regla dep_delay','GAM'),
  N=nrow(test),
  RMSE=c(rmse_test_baseline,
         sqrt(mean((test$arr_delay-regla_test)^2)),rmse_test_gam),
  MAE=c(mae_test_baseline,mean(abs(test$arr_delay-regla_test)),mae_test_gam),
  R2=c(r2_test_baseline,
       1-sum((test$arr_delay-regla_test)^2)/
           sum((test$arr_delay-mean(test$arr_delay))^2),r2_test_gam)
)
write.csv(metricas_test,file.path(salidas_protocolo, 'metricas_test.csv'),row.names=FALSE)
metricas_sub_test <- data.frame(
  Modelo=c('Regla dep_delay','GAM'),N=sum(idx_critico_test),
  RMSE=c(sqrt(mean((test$arr_delay[idx_critico_test]-regla_test[idx_critico_test])^2)),
         sqrt(mean((test$arr_delay[idx_critico_test]-pred_test_gam[idx_critico_test])^2))),
  MAE=c(mean(abs(test$arr_delay[idx_critico_test]-regla_test[idx_critico_test])),
        mean(abs(test$arr_delay[idx_critico_test]-pred_test_gam[idx_critico_test]))),
  Cobertura95=c(NA,mean(cobertura_vuelos[idx_critico_test]))
)
write.csv(metricas_sub_test,file.path(salidas_protocolo, 'subgrupo_test.csv'),row.names=FALSE)
write.csv(data.frame(N=nrow(test),Cobertura95=mean(cobertura_vuelos),
                     Amplitud_promedio=amplitud_test),
          file.path(salidas_protocolo, 'intervalos_test.csv'),row.names=FALSE)
write.csv(data.frame(
  Indice=seq_len(nrow(test)), fl_date=test$fl_date,
  obs=test$arr_delay, dep_delay=test$dep_delay, gam=pred_test_gam,
  inferior=intervalos_test$Limite_Inferior,
  superior=intervalos_test$Limite_Superior,
  subgrupo=idx_critico_test, cubierto=cobertura_vuelos),
  file.path(salidas_protocolo, 'predicciones_test.csv'),row.names=FALSE)

# Figuras fieles del GAM final: curvas centradas de efectos suaves, con IC
# No se grafica en todo rango si hay outliers extremos poco frecuentes.
crear_figura_efecto <- function(variable, etiqueta, archivo, desde, hasta) {
  x <- seq(desde,hasta,length.out=180)
  nuevo <- train[rep(1L,length(x)),,drop=FALSE]
  nuevo[[variable]] <- x
  p <- predict(modelo_gam,newdata=nuevo,type='terms',se.fit=TRUE)
  columna <- paste0('s(',variable,')')
  if (!columna %in% colnames(p$fit)) stop('Termino ausente: ', columna)
  est <- p$fit[,columna]; se <- p$se.fit[,columna]
  png(file.path(salidas_protocolo,archivo),width=1500,height=850,res=210)
  par(mar=c(4,4.6,1.3,0.8),family='sans')
  plot(x,est,type='n',xlab=etiqueta,ylab='Efecto parcial centrado (min)',
       ylim=range(est-1.96*se,est+1.96*se),las=1)
  polygon(c(x,rev(x)),c(est-1.96*se,rev(est+1.96*se)),
          border=NA,col=rgb(.45,.60,.72,.32))
  lines(x,est,lwd=2.4,col='#1B4F72')
  abline(h=0,lty=3,col='grey55')
  dev.off()
}
crear_figura_efecto('dep_delay','Retraso de salida (min)',
                    'gam_efecto_dep_delay.png',-10,150)
crear_figura_efecto('taxi_out','Rodaje de salida (min)',
                    'gam_efecto_taxi_out.png',0,80)

# Importancia por permutacion de RF (si RF se ajusto con importance=TRUE)
imp <- randomForest::importance(modelo_rf)
if ('%IncMSE' %in% colnames(imp)) {
  imp <- sort(imp[,'%IncMSE'],decreasing=TRUE)
  write.csv(data.frame(Variable=names(imp),IncMSE=as.numeric(imp)),
            file.path(salidas_protocolo, 'importancia_RF.csv'),row.names=FALSE)
  top <- head(imp,8)
  png(file.path(salidas_protocolo, 'importancia_RF.png'),width=1600,height=950,res=205)
  par(mar=c(4.4,9,1.2,1),family='sans')
  barplot(rev(top),horiz=TRUE,las=1,col='#4D7890',border=NA,
          xlab='Incremento porcentual del MSE al permutar (%)')
  dev.off()
}

# ============================================================
# TRAZABILIDAD DEL PROTOCOLO (FECHAS O FILAS)
# ============================================================
protocolo <- do.call(rbind, lapply(seq_along(folds_temporales$train), function(j) {
  tt <- folds_temporales$train[[j]]
  vv <- folds_temporales$test[[j]]
  a <- train[tt,,drop=FALSE]; b <- train[vv,,drop=FALSE]
  extreme <- which.max(b$arr_delay)
  data.frame(Fold=j, n_train=nrow(a), n_validacion=nrow(b),
             ultima_fecha_train=as.character(max(a$fl_date)),
             primera_fecha_validacion=as.character(min(b$fl_date)),
             ultima_fecha_validacion=as.character(max(b$fl_date)),
             max_dep_delay_train=max(a$dep_delay),
             max_dep_delay_validacion=max(b$dep_delay),
             max_arr_delay_validacion=max(b$arr_delay),
             fecha_max_arr_delay=as.character(b$fl_date[extreme]),
             max_arr_delay_dep_salida=b$dep_delay[extreme],
             n_subgrupo=sum(b$dep_delay>=15))
}))
write.csv(protocolo,file.path(salidas_protocolo, 'protocolo_folds.csv'),row.names=FALSE)
write.csv(resultados_k_gam, file.path(salidas_protocolo, 'sensibilidad_k_GAM.csv'),row.names=FALSE)

# La comparación GAM-Lasso debe ser PAREADA por la misma fecha/observación;
# la DE de RMSE por separado NO mide incertidumbre de su diferencia.
stopifnot(anyDuplicated(pred_gam_cv$rowIndex) == 0L,
          anyDuplicated(cv_lasso$pred$rowIndex) == 0L)
comp_pares <- merge(
  pred_gam_cv[,c('rowIndex','Resample','obs','pred')],
  cv_lasso$pred[,c('rowIndex','pred')],
  by='rowIndex', suffixes=c('_GAM','_Lasso'))
stopifnot(nrow(comp_pares)==nrow(pred_gam_cv),
          all.equal(comp_pares$obs,
                    train$arr_delay[comp_pares$rowIndex])==TRUE)
comparacion_pareada <- do.call(rbind, lapply(seq_along(folds_temporales$test), function(j) {
  ix <- folds_temporales$test[[j]]
  p <- comp_pares[comp_pares$rowIndex %in% ix,,drop=FALSE]
  rg <- sqrt(mean((p$obs-p$pred_GAM)^2))
  rl <- sqrt(mean((p$obs-p$pred_Lasso)^2))
  data.frame(Fold=j,N=nrow(p),RMSE_GAM=rg, RMSE_Lasso=rl,
             Diferencia_Lasso_menos_GAM=rl-rg)
}))
write.csv(comparacion_pareada,
          file.path(salidas_protocolo, 'comparacion_pareada_GAM_Lasso.csv'),row.names=FALSE)
delta <- comparacion_pareada$Diferencia_Lasso_menos_GAM
resumen_pareado <- data.frame(
  Diferencia_media=mean(delta), DE_diferencias=sd(delta),
  IC95_inferior=mean(delta)-qt(.975,length(delta)-1)*sd(delta)/sqrt(length(delta)),
  IC95_superior=mean(delta)+qt(.975,length(delta)-1)*sd(delta)/sqrt(length(delta)),
  folds=length(delta))
write.csv(resumen_pareado,
          file.path(salidas_protocolo, 'resumen_pareado_GAM_Lasso.csv'),row.names=FALSE)

# Gráfico del RMSE para el MISMO protocolo temporal, por fold.
extraer_rmse <- function(obj,nombre) {
  data.frame(Fold=seq_len(nrow(obj)), RMSE=obj$RMSE, Modelo=nombre)
}
plot_folds <- do.call(rbind,list(
  extraer_rmse(resultados_gam_folds,'GAM'),
  extraer_rmse(resultados_lasso_folds,'Lasso'),
  extraer_rmse(resultados_ols_folds,'OLS'),
  extraer_rmse(resultados_rf_folds,'Random Forest'),
  extraer_rmse(resultados_gbm_folds,'Boosting'),
  extraer_rmse(resultados_regla_folds,'Regla salida')))
write.csv(plot_folds,file.path(salidas_protocolo, 'rmse_por_fold.csv'),row.names=FALSE)
png(file.path(salidas_protocolo, 'rmse_por_fold.png'),width=1800,height=1050,res=200)
print(ggplot(plot_folds,aes(x=Fold,y=RMSE,color=Modelo,group=Modelo))+
        geom_line(linewidth=.85)+geom_point(size=2)+
        scale_x_continuous(breaks=1:5)+
        labs(x='Fold cronológico',y='RMSE (minutos)',
             title=if (FOLDS_POR_FECHA) 'Validación por fechas completas' else 'Validación por filas cronológicas')+
        theme_minimal(base_size=13)+
        theme(legend.position='bottom'))
dev.off()

# Los efectos suaves no sustituyen al PDP: se exporta un PDP con una
# muestra de referencia de TRAIN y una banda basada en covarianza GAM.
# ICE describe la respuesta de 18 vuelos representativos al variar X.
set.seed(123)
submuestra_pdp <- train[sample.int(nrow(train),min(500L,nrow(train))),,drop=FALSE]
submuestra_ice <- submuestra_pdp[seq_len(min(18L,nrow(submuestra_pdp))),,drop=FALSE]
exportar_pdp <- function(variable,nombre_archivo,desde,hasta) {
  grid <- seq(desde,hasta,length.out=50)
  pred_promedio <- numeric(length(grid)); se_promedio <- numeric(length(grid))
  ice <- do.call(rbind,lapply(seq_along(grid),function(j) {
    tmp <- submuestra_ice
    tmp[[variable]] <- grid[j]
    data.frame(x=grid[j], id=seq_len(nrow(tmp)),
               y=as.numeric(predict(modelo_final,newdata=tmp,type='response')))
  }))
  for (j in seq_along(grid)) {
    tmp <- submuestra_pdp
    tmp[[variable]] <- grid[j]
    X <- predict(modelo_final,newdata=tmp,type='lpmatrix')
    Xbar <- colMeans(X)
    pred_promedio[j] <- sum(Xbar*coef(modelo_final))
    se_promedio[j] <- sqrt(as.numeric(crossprod(Xbar,modelo_final$Vp%*%Xbar)))
  }
  curva <- data.frame(x=grid, valor=pred_promedio,
                      inf=pred_promedio-1.96*se_promedio,
                      sup=pred_promedio+1.96*se_promedio)
  write.csv(curva,file.path(salidas_protocolo,sub('png$','csv',nombre_archivo)),row.names=FALSE)
  png(file.path(salidas_protocolo,nombre_archivo),width=1550,height=900,res=200)
  print(ggplot()+
          geom_line(data=ice,aes(x=x,y=y,group=id),color='grey65',alpha=.26,linewidth=.35)+
          geom_ribbon(data=curva,aes(x=x,ymin=inf,ymax=sup),fill='#5D8DA7',alpha=.30)+
          geom_line(data=curva,aes(x=x,y=valor),color='#174766',linewidth=1)+
          labs(x=paste(variable,'(min)'),y='Retraso de llegada estimado (min)',
               title=paste('PDP e ICE:',variable))+
          theme_minimal(base_size=14))
  dev.off()
}
exportar_pdp('dep_delay','gam_PDP_ICE_dep_delay.png',-10,150)
exportar_pdp('taxi_out','gam_PDP_ICE_taxi_out.png',0,80)

# Información de procedencia de cada ejecución, sin números heredados.
write.csv(data.frame(
  fuente_CSV=basename(ruta_csv), md5=unname(tools::md5sum(ruta_csv)),
  fecha_corte=as.character(fecha_corte),
  n_train=nrow(train), n_test=nrow(test),
  num_folds=length(folds_temporales$train),
  semilla=123, FOLDS_POR_FECHA=FOLDS_POR_FECHA,
  fecha_generacion=as.character(Sys.Date())),
  file.path(salidas_protocolo, 'manifest_ejecucion.csv'),row.names=FALSE)

writeLines(capture.output(sessionInfo()),file.path(salidas_protocolo, 'sessionInfo.txt'))
message('Resultados reproducibles escritos en: ',normalizePath(salidas_protocolo))
