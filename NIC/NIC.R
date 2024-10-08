# Nicaragua --------------------------------------------------------------
# Autor: Marco Espinoza
# Comisión Económica para América Latina y el Caribe

# Datos -------------------------------------------------------------------

# Librerías

x <- c(
  'lme4',
  'stan4bart',
  'dplyr',
  'XboostingMM',
  'srvyr',
  'survey',
  'data.table',
  'ggplot2',
  'tidyr',
  'sf',
  'httr2',
  'ows4R',
  'rmapshaper',
  'janitor',
  'xgboost'
)

lapply(x, require, character.only = TRUE)

rm(list = ls())

encuesta_mrp <- readRDS("NIC/2014/encuesta_mrp.rds")
statelevel_predictors_df <- readRDS("NIC/2014/statelevel_predictors_df.rds")


byAgrega <-
  grep(
    pattern =  "^(n|pobreza|ingreso|lp|li)",
    x = names(encuesta_mrp),
    invert = TRUE,
    value = TRUE
  )

encuesta_df_agg <-
  encuesta_mrp %>%
  group_by_at(all_of(byAgrega)) %>%
  summarise(n = n(),
            ingreso = mean(ingreso),
            .groups = "drop") 


encuesta_df_agg <- full_join(encuesta_df_agg, statelevel_predictors_df,
                             by = "dam")


data <- encuesta_df_agg |>
  mutate_if(is.character, as.factor)

censo <- readRDS("NIC/2014/censo_mrp.rds") 

# Importante: ambos deben ser class() data.frame, por lo que es mejor convertirlos
data <- as.data.frame(data)
censo <- as.data.frame(censo)

dfsTrain <- data
dfsTest1 <- censo

# Código ------------------------------------------------------------------

set.seed(260224)

#- Parámetros
conv_memboost = 0.001
maxIter_memboost = 1000
verbose_memboost = F
minIter_memboost = 0

# Creamos una variable conjunta que será tratada como variable aleatoria
var_ale <- paste0(data$dam, data$area, data$sexodata$etnia,data$anoest, data$edad)
# Revisamos las combinaciones
length(unique(var_ale)) # Contamos con 504 combinaciones de efectos Aleatorios.

dfsTrain$var_ale <- var_ale

#- specify model
formula <- ingreso ~ luces_nocturnas + cubrimiento_rural + cubrimiento_urbano + modificacion_humana + accesibilidad_hospitales + accesibilidad_hosp_caminado + area1 + sexo2 + edad2 + edad3 + edad4 + edad5 + anoest2 + anoest3 + anoest4 + etnia1 + tiene_alcantarillado + tiene_electricidad + tiene_acueducto + tiene_gas + eliminar_basura + tiene_internet + piso_tierra + material_paredes + rezago_escolar + alfabeta + tasa_desocupacion
random <- ~ 1 | var_ale

# Preparaciones ============================================================

sum(is.na(dfsTrain))


result <- NULL
PredNames <- attr(stats::terms(formula), "term.labels")
OutcomeName <-  all.vars(formula)[1]
XTrain <- as.data.frame(dfsTrain[, PredNames, drop = FALSE])
# XTest1 <- as.data.frame(dfsTest1[, PredNames, drop = FALSE])
YTrain <- as.matrix(dfsTrain[, OutcomeName, drop = FALSE])

# Modelo ===================================================================

# Función de optimización
sqrterror <- function(preds, dtrain) {
  labels <- xgboost::getinfo(dtrain, "label")
  grad <- preds - labels
  hess <- rep(1, length(grad))
  return(list(grad = grad, hess = hess))
}


# Validación cruzada ------------------------------------------------------

# Aquí se determinan los parámetros para el modelo final
ajuste = dfsTrain[,-c(1,2,3,4,5,6,7,8,37)]

sparse_matrix <- sparse.model.matrix(ingreso ~ ., data = ajuste)[, -1]

y = encuesta_df_agg$ingreso

grid <- expand_grid(
  eta = seq(0.30, 1, 0.10),
  max_depth = seq(6, 10, 1),
  # min_child_weight = seq(1, 20, 5),
  # subsample = seq(0.5, 1, 0.5),
  lambda = seq(1, 10, 1),
  alpha = seq(0, 10, 1)
)

xgb_train_rmse <- numeric(nrow(grid))
xgb_test_rmse <- numeric(nrow(grid))


for(i in 1:nrow(grid)){
  xgb_untuned = xgb.cv(
    data = sparse_matrix,
    label = y,
    booster = "gbtree",
    eta = grid$eta[i],
    max_depth = grid$max_depth[i],
    # min_child_weight = grid$min_child_weight[i],
    # subsample = grid$subsample[i],
    lambda = grid$lambda[i],
    alpha = grid$alpha[i],
    objective = sqrterror,
    # eval_metric = "rmse",
    nrounds = 1000,
    early_stopping_rounds = 3, # training with a validation set will stop if the performance does not improve for k rounds (3)
    nfold = 5
  )
  
  xgb_train_rmse[i] <-
    xgb_untuned$evaluation_log$train_rmse_mean[xgb_untuned$best_iteration]
  xgb_test_rmse[i] <-
    xgb_untuned$evaluation_log$test_rmse_mean[xgb_untuned$best_iteration]
  
  cat(i, "\n")
}

xgb_train_rmse <- xgb_train_rmse |> 
  tibble() |> 
  mutate(simulacion = seq(1:4400))

xgb_test_rmse <- xgb_test_rmse |> 
  tibble() |> 
  mutate(simulacion = seq(1:4400))

#Guardamos las iteraciones para el primer modelo de la validación cruzada

saveRDS(xgb_train_rmse, file ="ingreso/output/xgb_train_rmse_all_tree.rds")
saveRDS(xgb_test_rmse, file = "ingreso/output/xgb_test_rmse_all_tree.rds")

xgb_train_rmse |> ggplot(aes(x = simulacion, y = xgb_train_rmse)) + 
  geom_line() 

# Mejores parámetros
grid[which.min(xgb_train_rmse$xgb_train_rmse), ] 

# =========================================================================

# Modelo
fitBoostMERT_L2 <- boost_mem(
  formula,
  data = dfsTrain,
  random = random,
  shrinkage = 1,
  interaction.depth = 6,
  n.trees = 100,
  loss = sqrterror,
  minsplit = 1,
  subsample = 1,
  lambda = 5,
  alpha = 10,
  verbose_memboost = verbose_memboost,
  minIter_memboost = minIter_memboost,
  maxIter_memboost = maxIter_memboost
)


# Predicción
fhat_Test1 <- XboostingMM:::predict.xgb(fitBoostMERT_L2$boosting_ensemble,
                                        newdata = dfsTest1,
                                        n.trees = 100, allow.new.levels = TRUE)

# Guardamos los resultados
# saveRDS(fitBoostMERT_L2, "output/fit.rds")
# saveRDS(fhat_Test1, "output/prediction.rds")

# Bayesian Additive Regression Tree with random intercept -----------------

fitBART <- stan4bart::stan4bart(
  formula = ingreso ~ (1 | var_ale) + bart(
    F182013_stable_lights + X2016_crops.coverfraction + X2016_urban.coverfraction + X2016_gHM + accessibility + accessibility_walking_only + area1 + sexo2 + edad2 + edad3 + edad4 + edad5 + anoest2 + anoest3 + anoest4 + discapacidad1 + etnia1 + tiene_alcantarillado + tiene_electricidad + tiene_acueducto + tiene_gas + eliminar_basura + tiene_internet + piso_tierra + material_paredes + material_techo + rezago_escolar + alfabeta + hacinamiento + tasa_desocupacion
  ),
  verbose = -1,
  # suppress ALL output
  data = dfsTrain,
  bart_args = list(keepTrees = T)
) # needed for prediction of (new) test data


# Yhat_BART_Test1 <- rowMeans(predict(fitBART, dfsTest1))

# saveRDS(fitBART, "output/fitBART.rds")

# Lmer --------------------------------------------------------------------

modelLMM <- ingreso ~ (1 | var_ale) + F182013_stable_lights  + X2016_crops.coverfraction + X2016_urban.coverfraction + X2016_gHM + accessibility + accessibility_walking_only + area1 + sexo2 + edad2 + edad3 + edad4 + edad5 + anoest2 + anoest3 + anoest4 + tiene_alcantarillado + tiene_electricidad + tiene_acueducto + tiene_gas + eliminar_basura + tiene_internet + piso_tierra + material_paredes + material_techo + rezago_escolar + alfabeta + hacinamiento + tasa_desocupacion

fitLMM <- lme4::lmer(modelLMM, weights = n, data = dfsTrain)

# saveRDS(fitLMM, "output/fitLMM.rds")

# Varianza y el error -----------------------------------------------------

rm(list = ls())

fitBoostMERT_L2 <- readRDS("output/fit.rds")
fitBART <- readRDS("output/fitBART.rds")
fitLMM <- readRDS("output/fitLMM.rds")

# Varibility betweeen planification regions (between-group variance)
Dhat <- matrix(
  c(
    fitBoostMERT_L2$var_random_effects,
    fitted(fitBART, type = "Sigma")$var_ale,
    lme4::VarCorr(fitLMM)$var_ale[1]
  ),
  nrow = 3,
  dimnames = list(
    Model = c("BoostMERT_L2", "BART", "LMM"),
    "Variance of random intercepts"
  )
  
)

# Variance of the residual errors (not explained by either fixed effects or random effects)
ErrorVar <- matrix(
  c(
    fitBoostMERT_L2$errorVar,
    fitted(fitBART, type = "sigma") ^ 2,
    sigma(fitLMM) ^ 2
  ),
  nrow = 3,
  dimnames = list(
    Model = c("BoostMERT_L2", "BART", "LMM"),
    "Variance of residual errors"
  )
)

Dhat
ErrorVar
# Bootstrap ---------------------------------------------------------------

rm(list = ls())

data <- readRDS("data/encuesta_df_agg.rds") |>
  mutate_if(is.character, as.factor)

censo <- readRDS("data/cens0.rds") |>
  select(dam) 

data <- as.data.frame(data)
censo <- as.data.frame(censo)
# Leemos la predicción
f <- readRDS("output/prediction.rds")
length(f)

# pegamos la predicción al censo
censo$f <- f

# 2. Efectos aleatorios
fit <- readRDS("output/fit.rds")
randomEffects <- fit$raneffs

# 3. Errores
errorTerms <- fit$errorTerms

# Calculamos el population B1

dominio <- sort(unique(censo$dam))
len_dom <- length(censo$dam)

# Tamaño de la base
ext <- dim(censo)[1]

count <- 1
limit <- 100
PBS <- matrix(0, nrow = ext, ncol = limit)
result <-  vector(mode = "numeric", length = ext)

while (count <= limit) {
  # sample del error
  errorPB <- sample(errorTerms, ext, replace = TRUE)
  
  for (dom in dominio) {
    # Primero verificamos que en el ciclo identificamos el dominio en el censo
    # Del cual deberia devolver las 1437686 observaciones con los dominios
    # identificados.
    
    # Hay que obtener el vector F filtrado para cada dominio
    indices <- which(censo$dam == dom)
    
    random_effect <- randomEffects[as.numeric(dom)]
    
    result[indices] <- censo$f[indices] + random_effect + errorPB[indices]
  }
  
  PBS[, count] <- result
  count <- count + 1
  
}

# saveRDS(PBS, "output/PBS.rds")

colnames(PBS) <- paste0("PB", 1:limit)

PBS <- as.data.frame(PBS)

PBS$dam <- censo$dam


# Cálculo de medias para cada dominio -------------------------------------
PBS_long <- PBS |>
  pivot_longer(cols = starts_with("PB"),
               names_to = "PB",
               values_to = "value")

mean_df <- PBS_long |>
  group_by(dam, PB) |>
  summarise(media = mean(value, na.rm = TRUE)) |>
  pivot_wider(names_from = PB, values_from = media)

# saveRDS(PBS_long, "output/PBS_long.rds")
# saveRDS(mean_df, "output/mean_df.rds")

# Cálculo de medias para todas las PB -------------------------------------

PBS_long <- readRDS("output/PBS_long.rds")
mean_df <- readRDS("output/mean_df.rds")

medias <- numeric(6)
varianzas <- numeric(6)

for (i in 1:6) {
  medias[i] <- mean(as.numeric(mean_df[i, -1]), na.rm = TRUE)
  varianzas[i] <- var(as.numeric(mean_df[i, -1]), na.rm = TRUE)
}

resultado <- matrix(
  c(medias, varianzas),
  nrow = 6,
  ncol = 2,
  dimnames = list(
    "Dam" = c("01", "02", "03", "04", "05", "06"),
    "Estimación" = c("Media", "Varianza")
  )
)

resultado <- as_tibble(resultado)
resultado$dam <- c("01","02","03","04","05","06")


# MSE ---------------------------------------------------------------------

est_puntual <- fhat_Test1

for(columna in 1:length(PBS)){
  MSE <- sum( (est_puntual - PBS[,columna])**2 )
  
}

# Validación --------------------------------------------------------------


# Calcular los intervalos de confianza
IC_df <- mean_df |>
  pivot_longer(cols = starts_with("PB"),
               names_to = "PB",
               values_to = "Media") |>
  group_by(dam) |>
  summarise(
    lower = quantile(Media, probs = 0.025, na.rm = TRUE),
    upper = quantile(Media, probs = 0.975, na.rm = TRUE)
  )

# Mostrar los intervalos de confianza junto con las medias y varianzas
final <- resultado |>
  left_join(IC_df, by = "dam")

# saveRDS(final, "output/bootstrap_results.rds")

final <- readRDS("output/bootstrap_results.rds")

margin <- (final$upper - final$lower) * 2

final$AdjustedLower <- final$lower - margin
final$AdjustedUpper <- final$upper + margin


final |>
  ggplot(aes(x = dam, y = Media)) + geom_point(col = "green") +
  labs(x = "Región de planificación económica", y = "Ingreso") +
  # ylim(150000,450000) +
  geom_errorbar(data = final, aes(x = dam, ymin = AdjustedLower,
                                  ymax = AdjustedUpper)) +
  theme_minimal()

# Mapas -------------------------------------------------------------------
rm(list = ls())

cantones <- st_read("geojson/cantones_ajustado_cr.geojson", quiet = TRUE)

ingreso_cantonal <- readRDS("output/ingreso_cantonal.rds")

mapa <- left_join(x = cantones,
                  y = ingreso_cantonal,
                  by = join_by(canton == canton))

glimpse(mapa)

mapa_plot <- ggplot(data = mapa, mapping = aes(fill = ingreso_medio)) +
  geom_sf(color = "white") +
  labs(fill = "Ingreso medio") +
  scale_fill_viridis_c() +
  theme_minimal()

# ggsave(mapa_plot, filename = "ingreso/output/mapa_cantonal.png")


# Mapa regiones de planificación ------------------------------------------

regiones <- st_read("geojson/regiones_cr.geojson")
ingreso_region <- readRDS("output/bootstrap_results.rds") |>
  mutate(
    region = recode(dam,
                    "01" = "Central",
                    "02" = "Chorotega",
                    "03" = "Pacífico Central",
                    "04" = "Brunca",
                    "05" = "Huetar Caribe",
                    "06" = "Huetar Norte"))


mapping <- left_join(x = regiones,
                     y = ingreso_region,
                     by = join_by(region == region))

region_plot <- ggplot(data = mapping, mapping = aes(fill = Media)) +
  geom_sf(color = "white") +
  labs(fill = "Ingreso medio") +
  scale_fill_viridis_c() +
  theme_minimal()

ggsave(region_plot, filename = "ingreso/output/mapa_mideplan.png")


