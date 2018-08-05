# ============================ Title: McKinsey Analytics Online Hackathon ============================ #

# Setting working directory
# ================================
rm(list = ls())
filepath <- c("/Users/nkaveti/Documents/Kaggle/McKinsey Online Hackathon")
setwd(filepath)

# Loading required libraries
# ================================
library(data.table)
library(caret)
library(xgboost)
library(Metrics)

# Reading data
# ================================
train <- fread("train_ZoGVYWq.csv")
test <- fread("test_66516Ee.csv")

train[, renewal := as.integer(renewal)]

# Function to clean the data
# ================================
clean_data <- function(){
  y <- "renewal"
  fdata <- rbind(train, test, fill = TRUE)
  cnames <- colnames(fdata)
  cnames <- setdiff(cnames, c(y, "id"))
  
  # Type casting of variables
  for(i in cnames){
    if(length(unique(fdata[[i]])) < 5 | is.character(fdata[[i]])){
      if(!is.null(fdata[[i]])) fdata[[i]] <- as.factor(fdata[[i]])
    }
  }
  count_missing <- sapply(fdata, function(x){sum(is.na(x))})
  count_missing <- count_missing[count_missing > 0]
  count_missing <- names(count_missing)
  count_missing <- setdiff(count_missing, y)
  for(i in count_missing){
    fdata[[i]][is.na(fdata[[i]])] <- median(fdata[[i]], na.rm = TRUE)
  }
  return(fdata)
}

handle_cat_variables <- function(inData){
  y <- "renewal"
  cat_var <- names(sapply(inData, is.factor)[sapply(inData, is.factor)])
  cat_var <- setdiff(cat_var, y)
  form <- as.formula(paste0("~ -1 + ", paste(cat_var, collapse = " + ")))
  inData <- cbind(inData, as.data.table(model.matrix(form, inData)))
  tr <- inData[!is.na(inData[[y]])]
  te <- inData[is.na(inData[[y]])]
  
  tr[, fold := createFolds(y = tr[[y]], k = 5, list = FALSE)]
  res <- data.table()
  for(i in 1:5){
    temp_tr <- tr[fold != i]
    temp_te <- tr[fold == i]
    temp <- temp_tr[, .(mean_y_sourcing_channel = mean(renewal)), by = sourcing_channel]
    temp_te <- merge(temp_te, temp, by = "sourcing_channel", all.x = TRUE)
    temp <- temp_tr[, .(mean_y_residence_area_type = mean(renewal)), by = residence_area_type]
    temp_te <- merge(temp_te, temp, by = "residence_area_type", all.x = TRUE)
    res <- rbind(res, temp_te)
  }
  temp <- tr[, .(mean_y_sourcing_channel = mean(renewal)), by = sourcing_channel]
  te <- merge(te, temp, by = "sourcing_channel", all.x = TRUE)
  temp <- tr[, .(mean_y_residence_area_type = mean(renewal)), by = residence_area_type]
  te <- merge(te, temp, by = "residence_area_type", all.x = TRUE)
  te[, renewal := NULL]
  return(list(train = res, test = te))
}

# Cleaning data
result <- clean_data()
result_cat_handled <- handle_cat_variables(result)
train <- result_cat_handled$train
test <- result_cat_handled$test[]

# Using xgboost to build the model
# ================================

# Splitting data into train and test
train_ind <- createDataPartition(y = train$renewal, p = 0.8, list = FALSE)
m_train <- train[train_ind]
m_test <- train[-train_ind]

temp <- m_train[, .(mean_y_count_3_6 = mean(renewal), mean_perc_prem_3_6 = mean(perc_premium_paid_by_cash_credit)), by = "Count_3-6_months_late"]
m_test <- merge(m_test, temp, by = "Count_3-6_months_late", all.x = TRUE)
m_train <- merge(m_train, temp, by = "Count_3-6_months_late", all.x = TRUE)
temp <- m_train[, .(mean_y_count_6_12 = mean(renewal), mean_perc_prem_6_12 = mean(perc_premium_paid_by_cash_credit)), by = "Count_6-12_months_late"]
m_test <- merge(m_test, temp, by = "Count_6-12_months_late", all.x = TRUE)
m_train <- merge(m_train, temp, by = "Count_6-12_months_late", all.x = TRUE)
temp <- m_train[, .(mean_y_count_more_than_12 = mean(renewal), mean_perc_prem_12 = mean(perc_premium_paid_by_cash_credit)), by = "Count_more_than_12_months_late"]
m_test <- merge(m_test, temp, by = "Count_more_than_12_months_late", all.x = TRUE)
m_train <- merge(m_train, temp, by = "Count_more_than_12_months_late", all.x = TRUE)

temp <- train[, .(mean_y_count_3_6 = mean(renewal), mean_perc_prem_3_6 = mean(perc_premium_paid_by_cash_credit)), by = "Count_3-6_months_late"]
test <- merge(test, temp, by = "Count_3-6_months_late", all.x = TRUE)
temp <- train[, .(mean_y_count_6_12 = mean(renewal), mean_perc_prem_6_12 = mean(perc_premium_paid_by_cash_credit)), by = "Count_6-12_months_late"]
test <- merge(test, temp, by = "Count_6-12_months_late", all.x = TRUE)
temp <- train[, .(mean_y_count_more_than_12 = mean(renewal), mean_perc_prem_12 = mean(perc_premium_paid_by_cash_credit)), by = "Count_more_than_12_months_late"]
test <- merge(test, temp, by = "Count_more_than_12_months_late", all.x = TRUE)


# List independent and dependent variables
y <- "renewal"
feats <- setdiff(colnames(test), c("id", "sourcing_channel", "residence_area_type"))
# feats <- setdiff(feats, c("sourcing_channelA", "sourcing_channelB", "sourcing_channelC", "sourcing_channelD", "sourcing_channelE", "residence_area_typeUrban", "fold"))

# Preparing xgb matrix for train and test
dtrain <- xgb.DMatrix(data = as.matrix(m_train[, feats, with = FALSE]), label = m_train[[y]])
dtest <- xgb.DMatrix(data = as.matrix(m_test[, feats, with = FALSE]), label = m_test[[y]])
watchlist <- list(eval = dtest, train = dtrain)

# Creating params list and running xgb model
# my_etas <- list(eta = c(0.5, 0.1))
# callbacks = list(cb.reset.parameters(my_etas))
param <- list(max_depth = 2, eta = 0.1, nthread = -1, objective = "binary:logistic", eval_metric = "auc", min_child_weight = 5, gamma = 0.0001, subsample = 0.7, colsample_bytree = 0.7)
xgb_model <- xgb.train(param, dtrain, watchlist, nrounds = 200, early_stopping_rounds = 10)

# prev -- 0.838181

# Predicting validation and actual test data
pred_m_test <- predict(xgb_model, dtest)
cat("Validation data AUC := ", auc(m_test[[y]], pred_m_test))
pred_test_actual <- predict(xgb_model, as.matrix(test[, feats, with = FALSE]))
test[, renewal := pred_test_actual]
renewal_solution <- test[, .(id, renewal)]

# Visualize prob distribution of internal validation with actual test
p1 <- hist(pred_test_actual)
p2 <- hist(pred_m_test)
c1 = rgb(173,216,230,max = 255, alpha = 80, names = "lt.blue")
c2 = rgb(255,192,203, max = 255, alpha = 80, names = "lt.pink")
plot(p1, col = c1)
plot(p2, col = c2, add = TRUE)

ss <- fread("sample_submission_sLex1ul.csv")
ss <- merge(ss, renewal_solution, by = "id", all.x = TRUE)
ss[, renewal.x := NULL]
colnames(ss)[3] <- "renewal"
ss <- ss[, .(id, renewal, incentives)]
fwrite(ss, "xgb_renewal_no_incentives.csv")

# Computing incentives

renewal_solution[, renewal_c := (1 - renewal)*100]
renewal_solution[, effort_hours := 20*(1-exp(-renewal_c/5))]
renewal_solution[, effort_hours_2 := -5*log(1-(renewal_c/20))]
renewal_solution[, incentives := 10*(1-exp(-effort_hours/400))]
renewal_solution[, incentives_2 := -400*log(1-(effort_hours_2/10))]

app_1 <- renewal_solution[, .(id, renewal, incentives)] # 0.710788
app_2 <- renewal_solution[, .(id, renewal, incentives_2)] # 0.710836
colnames(app_2)[3] <- "incentives"
fwrite(app_1, "incentive_aap_1_v5.csv")
fwrite(app_2, "incentive_aap_2_clip.csv")


