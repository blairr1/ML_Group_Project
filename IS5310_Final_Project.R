#This is for the ML Project forecasting wine sales
# Group members includes:
# Stephanie Bounds, Ryan Blair, Brad Grey
#
############################################################################
#Packages to Install 
install.packages("forecast")
install.packages("ggplot2")
install.packages("tseries")
install.packages("dplyr")
install.packages("tidyverse")
install.packages("lubridate")
install.packages('stringi')

#Installing librarys
library(ggplot2)
library(tseries)
library(dplyr)
library(tidyverse)
library(lubridate)
library(forecast)
library(knitr)

#Read the wine dataset from a csv file. Point toward correct path when running
wine_dataset <- read.csv('C:/Users/Ryan/Desktop/AustralianWines.csv')

# This line is taking the six different wines that we want to forecast and making a "time series object" to 
# use in the modeling below and allow for a check for seasonality. Because we are inputting a frequency of 12, 
# it is interpreting the data as monthly data and will check for monthly seasonality.

fortified_ts <- ts(wine_dataset[2], start=c(1980,1), end = c(1994, 12), frequency = 12)
red_ts <- ts(wine_dataset[3], start=c(1980,1), end = c(1994, 12), frequency = 12)
rose_ts <- ts(wine_dataset[4], start=c(1980,1), end = c(1994, 12), frequency = 12)
sparkling_ts <- ts(wine_dataset[5], start=c(1980,1), end = c(1994, 12), frequency = 12)
sweetwhite_ts <- ts(wine_dataset[6], start=c(1980,1), end = c(1994, 12), frequency = 12)
drywhite_ts <- ts(wine_dataset[7], start=c(1980,1), end = c(1994, 12), frequency = 12)

#Create a time series object for the entire dataset (for use in plotting below)
ts <-ts(wine_dataset, frequency = 12)
# Delete the first column (Month) and create a new time series object
# This will clean up the plot so only the 6 wine datasets will plot
wine_ts <- ts[,-1]
plot(wine_ts)

#For testing and training the models, set up the length of validation and training
num_Valid = 12 
num_Train = length(fortified_ts) - num_Valid

#######################################################################
# Create the date range of our dataset for a yearly look of the data
start_date <- '1980/1/1'
years <- 15
months_in_a_year <- 12

#Create dataframe containing the wine data to use below
#Placing this in one dataframe will allow us to reuse the "df" call
df <- data.frame("date" = seq(as.Date(start_date), 
                              by = 'month', 
                              length.out = months_in_a_year*years),
                 "sales" = ts)
#######################################################################
#######################################################################
#
# Fortified Wine
# 
#######################################################################

# Plot the Fortified wine data. 
#Looking a yearly graph of the data to determine if there's seasonal periods for the fortified wine
autoplot(fortified_ts)

# Fortified_p will organize the data for each year of wine sales per month of the year. 
fortified_p <- df %>% 
  mutate(
    year = factor(year(date)),     # use year to define separate curves
    date = update(date, year = 1)  # use a constant year for the x-axis
  ) %>% 
  ggplot(aes(date, df$sales.Fortified, color = year)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  labs(x = "Month", y = "Fortified Wine Sales", title = "Fortified Wine Sales by Month")

# Plot the seasonal plot for wine sales for fortified wine
fortified_p + geom_line()

###########################################################################

# Peform the train and validation for the following methods:
# naive, seasonal naive, drift, Holt, Dampened Holt, Holt Winter (Additive, Multiplicative)

###########################################################################

# Partition our data into a train and validation set
fort_train = window(fortified_ts, start=c(1980,1), end=c(1993,12)) 
fort_validate = window(fortified_ts, start=c(1994, 1), end = c(1994, 12))

##########################
# NAIVE FORECAST
##########################

# Perform Naive forecast with the training dataset
fort_naive_train <- naive(fort_train, h=12)

# Capture the forecasted values for the next 12 months
fort_naive_train_proj <- fort_naive_train$mean

# Check the accuracy of the the forecast against the validation data
fort_naive_acc <- accuracy(fort_naive_train, fort_validate)
fort_naive_acc

# Calculate the residuals and plot in a histogram
fort_naive_residuals <- fort_naive_train_proj-fort_validate
hist(fort_naive_residuals)

##########################
# Drift FORECAST
##########################

# Perform Drift forecast with the training dataset
fort_drift_train <- rwf(fort_train, h=12)

# Capture the forecasted values for the next 12 months
fort_drift_train_proj <- fort_drift_train$mean

# Check the accuracy of the the forecast against the validation data
fort_drift_acc <- accuracy(fort_drift_train, fort_validate)
fort_drift_acc

# Calculate the residuals and plot in a histogram
fort_drift_residuals <- fort_drift_train_proj-fort_validate
hist(fort_drift_residuals)

##########################
# SEASONAL NAIVE FORECAST
##########################

# Perform Seasonal Naive forecast with the training dataset
fort_snaive_train <- snaive(fort_train, h=12)

# Capture the forecasted values for the next 12 months
fort_snaive_train_proj <- fort_snaive_train$mean

# Check the accuracy of the the forecast against the validation data
fort_snaive_acc <- accuracy(fort_snaive_train, fort_validate)
fort_snaive_acc

# Calculate the residuals and plot in a histogram
fort_snaive_residuals <- fort_snaive_train_proj-fort_validate
hist(fort_snaive_residuals)

##########################
# HOLT FORECAST
##########################

# Perform Holt forecast with the training dataset
fort_holt_train <- holt(fort_train, h=12)

# Capture the forecasted values for the next 12 months
fort_holt_train_proj <- fort_holt_train$mean

# Check the accuracy of the the forecast against the validation data
fort_holt_acc <- accuracy(fort_holt_train, fort_validate)
fort_holt_acc

# Calculate the residuals and plot in a histogram
fort_holt_residuals <- fort_holt_train_proj-fort_validate
hist(fort_holt_residuals)

##########################
# DAMPENED HOLT FORECAST
##########################

# Perform Dampened Holt forecast with the training dataset
fort_dholt_train <- holt(fort_train, damped=TRUE, phi = 0.9, h=12)

# Capture the forecasted values for the next 12 months
fort_dholt_train_proj <- fort_dholt_train$mean

# Check the accuracy of the the forecast against the validation data
fort_dholt_acc <- accuracy(fort_dholt_train, fort_validate)
fort_dholt_acc

# Calculate the residuals and plot in a histogram
fort_dholt_residuals <- fort_dholt_train_proj-fort_validate
hist(fort_dholt_residuals)


################################
# HOLT WINTER ADDITIVE FORECAST
################################

# Perform Holt Winter Additive forecast with the training dataset
fort_hwA_train <- hw(fort_train, seasonal="additive", h=12)

# Capture the forecasted values for the next 12 months
fort_hwA_train_proj <- fort_hwA_train$mean

# Check the accuracy of the the forecast against the validation data
fort_hwA_acc <- accuracy(fort_hwA_train, fort_validate)
fort_hwA_acc

# Calculate the residuals and plot in a histogram
fort_hwA_residuals <- fort_hwA_train_proj-fort_validate
hist(fort_hwA_residuals)


#####################################
# HOLT WINTER MULTIPLICATIVE FORECAST
#####################################

# Perform Holt Winter Multiplicative forecast with the training dataset
fort_hwM_train <- hw(fort_train, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
fort_hwM_train_proj <- fort_hwM_train$mean

# Check the accuracy of the the forecast against the validation data
fort_hwM_acc <- accuracy(fort_hwM_train, fort_validate)
fort_hwM_acc

# Calculate the residuals and plot in a histogram
fort_hwM_residuals <- fort_hwM_train_proj-fort_validate
hist(fort_hwM_residuals)

##############################################
# HOLT WINTER MULTIPLICATIVE DAMPENED FORECAST
##############################################

# Perform Holt Winter Multiplicative dampened forecast with the training dataset
fort_dhwM_train <- hw(fort_train, damped=TRUE, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
fort_dhwM_train_proj <- fort_dhwM_train$mean

# Check the accuracy of the the forecast against the validation data
fort_dhwM_acc <- accuracy(fort_dhwM_train, fort_validate)
fort_dhwM_acc

# Calculate the residuals and plot in a histogram
fort_dhwM_residuals <- fort_dhwM_train_proj-fort_validate
hist(fort_dhwM_residuals)


###########################################
# Exponential Smoothing Forecasting Methods
###########################################
fortified_mod <- ets(fortified_ts)
summary(fortified_mod)
fortified_forecast <- forecast(fortified_mod, h = 12)
plot(fortified_forecast)
fortified_forecast

###############################################################################
#
#Red Wine
#
###############################################################################
#Plot the red wine sales
autoplot(red_ts)

# red_p will organize the data for each year of wine sales per month of the year. 
red_p <- df %>% 
  mutate(
    year = factor(year(date)),     # use year to define separate curves
    date = update(date, year = 1)  # use a constant year for the x-axis
  ) %>% 
  ggplot(aes(date, df$sales.Red, color = year)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  labs(x = "Month", y = "Red Wine Sales", title = "Red Wine Sales by Month")

# Plot the seasonal plot for wine sales for red wine
red_p + geom_line()

###########################################################################

# Peform the train and validation for the following methods:
# naive, seasonal naive, drift, Holt, Dampened Holt, Holt Winter (Additive, Multiplicative)

###########################################################################

# Partition our data into a train and validation set
red_train = window(red_ts, start=c(1980,1), end=c(1993,12)) 
red_validate = window(red_ts, start=c(1994, 1), end = c(1994, 12))

##########################
# NAIVE FORECAST
##########################

# Perform Naive forecast with the training dataset
red_naive_train <- naive(red_train, h=12)

# Capture the forecasted values for the next 12 months
red_naive_train_proj <- red_naive_train$mean

# Check the accuracy of the the forecast against the validation data
red_naive_acc <- accuracy(red_naive_train, red_validate)
red_naive_acc

# Calculate the residuals and plot in a histogram
red_naive_residuals <- red_naive_train_proj-red_validate
hist(red_naive_residuals)

##########################
# Drift FORECAST
##########################

# Perform Drift forecast with the training dataset
red_drift_train <- rwf(red_train, h=12)

# Capture the forecasted values for the next 12 months
red_drift_train_proj <- red_drift_train$mean

# Check the accuracy of the the forecast against the validation data
red_drift_acc <- accuracy(red_drift_train, red_validate)
red_drift_acc

# Calculate the residuals and plot in a histogram
red_drift_residuals <- red_drift_train_proj-red_validate
hist(red_drift_residuals)

##########################
# SEASONAL NAIVE FORECAST
##########################

# Perform Seasonal Naive forecast with the training dataset
red_snaive_train <- snaive(red_train, h=12)

# Capture the forecasted values for the next 12 months
red_snaive_train_proj <- red_snaive_train$mean

# Check the accuracy of the the forecast against the validation data
red_snaive_acc <- accuracy(red_snaive_train, red_validate)
red_snaive_acc

# Calculate the residuals and plot in a histogram
red_snaive_residuals <- red_snaive_train_proj-red_validate
hist(red_snaive_residuals)

##########################
# HOLT FORECAST
##########################

# Perform Holt forecast with the training dataset
red_holt_train <- holt(red_train, h=12)

# Capture the forecasted values for the next 12 months
red_holt_train_proj <- red_holt_train$mean

# Check the accuracy of the the forecast against the validation data
red_holt_acc <- accuracy(red_holt_train, red_validate)
red_holt_acc

# Calculate the residuals and plot in a histogram
red_holt_residuals <- red_holt_train_proj-red_validate
hist(red_holt_residuals)

##########################
# DAMPENED HOLT FORECAST
##########################

# Perform Dampened Holt forecast with the training dataset
red_dholt_train <- holt(red_train, damped=TRUE, phi = 0.9, h=12)

# Capture the forecasted values for the next 12 months
red_dholt_train_proj <- red_dholt_train$mean

# Check the accuracy of the the forecast against the validation data
red_dholt_acc <- accuracy(red_dholt_train, red_validate)
red_dholt_acc

# Calculate the residuals and plot in a histogram
red_dholt_residuals <- red_dholt_train_proj-red_validate
hist(red_dholt_residuals)


################################
# HOLT WINTER ADDITIVE FORECAST
################################

# Perform Holt Winter Additive forecast with the training dataset
red_hwA_train <- hw(red_train, seasonal="additive", h=12)

# Capture the forecasted values for the next 12 months
red_hwA_train_proj <- red_hwA_train$mean

# Check the accuracy of the the forecast against the validation data
red_hwA_acc <- accuracy(red_hwA_train, red_validate)
red_hwA_acc

# Calculate the residuals and plot in a histogram
red_hwA_residuals <- red_hwA_train_proj-red_validate
hist(red_hwA_residuals)


#####################################
# HOLT WINTER MULTIPLICATIVE FORECAST
#####################################

# Perform Holt Winter Multiplicative forecast with the training dataset
red_hwM_train <- hw(red_train, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
red_hwM_train_proj <- red_hwM_train$mean

# Check the accuracy of the the forecast against the validation data
red_hwM_acc <- accuracy(red_hwM_train, red_validate)
red_hwM_acc

# Calculate the residuals and plot in a histogram
red_hwM_residuals <- red_hwM_train_proj-red_validate
hist(red_hwM_residuals)

##############################################
# HOLT WINTER MULTIPLICATIVE DAMPENED FORECAST
##############################################

# Perform Holt Winter Multiplicative dampened forecast with the training dataset
red_dhwM_train <- hw(red_train, damped=TRUE, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
red_dhwM_train_proj <- red_dhwM_train$mean

# Check the accuracy of the the forecast against the validation data
red_dhwM_acc <- accuracy(red_dhwM_train, red_validate)
red_dhwM_acc

# Calculate the residuals and plot in a histogram
red_dhwM_residuals <- red_dhwM_train_proj-red_validate
hist(red_dhwM_residuals)

###########################################
# Exponential Smoothing Forecasting Methods
###########################################
red_mod <- ets(red_ts)
summary(red_mod)
red_forecast <- forecast(red_mod, h = 12)
plot(red_forecast)
red_forecast


############################################################################
#
#Rose 
#
############################################################################
#Plot the rose wine data
autoplot(rose_ts)

# rose_p will organize the data for each year of wine sales per month of the year. 
rose_p <- df %>% 
  mutate(
    year = factor(year(date)),     # use year to define separate curves
    date = update(date, year = 1)  # use a constant year for the x-axis
  ) %>% 
  ggplot(aes(date, df$sales.Rose, color = year)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  labs(x = "Month", y = "Rose Wine Sales", title = "Rose Wine Sales by Month")

# Plot the seasonal plot for wine sales for rose wine
rose_p + geom_line()

###########################################################################

# Peform the train and validation for the following methods:
# naive, seasonal naive, drift, Holt, Dampened Holt, Holt Winter (Additive, Multiplicative)

###########################################################################

# Partition our data into a train and validation set
rose_train = window(rose_ts, start=c(1980,1), end=c(1993,12)) 
rose_validate = window(rose_ts, start=c(1994, 1), end = c(1994, 12))

##########################
# NAIVE FORECAST
##########################

# Perform Naive forecast with the training dataset
rose_naive_train <- naive(rose_train, h=12)

# Capture the forecasted values for the next 12 months
rose_naive_train_proj <- rose_naive_train$mean

# Check the accuracy of the the forecast against the validation data
rose_naive_acc <- accuracy(rose_naive_train, rose_validate)
rose_naive_acc

# Calculate the residuals and plot in a histogram
rose_naive_residuals <- rose_naive_train_proj-rose_validate
hist(rose_naive_residuals)

##########################
# Drift FORECAST
##########################

# Perform Drift forecast with the training dataset
rose_drift_train <- rwf(rose_train, h=12)

# Capture the forecasted values for the next 12 months
rose_drift_train_proj <- rose_drift_train$mean

# Check the accuracy of the the forecast against the validation data
rose_drift_acc <- accuracy(rose_drift_train, rose_validate)
rose_drift_acc

# Calculate the residuals and plot in a histogram
rose_drift_residuals <- rose_drift_train_proj-rose_validate
hist(rose_drift_residuals)

##########################
# SEASONAL NAIVE FORECAST
##########################

# Perform Seasonal Naive forecast with the training dataset
rose_snaive_train <- snaive(rose_train, h=12)

# Capture the forecasted values for the next 12 months
rose_snaive_train_proj <- rose_snaive_train$mean

# Check the accuracy of the the forecast against the validation data
rose_snaive_acc <- accuracy(rose_snaive_train, rose_validate)
rose_snaive_acc

# Calculate the residuals and plot in a histogram
rose_snaive_residuals <- rose_snaive_train_proj-rose_validate
hist(rose_snaive_residuals)

##########################
# HOLT FORECAST
##########################

# Perform Holt forecast with the training dataset
rose_holt_train <- holt(rose_train, h=12)

# Capture the forecasted values for the next 12 months
rose_holt_train_proj <- rose_holt_train$mean

# Check the accuracy of the the forecast against the validation data
rose_holt_acc <- accuracy(rose_holt_train, rose_validate)
rose_holt_acc

# Calculate the residuals and plot in a histogram
rose_holt_residuals <- rose_holt_train_proj-rose_validate
hist(rose_holt_residuals)

##########################
# DAMPENED HOLT FORECAST
##########################

# Perform Dampened Holt forecast with the training dataset
rose_dholt_train <- holt(rose_train, damped=TRUE, phi = 0.9, h=12)

# Capture the forecasted values for the next 12 months
rose_dholt_train_proj <- rose_dholt_train$mean

# Check the accuracy of the the forecast against the validation data
rose_dholt_acc <- accuracy(rose_dholt_train, rose_validate)
rose_dholt_acc

# Calculate the residuals and plot in a histogram
rose_dholt_residuals <- rose_dholt_train_proj-rose_validate
hist(rose_dholt_residuals)


################################
# HOLT WINTER ADDITIVE FORECAST
################################

# Perform Holt Winter Additive forecast with the training dataset
rose_hwA_train <- hw(rose_train, seasonal="additive", h=12)

# Capture the forecasted values for the next 12 months
rose_hwA_train_proj <- rose_hwA_train$mean

# Check the accuracy of the the forecast against the validation data
rose_hwA_acc <- accuracy(rose_hwA_train, rose_validate)
rose_hwA_acc

# Calculate the residuals and plot in a histogram
rose_hwA_residuals <- rose_hwA_train_proj-rose_validate
hist(rose_hwA_residuals)


#####################################
# HOLT WINTER MULTIPLICATIVE FORECAST
#####################################

# Perform Holt Winter Multiplicative forecast with the training dataset
rose_hwM_train <- hw(rose_train, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
rose_hwM_train_proj <- rose_hwM_train$mean

# Check the accuracy of the the forecast against the validation data
rose_hwM_acc <- accuracy(rose_hwM_train, rose_validate)
rose_hwM_acc

# Calculate the residuals and plot in a histogram
rose_hwM_residuals <- rose_hwM_train_proj-rose_validate
hist(rose_hwM_residuals)

##############################################
# HOLT WINTER MULTIPLICATIVE DAMPENED FORECAST
##############################################

# Perform Holt Winter Multiplicative dampened forecast with the training dataset
rose_dhwM_train <- hw(rose_train, damped=TRUE, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
rose_dhwM_train_proj <- rose_dhwM_train$mean

# Check the accuracy of the the forecast against the validation data
rose_dhwM_acc <- accuracy(rose_dhwM_train, rose_validate)
rose_dhwM_acc

# Calculate the residuals and plot in a histogram
rose_dhwM_residuals <- rose_dhwM_train_proj-rose_validate
hist(rose_dhwM_residuals)


###########################################
# Exponential Smoothing Forecasting Methods
###########################################
rose_mod <- ets(rose_ts)
summary(rose_mod)
rose_forecast <- forecast(rose_mod, h = 12)
plot(rose_forecast)
rose_forecast


################################################################################
#
#Sparkling
#
################################################################################
#Plot the sparkling wine sales data
autoplot(sparkling_ts)

# sparkling_p will organize the data for each year of wine sales per month of the year. 
sparkling_p <- df %>% 
  mutate(
    year = factor(year(date)),     # use year to define separate curves
    date = update(date, year = 1)  # use a constant year for the x-axis
  ) %>% 
  ggplot(aes(date, df$sales.sparkling, color = year)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  labs(x = "Month", y = "Sparkling Wine Sales", title = "Sparkling Wine Sales by Month")

# Plot the seasonal plot for wine sales for sparkling wine
sparkling_p + geom_line()

###########################################################################

# Peform the train and validation for the following methods:
# naive, seasonal naive, drift, Holt, Dampened Holt, Holt Winter (Additive, Multiplicative)

###########################################################################

# Partition our data into a train and validation set
spark_train = window(sparkling_ts, start=c(1980,1), end=c(1993,12)) 
spark_validate = window(sparkling_ts, start=c(1994, 1), end = c(1994, 12))

##########################
# NAIVE FORECAST
##########################

# Perform Naive forecast with the training dataset
spark_naive_train <- naive(spark_train, h=12)

# Capture the forecasted values for the next 12 months
spark_naive_train_proj <- spark_naive_train$mean

# Check the accuracy of the the forecast against the validation data
spark_naive_acc <- accuracy(spark_naive_train, spark_validate)
spark_naive_acc

# Calculate the residuals and plot in a histogram
spark_naive_residuals <- spark_naive_train_proj-spark_validate
hist(spark_naive_residuals)

##########################
# Drift FORECAST
##########################

# Perform Drift forecast with the training dataset
spark_drift_train <- rwf(spark_train, h=12)

# Capture the forecasted values for the next 12 months
spark_drift_train_proj <- spark_drift_train$mean

# Check the accuracy of the the forecast against the validation data
spark_drift_acc <- accuracy(spark_drift_train, spark_validate)
spark_drift_acc

# Calculate the residuals and plot in a histogram
spark_drift_residuals <- spark_drift_train_proj-spark_validate
hist(spark_drift_residuals)

##########################
# SEASONAL NAIVE FORECAST
##########################

# Perform Seasonal Naive forecast with the training dataset
spark_snaive_train <- snaive(spark_train, h=12)

# Capture the forecasted values for the next 12 months
spark_snaive_train_proj <- spark_snaive_train$mean

# Check the accuracy of the the forecast against the validation data
spark_snaive_acc <- accuracy(spark_snaive_train, spark_validate)
spark_snaive_acc

# Calculate the residuals and plot in a histogram
spark_snaive_residuals <- spark_snaive_train_proj-spark_validate
hist(spark_snaive_residuals)

##########################
# HOLT FORECAST
##########################

# Perform Holt forecast with the training dataset
spark_holt_train <- holt(spark_train, h=12)

# Capture the forecasted values for the next 12 months
spark_holt_train_proj <- spark_holt_train$mean

# Check the accuracy of the the forecast against the validation data
spark_holt_acc <- accuracy(spark_holt_train, spark_validate)
spark_holt_acc

# Calculate the residuals and plot in a histogram
spark_holt_residuals <- spark_holt_train_proj-spark_validate
hist(spark_holt_residuals)

##########################
# DAMPENED HOLT FORECAST
##########################

# Perform Dampened Holt forecast with the training dataset
spark_dholt_train <- holt(spark_train, damped=TRUE, phi = 0.9, h=12)

# Capture the forecasted values for the next 12 months
spark_dholt_train_proj <- spark_dholt_train$mean

# Check the accuracy of the the forecast against the validation data
spark_dholt_acc <- accuracy(spark_dholt_train, spark_validate)
spark_dholt_acc

# Calculate the residuals and plot in a histogram
spark_dholt_residuals <- spark_dholt_train_proj-spark_validate
hist(spark_dholt_residuals)


################################
# HOLT WINTER ADDITIVE FORECAST
################################

# Perform Holt Winter Additive forecast with the training dataset
spark_hwA_train <- hw(spark_train, seasonal="additive", h=12)

# Capture the forecasted values for the next 12 months
spark_hwA_train_proj <- spark_hwA_train$mean

# Check the accuracy of the the forecast against the validation data
spark_hwA_acc <- accuracy(spark_hwA_train, spark_validate)
spark_hwA_acc

# Calculate the residuals and plot in a histogram
spark_hwA_residuals <- spark_hwA_train_proj-spark_validate
hist(spark_hwA_residuals)


#####################################
# HOLT WINTER MULTIPLICATIVE FORECAST
#####################################

# Perform Holt Winter Multiplicative forecast with the training dataset
spark_hwM_train <- hw(spark_train, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
spark_hwM_train_proj <- spark_hwM_train$mean

# Check the accuracy of the the forecast against the validation data
spark_hwM_acc <- accuracy(spark_hwM_train, spark_validate)
spark_hwM_acc

# Calculate the residuals and plot in a histogram
spark_hwM_residuals <- spark_hwM_train_proj-spark_validate
hist(spark_hwM_residuals)

##############################################
# HOLT WINTER MULTIPLICATIVE DAMPENED FORECAST
##############################################

# Perform Holt Winter Multiplicative dampened forecast with the training dataset
spark_dhwM_train <- hw(spark_train, damped=TRUE, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
spark_dhwM_train_proj <- spark_dhwM_train$mean

# Check the accuracy of the the forecast against the validation data
spark_dhwM_acc <- accuracy(spark_dhwM_train, spark_validate)
spark_dhwM_acc

# Calculate the residuals and plot in a histogram
spark_dhwM_residuals <- spark_dhwM_train_proj-spark_validate
hist(spark_dhwM_residuals)


###########################################
# Exponential Smoothing Forecasting Methods
###########################################
sparkling_mod <- ets(sparkling_ts)
summary(sparkling_mod)
sparkling_forecast <- forecast(sparkling_mod, h = 12)
plot(sparkling_forecast)
sparkling_forecast


################################################################################
#
#Sweet White
#
################################################################################
#Plot the sweetwhite wine sales
plot(sweetwhite_ts)

# sweetwhite_p will organize the data for each year of wine sales per month of the year. 
sweetwhite_p <- df %>% 
  mutate(
    year = factor(year(date)),     # use year to define separate curves
    date = update(date, year = 1)  # use a constant year for the x-axis
  ) %>% 
  ggplot(aes(date, df$sales.Sweet.white, color = year)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  labs(x = "Month", y = "Sweet White Wine Sales", title = "Sweet White Wine Sales by Month")

# Plot the seasonal plot for wine sales for sweet white wine
sweetwhite_p + geom_line()


###########################################################################

# Peform the train and validation for the following methods:
# naive, seasonal naive, drift, Holt, Dampened Holt, Holt Winter (Additive, Multiplicative)

###########################################################################

# Partition our data into a train and validation set
sweetw_train = window(sweetwhite_ts, start=c(1980,1), end=c(1993,12)) 
sweetw_validate = window(sweetwhite_ts, start=c(1994, 1), end = c(1994, 12))

##########################
# NAIVE FORECAST
##########################

# Perform Naive forecast with the training dataset
sweetw_naive_train <- naive(sweetw_train, h=12)

# Capture the forecasted values for the next 12 months
sweetw_naive_train_proj <- sweetw_naive_train$mean

# Check the accuracy of the the forecast against the validation data
sweetw_naive_acc <- accuracy(sweetw_naive_train, sweetw_validate)
sweetw_naive_acc

# Calculate the residuals and plot in a histogram
sweetw_naive_residuals <- sweetw_naive_train_proj-sweetw_validate
hist(sweetw_naive_residuals)

##########################
# Drift FORECAST
##########################

# Perform Drift forecast with the training dataset
sweetw_drift_train <- rwf(sweetw_train, h=12)

# Capture the forecasted values for the next 12 months
sweetw_drift_train_proj <- sweetw_drift_train$mean

# Check the accuracy of the the forecast against the validation data
sweetw_drift_acc <- accuracy(sweetw_drift_train, sweetw_validate)
sweetw_drift_acc

# Calculate the residuals and plot in a histogram
sweetw_drift_residuals <- sweetw_drift_train_proj-sweetw_validate
hist(sweetw_drift_residuals)

##########################
# SEASONAL NAIVE FORECAST
##########################

# Perform Seasonal Naive forecast with the training dataset
sweetw_snaive_train <- snaive(sweetw_train, h=12)

# Capture the forecasted values for the next 12 months
sweetw_snaive_train_proj <- sweetw_snaive_train$mean

# Check the accuracy of the the forecast against the validation data
sweetw_snaive_acc <- accuracy(sweetw_snaive_train, sweetw_validate)
sweetw_snaive_acc

# Calculate the residuals and plot in a histogram
sweetw_snaive_residuals <- sweetw_snaive_train_proj-sweetw_validate
hist(sweetw_snaive_residuals)

##########################
# HOLT FORECAST
##########################

# Perform Holt forecast with the training dataset
sweetw_holt_train <- holt(sweetw_train, h=12)

# Capture the forecasted values for the next 12 months
sweetw_holt_train_proj <- sweetw_holt_train$mean

# Check the accuracy of the the forecast against the validation data
sweetw_holt_acc <- accuracy(sweetw_holt_train, sweetw_validate)
sweetw_holt_acc

# Calculate the residuals and plot in a histogram
sweetw_holt_residuals <- sweetw_holt_train_proj-sweetw_validate
hist(sweetw_holt_residuals)

##########################
# DAMPENED HOLT FORECAST
##########################

# Perform Dampened Holt forecast with the training dataset
sweetw_dholt_train <- holt(sweetw_train, damped=TRUE, phi = 0.9, h=12)

# Capture the forecasted values for the next 12 months
sweetw_dholt_train_proj <- sweetw_dholt_train$mean

# Check the accuracy of the the forecast against the validation data
sweetw_dholt_acc <- accuracy(sweetw_dholt_train, sweetw_validate)
sweetw_dholt_acc

# Calculate the residuals and plot in a histogram
sweetw_dholt_residuals <- sweetw_dholt_train_proj-sweetw_validate
hist(sweetw_dholt_residuals)


################################
# HOLT WINTER ADDITIVE FORECAST
################################

# Perform Holt Winter Additive forecast with the training dataset
sweetw_hwA_train <- hw(sweetw_train, seasonal="additive", h=12)

# Capture the forecasted values for the next 12 months
sweetw_hwA_train_proj <- sweetw_hwA_train$mean

# Check the accuracy of the the forecast against the validation data
sweetw_hwA_acc <- accuracy(sweetw_hwA_train, sweetw_validate)
sweetw_hwA_acc

# Calculate the residuals and plot in a histogram
sweetw_hwA_residuals <- sweetw_hwA_train_proj-sweetw_validate
hist(sweetw_hwA_residuals)


#####################################
# HOLT WINTER MULTIPLICATIVE FORECAST
#####################################

# Perform Holt Winter Multiplicative forecast with the training dataset
sweetw_hwM_train <- hw(sweetw_train, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
sweetw_hwM_train_proj <- sweetw_hwM_train$mean

# Check the accuracy of the the forecast against the validation data
sweetw_hwM_acc <- accuracy(sweetw_hwM_train, sweetw_validate)
sweetw_hwM_acc

# Calculate the residuals and plot in a histogram
sweetw_hwM_residuals <- sweetw_hwM_train_proj-sweetw_validate
hist(sweetw_hwM_residuals)

##############################################
# HOLT WINTER MULTIPLICATIVE DAMPENED FORECAST
##############################################

# Perform Holt Winter Multiplicative dampened forecast with the training dataset
sweetw_dhwM_train <- hw(sweetw_train, damped=TRUE, seasonal="multiplicative", h=12)


# Capture the forecasted values for the next 12 months
sweetw_dhwM_train_proj <- sweetw_dhwM_train$mean

# Check the accuracy of the the forecast against the validation data
sweetw_dhwM_acc <- accuracy(sweetw_dhwM_train, sweetw_validate)
sweetw_dhwM_acc

# Calculate the residuals and plot in a histogram
sweetw_dhwM_residuals <- sweetw_dhwM_train_proj-sweetw_validate
hist(sweetw_dhwM_residuals)


###########################################
# Exponential Smoothing Forecasting Methods
###########################################
sweetwhite_mod <- ets(sweetwhite_ts)
summary(sweetwhite_mod)
sweetwhite_forecast <- forecast(sweetwhite_mod, h = 12)
plot(sweetwhite_forecast)
sweetwhite_forecast


################################################################################
#
#Dry White
#
################################################################################
#Plot the drywhite wine sales
autoplot(drywhite_ts)

# drywhite_p will organize the data for each year of wine sales per month of the year. 
drywhite_p <- df %>% 
  mutate(
    year = factor(year(date)),     # use year to define separate curves
    date = update(date, year = 1)  # use a constant year for the x-axis
  ) %>% 
  ggplot(aes(date, df$sales.Dry.white, color = year)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b") +
  labs(x = "Month", y = "Dry White Wine Sales", title = "Dry White Wine Sales by Month")

# Plot the seasonal plot for wine sales for dry white wine
drywhite_p + geom_line()

###########################################################################

# Peform the train and validation for the following methods:
# naive, seasonal naive, drift, Holt, Dampened Holt, Holt Winter (Additive, Multiplicative)

###########################################################################

# Partition our data into a train and validation set
dryw_train = window(drywhite_ts, start=c(1980,1), end=c(1993,12)) 
dryw_validate = window(drywhite_ts, start=c(1994, 1), end = c(1994, 12))

##########################
# NAIVE FORECAST
##########################

# Perform Naive forecast with the training dataset
dryw_naive_train <- naive(dryw_train, h=12)

# Capture the forecasted values for the next 12 months
dryw_naive_train_proj <- dryw_naive_train$mean

# Check the accuracy of the the forecast against the validation data
dryw_naive_acc <- accuracy(dryw_naive_train, dryw_validate)
dryw_naive_acc

# Calculate the residuals and plot in a histogram
dryw_naive_residuals <- dryw_naive_train_proj-dryw_validate
hist(dryw_naive_residuals)

##########################
# Drift FORECAST
##########################

# Perform Drift forecast with the training dataset
dryw_drift_train <- rwf(dryw_train, h=12)

# Capture the forecasted values for the next 12 months
dryw_drift_train_proj <- dryw_drift_train$mean

# Check the accuracy of the the forecast against the validation data
dryw_drift_acc <- accuracy(dryw_drift_train, dryw_validate)
dryw_drift_acc

# Calculate the residuals and plot in a histogram
dryw_drift_residuals <- dryw_drift_train_proj-dryw_validate
hist(dryw_drift_residuals)

##########################
# SEASONAL NAIVE FORECAST
##########################

# Perform Seasonal Naive forecast with the training dataset
dryw_snaive_train <- snaive(dryw_train, h=12)

# Capture the forecasted values for the next 12 months
dryw_snaive_train_proj <- dryw_snaive_train$mean

# Check the accuracy of the the forecast against the validation data
dryw_snaive_acc <- accuracy(dryw_snaive_train, dryw_validate)
dryw_snaive_acc

# Calculate the residuals and plot in a histogram
dryw_snaive_residuals <- dryw_snaive_train_proj-dryw_validate
hist(dryw_snaive_residuals)

##########################
# HOLT FORECAST
##########################

# Perform Holt forecast with the training dataset
dryw_holt_train <- holt(dryw_train, h=12)

# Capture the forecasted values for the next 12 months
dryw_holt_train_proj <- dryw_holt_train$mean

# Check the accuracy of the the forecast against the validation data
dryw_holt_acc <- accuracy(dryw_holt_train, dryw_validate)
dryw_holt_acc

# Calculate the residuals and plot in a histogram
dryw_holt_residuals <- dryw_holt_train_proj-dryw_validate
hist(dryw_holt_residuals)

##########################
# DAMPENED HOLT FORECAST
##########################

# Perform Dampened Holt forecast with the training dataset
dryw_dholt_train <- holt(dryw_train, damped=TRUE, phi = 0.9, h=12)

# Capture the forecasted values for the next 12 months
dryw_dholt_train_proj <- dryw_dholt_train$mean

# Check the accuracy of the the forecast against the validation data
dryw_dholt_acc <- accuracy(dryw_dholt_train, dryw_validate)
dryw_dholt_acc

# Calculate the residuals and plot in a histogram
dryw_dholt_residuals <- dryw_dholt_train_proj-dryw_validate
hist(dryw_dholt_residuals)


################################
# HOLT WINTER ADDITIVE FORECAST
################################

# Perform Holt Winter Additive forecast with the training dataset
dryw_hwA_train <- hw(dryw_train, seasonal="additive", h=12)

# Capture the forecasted values for the next 12 months
dryw_hwA_train_proj <- dryw_hwA_train$mean

# Check the accuracy of the the forecast against the validation data
dryw_hwA_acc <- accuracy(dryw_hwA_train, dryw_validate)
dryw_hwA_acc

# Calculate the residuals and plot in a histogram
dryw_hwA_residuals <- dryw_hwA_train_proj-dryw_validate
hist(dryw_hwA_residuals)


#####################################
# HOLT WINTER MULTIPLICATIVE FORECAST
#####################################

# Perform Holt Winter Multiplicative forecast with the training dataset
dryw_hwM_train <- hw(dryw_train, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
dryw_hwM_train_proj <- dryw_hwM_train$mean

# Check the accuracy of the the forecast against the validation data
dryw_hwM_acc <- accuracy(dryw_hwM_train, dryw_validate)
dryw_hwM_acc

# Calculate the residuals and plot in a histogram
dryw_hwM_residuals <- dryw_hwM_train_proj-dryw_validate
hist(dryw_hwM_residuals)

##############################################
# HOLT WINTER MULTIPLICATIVE DAMPENED FORECAST
##############################################

# Perform Holt Winter Multiplicative dampened forecast with the training dataset
dryw_dhwM_train <- hw(dryw_train, damped=TRUE, seasonal="multiplicative", h=12)

# Capture the forecasted values for the next 12 months
dryw_dhwM_train_proj <- dryw_dhwM_train$mean

# Check the accuracy of the the forecast against the validation data
dryw_dhwM_acc <- accuracy(dryw_dhwM_train, dryw_validate)
dryw_dhwM_acc

# Calculate the residuals and plot in a histogram
dryw_dhwM_residuals <- dryw_dhwM_train_proj-dryw_validate
hist(dryw_dhwM_residuals)


###########################################
# Exponential Smoothing Forecasting Methods
###########################################
drywhite_mod <- ets(drywhite_ts)
summary(drywhite_mod)
drywhite_forecast <- forecast(drywhite_mod, h = 12)
plot(drywhite_forecast)
drywhite_forecast




##############################################################
# This Section contains the forecasting model used for each wine
#This section will contain the plotting and graphing code used
#to put together our presentation and paper
#
##############################################################

##################
#Fortified
##################
#Plot the fortified data with a trend line
plot(fortified_ts, bty="l", main="Fortified Wine")
abline(reg=lm(fortified_ts~time(fortified_ts)))

#Create Tables to help determine best model to use for forecasting
kable(fort_naive_acc)
kable(fort_snaive_acc)
kable(fort_drift_acc)
kable(fort_holt_acc)
kable(fort_dholt_acc)
kable(fort_hwA_acc)
kable(fort_hwM_acc)
kable(fort_dhwM_acc)

#The Holt Winter Multiplicative forecast will be used
#Look at the predicted value versus the validation data
autoplot(fortified_ts)+
  autolayer(fort_hwM_train_proj)

#Create the forecast for the 1995 year with this method
fort_final_forecast <- hw(fortified_ts,seasonal="multiplicative", h =12)
kable(fort_final_forecast)

autoplot(fortified_ts) +
  autolayer(fort_final_forecast, series="HW multiplicative forecasts",PI=FALSE) +
  xlab("Year") +
  ylab("Wine Sales") +
  ggtitle("Fortified Wine Sales") +
  guides(colour=guide_legend(title="Forecast"))

# Create plot of other forecast methods Naive, Seasonal Naive, Drift, Holt
# Dampened Holt, HW Additive, HW Multiplicative, and HW Multiplicative Dampened
autoplot(fortified_ts) +
  autolayer(naive(fortified_ts, h=12),
            series="Naïve", PI=FALSE) +
  autolayer(snaive(fortified_ts, h=12),
            series="Seasonal naïve", PI=FALSE) +
  autolayer(rwf(fortified_ts, drift=TRUE, h=12),
            series="Drift", PI=FALSE) +
  autolayer(holt(fortified_ts, h=12), series="Holt's method", PI=FALSE) +
  autolayer(holt(fortified_ts, damped=TRUE, phi = 0.9, h=12), series="Damped Holt's method", PI=FALSE) +
  autolayer(hw(fortified_ts,seasonal="additive", h=12), series="HW additive forecasts", PI=FALSE) +
  autolayer(hw(fortified_ts,seasonal="multiplicative", h =12), series="HW multiplicative forecasts",
            PI=FALSE) +
  autolayer(hw(fortified_ts, damped=TRUE, seasonal="multiplicative", h=12), series = 'HW multiplicative dampened forecasts', 
            PI = FALSE) +
  ggtitle("Forecasts for Monthly Fortified Wine Sales") +
  xlab("Time") + ylab("Volume") +
  guides(colour=guide_legend(title="Forecast"))


##################
#Red
##################
#Plot the fortified data with a trend line
plot(red_ts, bty="l", main="Red Wine")
abline(reg=lm(red_ts~time(red_ts)))

#Create Tables to help determine best model to use for forecasting
kable(red_naive_acc)
kable(red_snaive_acc)
kable(red_drift_acc)
kable(red_holt_acc)
kable(red_dholt_acc)
kable(red_hwA_acc)
kable(red_hwM_acc)
kable(red_dhwM_acc)

#The Holt Winter Multiplicative forecast will be used
#Look at the predicted value versus the validation data
autoplot(red_ts)+
  autolayer(red_hwM_train_proj)

#Create the forecast for the 1995 year with this method
red_final_forecast <- hw(red_ts,seasonal="multiplicative", h =12)
kable(red_final_forecast)

autoplot(red_ts) +
  autolayer(red_final_forecast, series="HW multiplicative forecasts",PI=FALSE) +
  xlab("Year") +
  ylab("Wine Sales") +
  ggtitle("Red Wine Sales") +
  guides(colour=guide_legend(title="Forecast"))

# Create plot of other forecast methods Naive, Seasonal Naive, Drift, Holt
# Dampened Holt, HW Additive, HW Multiplicative, and HW Multiplicative Dampened
autoplot(red_ts) +
  autolayer(naive(red_ts, h=12),
            series="Naïve", PI=FALSE) +
  autolayer(snaive(red_ts, h=12),
            series="Seasonal naïve", PI=FALSE) +
  autolayer(rwf(red_ts, drift=TRUE, h=12),
            series="Drift", PI=FALSE) +
  autolayer(holt(red_ts, h=12), series="Holt's method", PI=FALSE) +
  autolayer(holt(red_ts, damped=TRUE, phi = 0.9, h=12), series="Damped Holt's method", PI=FALSE) +
  autolayer(hw(red_ts,seasonal="additive", h=12), series="HW additive forecasts", PI=FALSE) +
  autolayer(hw(red_ts,seasonal="multiplicative", h =12), series="HW multiplicative forecasts",
            PI=FALSE) +
  autolayer(hw(red_ts, damped=TRUE, seasonal="multiplicative", h=12), series = 'HW multiplicative dampened forecasts', 
            PI = FALSE) +
  ggtitle("Forecasts for Monthly Red Wine Sales") +
  xlab("Time") + ylab("Volume") +
  guides(colour=guide_legend(title="Forecast"))


##################
#Rose
##################
#Plot the fortified data with a trend line
plot(rose_ts, bty="l", main="Rose Wine")
abline(reg=lm(rose_ts~time(rose_ts)))

#Create Tables to help determine best model to use for forecasting
kable(rose_naive_acc)
kable(rose_snaive_acc)
kable(rose_drift_acc)
kable(rose_holt_acc)
kable(rose_dholt_acc)
kable(rose_hwA_acc)
kable(rose_hwM_acc)
kable(rose_dhwM_acc)

#The Holt Winter Multiplicative Dampened forecast will be used
#Look at the predicted value versus the validation data
autoplot(rose_ts)+
  autolayer(rose_dhwM_train_proj)

#Create the forecast for the 1995 year with this method
rose_final_forecast <- hw(rose_ts,damped=TRUE,seasonal="multiplicative", h =12)
kable(rose_final_forecast)

autoplot(rose_ts) +
  autolayer(rose_final_forecast, series="HW multiplicative dampened forecasts",PI=FALSE) +
  xlab("Year") +
  ylab("Wine Sales") +
  ggtitle("Rose Wine Sales") +
  guides(colour=guide_legend(title="Forecast"))

# Create plot of other forecast methods Naive, Seasonal Naive, Drift, Holt
# Dampened Holt, HW Additive, HW Multiplicative, and HW Multiplicative Dampened
autoplot(rose_ts) +
  autolayer(naive(rose_ts, h=12),
            series="Naïve", PI=FALSE) +
  autolayer(snaive(rose_ts, h=12),
            series="Seasonal naïve", PI=FALSE) +
  autolayer(rwf(rose_ts, drift=TRUE, h=12),
            series="Drift", PI=FALSE) +
  autolayer(holt(rose_ts, h=12), series="Holt's method", PI=FALSE) +
  autolayer(holt(rose_ts, damped=TRUE, phi = 0.9, h=12), series="Damped Holt's method", PI=FALSE) +
  autolayer(hw(rose_ts,seasonal="additive", h=12), series="HW additive forecasts", PI=FALSE) +
  autolayer(hw(rose_ts,seasonal="multiplicative", h =12), series="HW multiplicative forecasts",
            PI=FALSE) +
  autolayer(hw(rose_ts, damped=TRUE, seasonal="multiplicative", h=12), series = 'HW multiplicative dampened forecasts', 
            PI = FALSE) +
  ggtitle("Forecasts for Monthly Rose Wine Sales") +
  xlab("Time") + ylab("Volume") +
  guides(colour=guide_legend(title="Forecast"))

##################
#Sparkling
##################
#Plot the fortified data with a trend line
plot(sparkling_ts, bty="l", main="Sparkling Wine")
abline(reg=lm(sparkling_ts~time(sparkling_ts)))

#Create Tables to help determine best model to use for forecasting
kable(spark_naive_acc)
kable(spark_snaive_acc)
kable(spark_drift_acc)
kable(spark_holt_acc)
kable(spark_dholt_acc)
kable(spark_hwA_acc)
kable(spark_hwM_acc)
kable(spark_dhwM_acc)

#The Holt Winter Multiplicative forecast will be used
#Look at the predicted value versus the validation data
autoplot(sparkling_ts)+
  autolayer(spark_hwM_train_proj)

#Create the forecast for the 1995 year with this method
spark_final_forecast <- hw(sparkling_ts,seasonal="multiplicative", h =12)
kable(spark_final_forecast)

autoplot(sparkling_ts) +
  autolayer(spark_final_forecast, series="HW multiplicative forecasts",PI=FALSE) +
  xlab("Year") +
  ylab("Wine Sales") +
  ggtitle("Sparkling Wine Sales") +
  guides(colour=guide_legend(title="Forecast"))

# Create plot of other forecast methods Naive, Seasonal Naive, Drift, Holt
# Dampened Holt, HW Additive, HW Multiplicative, and HW Multiplicative Dampened
autoplot(sparkling_ts) +
  autolayer(naive(sparkling_ts, h=12),
            series="Naïve", PI=FALSE) +
  autolayer(snaive(sparkling_ts, h=12),
            series="Seasonal naïve", PI=FALSE) +
  autolayer(rwf(sparkling_ts, drift=TRUE, h=12),
            series="Drift", PI=FALSE) +
  autolayer(holt(sparkling_ts, h=12), series="Holt's method", PI=FALSE) +
  autolayer(holt(sparkling_ts, damped=TRUE, phi = 0.9, h=12), series="Damped Holt's method", PI=FALSE) +
  autolayer(hw(sparkling_ts,seasonal="additive", h=12), series="HW additive forecasts", PI=FALSE) +
  autolayer(hw(sparkling_ts,seasonal="multiplicative", h =12), series="HW multiplicative forecasts",
            PI=FALSE) +
  autolayer(hw(sparkling_ts, damped=TRUE, seasonal="multiplicative", h=12), series = 'HW multiplicative dampened forecasts', 
            PI = FALSE) +
  ggtitle("Forecasts for Monthly Sparkling Wine Sales") +
  xlab("Time") + ylab("Volume") +
  guides(colour=guide_legend(title="Forecast"))
##################
#Sweet White
##################
#Plot the fortified data with a trend line
plot(sweetwhite_ts, bty="l", main="Sweet White Wine")
abline(reg=lm(sweetwhite_ts~time(sweetwhite_ts)))

#Create Tables to help determine best model to use for forecasting
kable(sweetw_naive_acc)
kable(sweetw_snaive_acc)
kable(sweetw_drift_acc)
kable(sweetw_holt_acc)
kable(sweetw_dholt_acc)
kable(sweetw_hwA_acc)
kable(sweetw_hwM_acc)
kable(sweetw_dhwM_acc)

#The Holt Winter Multiplicative Dampened forecast will be used
#Look at the predicted value versus the validation data
autoplot(sweetwhite_ts)+
  autolayer(sweetw_dhwM_train_proj)

#Create the forecast for the 1995 year with this method
sweetw_final_forecast <- hw(sweetwhite_ts,damped=TRUE,seasonal="multiplicative", h =12)
kable(sweetw_final_forecast)

autoplot(sweetwhite_ts) +
  autolayer(sweetw_final_forecast, series="HW multiplicative dampened forecasts",PI=FALSE) +
  xlab("Year") +
  ylab("Wine Sales") +
  ggtitle("Sweet White Wine Sales") +
  guides(colour=guide_legend(title="Forecast"))

# Create plot of other forecast methods Naive, Seasonal Naive, Drift, Holt
# Dampened Holt, HW Additive, HW Multiplicative, and HW Multiplicative Dampened
autoplot(sweetwhite_ts) +
  autolayer(naive(sweetwhite_ts, h=12),
            series="Naïve", PI=FALSE) +
  autolayer(snaive(sweetwhite_ts, h=12),
            series="Seasonal naïve", PI=FALSE) +
  autolayer(rwf(sweetwhite_ts, drift=TRUE, h=12),
            series="Drift", PI=FALSE) +
  autolayer(holt(sweetwhite_ts, h=12), series="Holt's method", PI=FALSE) +
  autolayer(holt(sweetwhite_ts, damped=TRUE, phi = 0.9, h=12), series="Damped Holt's method", PI=FALSE) +
  autolayer(hw(sweetwhite_ts,seasonal="additive", h=12), series="HW additive forecasts", PI=FALSE) +
  autolayer(hw(sweetwhite_ts,seasonal="multiplicative", h =12), series="HW multiplicative forecasts",
            PI=FALSE) +
  autolayer(hw(sweetwhite_ts, damped=TRUE, seasonal="multiplicative", h=12), series = 'HW multiplicative dampened forecasts', 
            PI = FALSE) +
  ggtitle("Forecasts for Monthly Sweet White Wine Sales") +
  xlab("Time") + ylab("Volume") +
  guides(colour=guide_legend(title="Forecast"))
##################
#Dry White
##################
#Plot the wine data with a trend line
plot(drywhite_ts, bty="l", main="Dry White Wine")
abline(reg=lm(drywhite_ts~time(drywhite_ts)))

#Create Tables to help determine best model to use for forecasting
kable(dryw_naive_acc)
kable(dryw_snaive_acc)
kable(dryw_drift_acc)
kable(dryw_holt_acc)
kable(dryw_dholt_acc)
kable(dryw_hwA_acc)
kable(dryw_hwM_acc)
kable(dryw_dhwM_acc)

#The Holt Winter Multiplicative forecast will be used
#Look at the predicted value versus the validation data
autoplot(drywhite_ts)+
  autolayer(dryw_hwM_train_proj)

hist(dryw_hwM_residuals, main = "Error in HW Forecast \n on the training set for Rose Wines", xlab="Residuals")

#Create the forecast for the 1995 year with this method
dryw_final_forecast <- hw(drywhite_ts,seasonal="multiplicative", h =12)
kable(dryw_final_forecast)

autoplot(drywhite_ts) +
  autolayer(dryw_final_forecast, series="HW multiplicative forecasts",PI=FALSE) +
  xlab("Year") +
  ylab("Wine Sales") +
  ggtitle("Dry White Wine Sales") +
  guides(colour=guide_legend(title="Forecast"))

# Create plot of other forecast methods Naive, Seasonal Naive, Drift, Holt
# Dampened Holt, HW Additive, HW Multiplicative, and HW Multiplicative Dampened
autoplot(drywhite_ts) +
  autolayer(naive(drywhite_ts, h=12),
            series="Naïve", PI=FALSE) +
  autolayer(snaive(drywhite_ts, h=12),
            series="Seasonal naïve", PI=FALSE) +
  autolayer(rwf(drywhite_ts, drift=TRUE, h=12),
            series="Drift", PI=FALSE) +
  autolayer(holt(drywhite_ts, h=12), series="Holt's method", PI=FALSE) +
  autolayer(holt(drywhite_ts, damped=TRUE, phi = 0.9, h=12), series="Damped Holt's method", PI=FALSE) +
  autolayer(hw(drywhite_ts,seasonal="additive", h=12), series="HW additive forecasts", PI=FALSE) +
  autolayer(hw(drywhite_ts,seasonal="multiplicative", h =12), series="HW multiplicative forecasts",
            PI=FALSE) +
  autolayer(hw(drywhite_ts, damped=TRUE, seasonal="multiplicative", h=12), series = 'HW multiplicative dampened forecasts', 
            PI = FALSE) +
  ggtitle("Forecasts for Monthly Dry White Wine Sales") +
  xlab("Time") + ylab("Volume") +
  guides(colour=guide_legend(title="Forecast"))

