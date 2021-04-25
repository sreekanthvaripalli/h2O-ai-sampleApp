import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator

h2o.init()

airlines_train_data = h2o.import_file("https://s3.amazonaws.com/h2o-airlines-unpacked/allyears2k.csv")

print(airlines_train_data)

print("--------------------------")

gbm_model = H2OGradientBoostingEstimator()
gbm_model.train(x=["Month", "DayOfWeek", "Distance"], y="IsArrDelayed", training_frame=airlines_train_data)
print(gbm_model)
prediction = gbm_model.predict(airlines_train_data)
print("prediction of gbm : \n", prediction)

xgb_model = H2OXGBoostEstimator()
xgb_model.train(x=["Month", "DayOfWeek", "Distance"], y="IsArrDelayed", training_frame=airlines_train_data)
print(xgb_model)
xgb_prediction = xgb_model.predict(airlines_train_data)
print("prediction of xgb : \n", xgb_prediction)
