from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("catboost_employee_attrition_model.cbm")

print("Model expects these features in order:")
print(model.feature_names_)
