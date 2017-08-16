from sklearn.externals import joblib

model = joblib.load('trained_model.pkl')

model_to_result = [
   # ATTRIBUTES
]

models_to_result = [
    model_to_result
]

# Run the model and make a prediction
predicted_result = model.predict(models_to_result)
predicted_value = predicted_result[0]

print("The estimated result: ${:,.2f}".format(predicted_value))

