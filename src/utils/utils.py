def evaluate_models(    X_train,    y_train,   X_test,    y_test,    models):
    report = {}
	for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
	r2 = r2_score(y_test, predictions)

	mae = mean_absolute_error(y_test, predictions)

	rmse = np.sqrt(
		mean_squared_error(y_test, predictions))
		report[name] = {"R2": r2, "MAE": mae, "RMSE": rmse	
	
	}			
	return report