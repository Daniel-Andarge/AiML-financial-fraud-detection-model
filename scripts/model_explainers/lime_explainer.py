from lime import lime_tabular

def explain_with_lime(model, X_test):
    """
    Explain a model's predictions using LIME.
    
    Args:
        model (object): The trained machine learning model.
        X_test (pandas.DataFrame): The input data to explain.
    """
    # Get a sample of data 
    X_sample = X_test[:1]  

    # Create a LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(X_test, feature_names=X_test.columns)

    # Explain a prediction
    exp = explainer.explain_instance(X_sample.iloc[0], model.predict_proba)

    # Plot the LIME feature importance
    exp.as_pyplot_figure()