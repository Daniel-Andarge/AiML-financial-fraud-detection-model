import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import lime

def explain_with_shap(model, X_test, feature_name):
    """
    Explain a model's predictions using SHAP.
    
    Args:
        model (object): The trained machine learning model.
        X_test (pandas.DataFrame): The input data to explain.
    """
 
    X_sample = X_test[:100]

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

     # Summary plot
    shap.summary_plot(shap_values, X_sample) 
        # Force plot
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_sample.iloc[0,:], matplotlib=True)
    plt.show()
    # Dependence plot
    shap.dependence_plot(feature_name, shap_values, X_sample)  


def explain_with_lime(model, X_test, feature_names=None):
    """
    Explain a model's predictions using LIME.

    Args:
        model (object): The trained machine learning model.
        X_test (pandas.DataFrame): The input data to explain.
        feature_names (list): Optional - Names of features in X_test.

    Returns:
        matplotlib.pyplot.figure: A plot showing LIME feature importances.
    """
    if feature_names is None:
        feature_names = X_test.columns.tolist()

    # Get a sample of data
    X_sample = X_test.iloc[:1]

    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_test.values,
                                                       feature_names=feature_names,
                                                       class_names=['Not Fraud', 'Fraud'],
                                                       mode='classification')

    # Explain a prediction
    exp = explainer.explain_instance(X_sample.values[0], model.predict_proba, num_features=len(feature_names))

    # Plot the LIME feature importance
    fig = exp.as_pyplot_figure()
    plt.title('LIME Feature Importance')
    plt.xlabel('Impact on Prediction')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

