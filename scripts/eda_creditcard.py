import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

from scipy.stats import chi2_contingency, f_oneway


# Credit card data analysis
def plot_random_forest_feature_importance(df):
    """
    Plots the Random Forest Feature Importance .
    
    Parameters:
    dataset_path (str): The file path of the dataset.
    """
  

    # Feature selection
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train the Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    # Get the feature importances
    feature_importances = rf.feature_importances_
    feature_names = X.columns

    # Create the Plotly plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_names,
        y=feature_importances,
        text=feature_importances,
        textposition='outside',
        marker_color='#6699cc'
    ))

    fig.update_layout(
        title='Random Forest Feature Importance',
        xaxis_title='Feature',
        yaxis_title='Importance',
        bargap=0.1
    )

    fig.show()




def plot_feature_interactions(df):
    """
    Plots the Feature Interactions.
    
    Parameters:
    dataset_path (str): The file path of the dataset.
    """


    # Feature selection
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Calculate the correlation matrix
    corr_matrix = X.corr()

    # Find the top 4 feature interactions
    top_interactions = corr_matrix.abs().unstack().sort_values(ascending=False)[:4]

    # Create the Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        x=X.columns,
        y=X.columns,
        z=corr_matrix,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title="Correlation",
            tickvals=[-1, 0, 1],
            ticktext=["Negative", "No Correlation", "Positive"]
        )
    ))

    # Add scatter plots for the top 4 feature interactions
    for (feature1, feature2), corr in top_interactions.items():
        fig.add_trace(go.Scatter(
            x=X[feature1],
            y=X[feature2],
            mode='markers',
            marker=dict(
                color=corr,
                colorscale='RdBu',
                size=10,
                colorbar=dict(
                    title="Correlation"
                )
            ),
            name=f"{feature1} vs {feature2} (Correlation: {corr:.2f})"
        ))

    fig.update_layout(
        title='Feature Interactions',
        xaxis_title='Feature',
        yaxis_title='Feature',
        xaxis_tickangle=-45
    )

    fig.show()