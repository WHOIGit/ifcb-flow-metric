import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from time import time, sleep

def load_point_cloud(pid):
    """Load point cloud data for a given PID."""
    sleep(0.5)  # Simulate loading time
    return np.random.rand(100, 2)

def plot_2d_point_cloud(points):
    """Create a scatter plot of a 2D point cloud."""
    fig = px.scatter(
        x=points[:, 0], 
        y=points[:, 1],
        title="2D Point Cloud",
        labels={'x': 'X (width)', 'y': 'Y (height)'}
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Generate sample data
dates = pd.date_range("2023-01-01", "2023-01-31", periods=100)
df = pd.DataFrame({
    'timestamp': dates,
    'anomaly_score': np.random.uniform(0, 1, size=100),
    'pid': [f'sample_{i:04d}' for i in range(100)]
})

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Interactive Point Cloud Anomaly Explorer'),
    html.Hr(),
    
    html.Div([
        html.H3('Anomaly Score Timeline'),
        dcc.Graph(
            id='anomaly-time-series',
            config={'scrollZoom': True}
        ),
        
        html.Hr(),
        
        html.H3('Selected Point Cloud'),
        html.Div(id='point-cloud-details'),
        html.Div([
            html.Label('PID: ', style={'fontWeight': 'bold'}),
            html.Span(id='selected-pid')
        ]),
        dcc.Graph(id='2d-point-cloud'),
        dcc.Store(id='last-hover-time', data=time())
    ], style={'padding': '20px'})
])

@app.callback(
    Output('anomaly-time-series', 'figure'),
    Input('anomaly-time-series', 'id')
)
def update_timeline(_):
    fig = px.line(
        df, 
        x='timestamp', 
        y='anomaly_score',
        title='Anomaly Scores Over Time'
    )
    fig.update_layout(
        hovermode='x',
        hoverlabel=dict(bgcolor="white"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

@app.callback(
    [Output('last-hover-time', 'data'),
     Output('selected-pid', 'children'),
     Output('2d-point-cloud', 'figure'),
     Output('point-cloud-details', 'children')],
    [Input('anomaly-time-series', 'hoverData'),
     Input('last-hover-time', 'data')],
    prevent_initial_call=True
)
def update_on_hover(hover_data, last_hover):
    if hover_data is None:
        return time(), '', {}, None

    current_time = time()
    if current_time - last_hover < 0.5:
        raise dash.exceptions.PreventUpdate

    try:
        timestamp = pd.Timestamp(hover_data['points'][0]['x'])
        selected_point = df.iloc[(df['timestamp'] - timestamp).abs().argsort()[:1]]
        
        if not selected_point.empty:
            pid = selected_point['pid'].iloc[0]
            score = selected_point['anomaly_score'].iloc[0]
            points = load_point_cloud(pid)
            figure = plot_2d_point_cloud(points)
            details = html.P([
                html.Strong("Anomaly Score: "), f"{score:.4f}"
            ])
            return current_time, pid, figure, details
                
    except Exception as e:
        print(f"Error updating visualization: {e}")
        
    return current_time, '', {}, None

if __name__ == '__main__':
    app.run_server(debug=True)