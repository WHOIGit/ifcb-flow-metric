import os
import re
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from time import time, sleep
import requests
import argparse

DASHBOARD_BASE_URL = os.getenv('DASHBOARD_BASE_URL', 'http://localhost:8000')

def dashboard_link(pid):
    return f'{DASHBOARD_BASE_URL}/bin?bin={pid}'

def load_point_cloud(pid):
    """Load point cloud data for a given PID."""
    URL=f'{DASHBOARD_BASE_URL}/api/plot/{pid}'
    response = requests.get(URL)
    if response.status_code != 200:
        return np.random.rand(1,2)
    else:
        j = response.json()
        return np.dstack((j['roi_x'], j['roi_y']))[0]

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
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        )
    )
    return fig

def pid_to_datetime(pid):
    if pid.startswith('D'):
        return pd.to_datetime(pid[1:16], format='%Y%m%dT%H%M%S')
    elif pid.startswith('I'):
        ts = re.sub(r'^IFCB\d_', '', pid)
        return pd.to_datetime(ts[0:15], format='%Y_%j_%H%M%S')

def load_data(file_path, month=None):
    print(f'loading data from {file_path}')
    data = pd.read_csv(file_path)
    dates = [pid_to_datetime(pid) for pid in data['pid']]
    df = pd.DataFrame({
        'timestamp': dates,
        'anomaly_score': data['anomaly_score'],
        'pid': data['pid']
    })
    
    if month:
        year = int(month[:4])
        month_num = int(month[4:])
        df = df[
            (df['timestamp'].dt.year == year) & 
            (df['timestamp'].dt.month == month_num)
        ]
    print(f'loaded {len(df)} records')
    return df

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
    fig = px.scatter(
        df, 
        x='timestamp', 
        y='anomaly_score',
        title='Anomaly Scores Over Time'
    )
    fig.update_layout(
        hovermode='x',
        hoverlabel=dict(bgcolor="white"),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(fixedrange=True),
    )
    return fig

@app.callback(
    [Output('last-hover-time', 'data'),
     Output('selected-pid', 'children'),
     Output('2d-point-cloud', 'figure'),
     Output('point-cloud-details', 'children')],
    [Input('anomaly-time-series', 'clickData'),
     Input('last-hover-time', 'data')],
)
def update_on_hover(hover_data, last_hover):
    if hover_data is None:
        return time(), '', {}, None

    current_time = time()
    # if current_time - last_hover < 0.5:
    #    raise dash.exceptions.PreventUpdate

    try:
        timestamp = pd.Timestamp(hover_data['points'][0]['x'])
        selected_point = df.iloc[(df['timestamp'] - timestamp).abs().argsort()[:1]]
        
        if not selected_point.empty:
            pid = selected_point['pid'].iloc[0]
            score = selected_point['anomaly_score'].iloc[0]
            points = load_point_cloud(pid)
            figure = plot_2d_point_cloud(points)
            details = html.P([
                html.Strong("PID: "), 
                html.A(pid, href=dashboard_link(pid), target="_blank"), 
                html.Br(),
                html.Strong("Anomaly Score: "), f"{score:.4f}"
            ])
            return current_time, pid, figure, details
                
    except Exception as e:
        print(f"Error updating visualization: {e}")
        
    return current_time, '', {}, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Point Cloud Anomaly Explorer')
    parser.add_argument('--file', default='/Users/jfutrelle/Data/flow-scores/mvco_scores_nonan.csv',
                        help='Path to the CSV file containing anomaly scores')
    parser.add_argument('--month', help='Month to display in YYYYMM format (e.g., 202401)')
    parser.add_argument('--decimate', type=int, default=10, help='Decimation factor for time series plot')
    args = parser.parse_args()
    
    df = load_data(args.file, args.month)
    if len(df) == 0:
        print(f"No data found for month {args.month}")
        exit(1)
    if args.decimate > 1:
        df = df.iloc[::args.decimate, :]
        print(f'Decimated data to {len(df)} records')
        
    app.run_server(debug=True)