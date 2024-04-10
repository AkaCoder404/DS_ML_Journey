# IBM Data Science Professional Certificate
# Applied Data Science Capstone
# Build a Dashboard Application with Plotly Dash

# Tasks
# Task 1: Add a Launch Site Drop-down Input Component
# Task 2: Add a Callback Function to Render 'success-pie-chart' based on selected site dropdown
# Task 3: Add a Range Slider to Select Payload
# Task 4: Add a Callback Function to Render 'success-payload-scatter-chart' scatter plot

# Import required libraries
import pandas as pd
import dash
# import dash_html_components as html
# import dash_core_components as dcc
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()
print(spacex_df.head())
print(spacex_df.columns)

launch_sites = spacex_df['Launch Site'].unique()
launch_sites = [{'label': site, 'value': site} for site in launch_sites]
launch_sites.insert(0, {'label': 'All Sites', 'value': 'ALL'})
print(launch_sites)


# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                # dcc.Dropdown(id='site-dropdown',...)
                                dcc.Dropdown(
                                    id='site-dropdown',
                                    options=launch_sites,
                                    value='ALL', 
                                    placeholder='Select a Launch Site here', 
                                    searchable=True),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                #dcc.RangeSlider(id='payload-slider',...)
                                dcc.RangeSlider(id='payload-slider', min=0, max=10000, step=1000, value=[min_payload, max_payload]),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'), 
    Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(entered_site):
    print(entered_site)
    if entered_site == 'ALL':
        filtered_df = spacex_df
        fig = px.pie(spacex_df, values='class',
                     names = 'Launch Site',
                     title = 'Total Success Launches By Site')
        return fig
    else:
        # return the outcomes piechart for a selected site     
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        success = sum(filtered_df['class'] == 1)
        failure = sum(filtered_df['class'] == 0)
        print(filtered_df.head())
        # plot the percentage of successful launches for the selected site
        fig = px.pie(filtered_df, 
                    values=[success, failure],
                     names=['Success', 'Failure'],
                     title='Total Success Launches for Site ' + entered_site,
                    )
        
        return fig
        

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'),
              Input(component_id='payload-slider', component_property='value'))
def get_scatter_chart(entered_site, payload_range):
    print(entered_site)
    print(payload_range)
    if entered_site == 'ALL':
        filtered_df = spacex_df
        fig = px.scatter(
            filtered_df, 
            x='Payload Mass (kg)', 
            y='class', 
            color='Booster Version Category',
            title="Correlation between Payload and Success for all Sites",
            range_x=[payload_range[0], payload_range[1]]
            )
        return fig
    else:
        # return the outcomes piechart for a selected site     
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        fig = px.scatter(
            filtered_df, 
            x='Payload Mass (kg)', 
            y='class', 
            color='Booster Version Category',
            title='Correlation between Payload and Success for Site ' + entered_site,
            range_x=[payload_range[0], payload_range[1]]
            )
        return fig

# Run the app
if __name__ == '__main__':
    app.run_server()
