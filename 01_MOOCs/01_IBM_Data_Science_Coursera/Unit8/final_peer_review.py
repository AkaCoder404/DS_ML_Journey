#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Load the data using pandas
data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv')
print(data.head())

# Initialize the Dash app
app = dash.Dash(__name__)

# Set the title of the dashboard
#app.title = "Automobile Statistics Dashboard"

#---------------------------------------------------------------------------------
# Create the dropdown menu options
dropdown_options = [
    {'label': '...........', 'value': 'Yearly Statistics'},
    {'label': 'Recession Period Statistics', 'value': '.........'}
]
# List of years 
year_list = [i for i in range(1980, 2024, 1)]
#---------------------------------------------------------------------------------------
# Create the layout of the app
app.layout = html.Div([
    #TASK 2.1 Add title to the dashboard
    html.H1("Automobile Sales Statistics Dashboard", style={'textAlign': 'center', 'color': '#503D36', 'fontSize': 24}),#May include style for title
    html.Div([#TASK 2.2: Add two dropdown menus
        html.Label("Select Statistics:"),
        dcc.Dropdown(
            id='dropdown-statistics',
            options=[
                { 'label': 'Yearly Statistics', 'value': 'Yearly Statistics'},
                { 'label': 'Recession Period Statistics', 'value': 'Recession Period Statistics'}
                ],
            value='Select Statistics',
            placeholder='Select a report type'
        )
    ]),
    html.Div(dcc.Dropdown(
            id='select-year',
            options=[{'label': i, 'value': i} for i in year_list],
            value='1980'
        )),
    html.Div([#TASK 2.3: Add a division for output display
    html.Div(id='output-container', className='chart-grid', style={'display': 'flex'}),])
])

#TASK 2.4: Creating Callbacks
# Define the callback function to update the input container based on the selected statistics
@app.callback(
    Output(component_id='select-year', component_property='disabled'),
    Input(component_id='dropdown-statistics',component_property='value'))

def update_input_container(selected_statistics):
    print("Callback 1", selected_statistics)
    if selected_statistics == 'Yearly Statistics': 
        print("Yearly Statistics Selected")
        return False
    else: 
        print("Recession Period Statistics Selected")
        return True

#Callback for plotting
# Define the callback function to update the input container based on the selected statistics
@app.callback(
    Output(component_id='output-container', component_property='children'),
    [Input(component_id='select-year', component_property='value'), Input(component_id='dropdown-statistics', component_property='value')])


def update_output_container(selected_year, selected_statistics):
    print("Callback 2", selected_year, selected_statistics)
    print()
    if selected_statistics == 'Recession Period Statistics':
        # Filter the data for recession periods
        recession_data = data[data['Recession'] == 1]
        
# #TASK 2.5: Creating Graphs

#Plot 1 Automobile sales fluctuate over Recession Period (year wise)
        # use groupby to create relevant data for plotting
        yearly_rec = recession_data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        R_chart1 = dcc.Graph(
            figure=px.line(yearly_rec, 
                x='Year',
                y='Automobile_Sales',
                title="Average Automobile Sales fluctuation over Recession Period"))

# #Plot 2 Calculate the average number of vehicles sold by vehicle type       
        # use groupby to create relevant data for plotting
        average_sales = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()                           
        R_chart2 = dcc.Graph(
            figure=px.pie(average_sales, 
                values='Automobile_Sales', 
                names='Vehicle_Type', 
                title='Average number of vehicles sold by vehicle type'))
        
# # Plot 3 Pie chart for total expenditure share by vehicle type during recessions
#         # use groupby to create relevant data for plotting
        exp_rec = recession_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        R_chart3 = dcc.Graph(
            figure=px.pie(exp_rec, 
                values='Advertising_Expenditure', 
                names='Vehicle_Type', 
                title='Total expenditure share by vehicle type during recessions'))
        

# # Plot 4 bar chart for the effect of unemployment rate on vehicle type and sales
        unemp_rec = recession_data.groupby(['Vehicle_Type', 'unemployment_rate'])['Automobile_Sales'].mean().reset_index()
        R_chart4 = dcc.Graph(
            figure=px.bar(unemp_rec, 
                x='Vehicle_Type', 
                y='Automobile_Sales',
                color='Vehicle_Type',
                title='Effect of unemployment rate on vehicle type and sales'))

# #TASK 2.6: Returning the graphs for display
        return [
            html.Div(className='chart-grid', children=[
                html.Div(className='chart-item-row', children=[
                    html.Div(children=R_chart1,),
                    html.Div(children=R_chart2,),
                ], style={'width': '100%', 'display': 'flex'}),
                html.Div(className='chart-item-row', children=[
                    html.Div(children=R_chart3, style={'width': '50%'}),
                    html.Div(children=R_chart4, style={'width': '50%'}),
                ], style={'display': 'flex'}),
            ])
        ]

#  # Yearly Statistic Repot Plots                             
    elif (selected_year and selected_statistics=='Yearly Statistics') :
        yearly_data = data[data['Year'] == selected_year]
                              

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)


# %%
