# IMPORT
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_cytoscape as cyto
import dash_daq as daq
import numpy as np
import scipy.stats as stats
import pandas as pd
import pickle
import networkx as nx


## Read Location/Graph/Travel Time/Demand Data

# Grab Station Names, Lat and Long
file = open('mappings.pkl', 'rb')
IDs, name_map, lat_map, lon_map, ridership_sum_map = pickle.load(file)
file.close()

# Grab network Graph
file = open('graph.pkl', 'rb')
G, layout, labels, colors, p = pickle.load(file)
file.close()

# Grab travel times
file = open('travel_times.pkl', 'rb')
travel_times = pickle.load(file)
file.close()

# Grab travel times
file = open('demand.pkl', 'rb')
demand = pickle.load(file)
file.close()


# Convert Graph Keys from int->str
edges = []
for E in G.edges:
    edges += [(str(E[0]),str(E[1]))]
edges

G = nx.Graph(edges)


# Convert Latitude and Longitude to Positions in Graph
pos_x = lon_map[IDs] - lon_map[IDs].min()
pos_x = pos_x/(pos_x.max())
pos_x = pos_x * 1000
pos_y = lat_map[IDs] - lat_map[IDs].min()
pos_y = pos_y/(pos_y.max())
pos_y = pos_y * -1000


## Read Station/Complex Data

df_complexes = pd.read_csv("MTA_Subway_Stations_and_Complexes_20241022.csv")
df_complexes = df_complexes.set_index("Complex ID")[['Stop Name','Display Name','Daytime Routes','Borough','CBD','Structure Type','ADA']]
df_complexes = df_complexes[df_complexes["Borough"] != "SI"]

# Converts ADA status to String
def ADA_to_String(b):
    if b == 1:
        return "ADA"
    else:
        return "Non-ADA"

# Converts Routes to String with once char per route i.e. "NW7"
def routes(s):
    return s.replace(" ","")
    # return tuple(s.split(" "))

# Returns number of routes that go to a station
def Num_Routes(s):
    routes = 1
    for c in s:
        if c == ' ':
            routes += 1
    return routes

# Converts Borough Name from 1-2 Chars to Full Name
def Full_Borough(c):
    if c == "Q":
        return "Queens"
    if c == "M":
        return "Manhattan"
    if c == "Bk":
        return "Brooklyn"
    if c == "Bx":
        return "Bronx"
    return "Unknown"

# Converts CBD value to String
def Full_CBD(b):
    if b:
        return "CBD"
    return "Non-CBD"

# Manipulte Data
df_complexes['ADA'] = df_complexes["ADA"].apply(ADA_to_String)
df_complexes['Routes'] = df_complexes["Daytime Routes"].apply(routes)
df_complexes['number of routes'] = df_complexes["Routes"].apply(len)
df_complexes['Borough'] = df_complexes['Borough'].apply(Full_Borough)
df_complexes['CBD'] = df_complexes['CBD'].apply(Full_CBD)

df_complexes.drop(["Daytime Routes"],axis='columns',inplace=True)

# df_complexes.sort_values(["Number of Routes"])

# name_map = df_complexes["Stop Name"]

# Create dict for getting ID from Name
ID_map = {}
for ID in df_complexes.index:
    ID_map[df_complexes["Display Name"][ID]] = ID

# df_complexes


# Compile all Node and Edge data into one dict for each

node_data = {}
for id in IDs:
    s = str(id)
    node_data[s] = {}
    node_data[s]['name'] = name_map[id]
    node_data[s]['x'] = pos_x[id]
    node_data[s]['y'] = pos_y[id]
    node_data[s]['ridership'] = ridership_sum_map[id]
    node_data[s]['borough'] = df_complexes['Borough'][id]
    node_data[s]['cbd'] = df_complexes['CBD'][id]
    node_data[s]['ada'] = df_complexes['ADA'][id]
    node_data[s]['routes'] = df_complexes['Routes'][id]
    node_data[s]['number of routes'] = df_complexes['number of routes'][id]

edge_data = {}
for (s,t) in G.edges:
    E = (str(s),str(t))
    edge_data[E] = {}
    try:
        edge_data[E]['travel time'] = travel_times[(int(s),int(t))]
    except:
        edge_data[E]['travel time'] = travel_times[(int(t),int(s))]
    try:
        edge_data[E]['demand'] = int(demand[(int(s),int(t))])
    except:
        edge_data[E]['demand'] = int(demand[(int(t),int(s))])



# Helper Function to return sub-Graph based on regex expression
# Ex: "AB|C|~D|E~F" returns all stations with (A and B trains) or (C trains) or (not D trains) or (E and not F trains)
def parse_lines(G_, lines):

    # Deep Copy
    G = G_.copy()
    
    # Clear Spaces
    lines = lines.upper().replace(" ","")

    # Don't filter if odd characters are present
    for c in lines:
        if c not in "ABCDEFGJLMNQRSWZ1234567|~":
            return G
        
    # Undefined behavior
    if ("~|" in lines) or ("~~" in lines):
        return G
    
    # Split into or segements
    or_segments = lines.split("|")

    # Placeholders to OR on after for loop
    keep_pieces = [[] for i in range(len(or_segments))]
    keep_list = [False for i in G.nodes]

    # For every or segment
    for i in range(len(or_segments)):
        # Check each node in graph
        for node in G.nodes:
            include = True
            not_ = False

            # For every char in this or segment
            for c in or_segments[i]:

                # If its a ~ (not)
                if c == '~':
                    # Keep track of not for next char
                    not_ = True
                    continue
                
                # If this char is negated
                if not_:
                    # Exclude all nodes *with* this route
                    if c in node_data[node]['routes']:
                        include = False

                # If this char is not negated (regular)
                else:
                    # Exclude all nodes *without* this route
                    if not c in node_data[node]['routes']:
                        include = False
                
                # Reset to not negated (regular)
                if not_:
                    not_ = False

            # Save pieces from this or segment to include        
            keep_pieces[i] += [include]

    # Or every node for each or segment
    for i in range(len(or_segments)):
        keep_list = np.logical_or(keep_list,keep_pieces[i])
    
    # Remove nodes with False
    node_list = np.array(G.nodes)
    for i in range(len(G.nodes)):
        if not keep_list[i]:
            G.remove_node(node_list[i])

    return G


# Helper function returns edge travel time
def travel_times_fn(O,D,_):
    O = int(O)
    D = int(D)
    if O == D:
        return 0
    try:
        return travel_times[(O,D)]
    except:
        return travel_times[(D,O)]
    

# Function to take a NetworkX Graph and return a Dash Cytoscape Graph
def graph_to_elements(G, node_size, edge_size, reset=False, label_on=True, O=None, O_D=None):

    nodes_and_edges = []

    if O_D != None:
        (s,t) = O_D
        p = nx.shortest_path(G,s,t,weight=travel_times_fn)
        p_e = []
        for i in range(len(p)-1):
            p_e += [(p[i],p[i+1])]

    for node in G.nodes:
        d = {}
        
        d['classes'] = node_data[node]['borough']

        if O != None:
            if node == O:
                d['classes'] = "Path_Start"

        if O_D != None:
            if node in p:
                d['classes'] = "On_Path"
            if node == p[0]:
                d['classes'] = "Path_Start"
            if node == p[-1]:
                d['classes'] = "Path_End"

        # Transfer data for each node (to be displayed)
        d['data'] = {}
        d['data']['id'] = node
        d['data']['name']             = node_data[node]['name']
        d['data']['ridership']        = node_data[node]['ridership']
        d['data']['number of routes'] = node_data[node]['number of routes']

        # Label depending on input
        if label_on:
            d['data']['label']        = node_data[node]['name']
        else:
            d['data']['label']        = ""

        # Node size depends on input
        if(node_size == 'constant'):
            d['data']['size']         = 2
        elif(node_size == 'ridership'):
            d['data']['size']         = node_data[node][node_size] * (12/80000)
        else:
            d['data']['size']         = node_data[node][node_size]

        # Reset to original positions (saved)
        if reset:
            d['position'] = {}
            d['position']['x'] = node_data[node]['x']
            d['position']['y'] = node_data[node]['y']
            
        nodes_and_edges += [d]
    

    for edge in G.edges:

        # Transfer data for each edge (to be displayed)
        d = {}
        d['data'] = {}
        d['data']['source'] = edge[0]
        d['data']['target'] = edge[1]
        d['data']['travel time'] = edge_data[edge]['travel time']
        d['data']['demand']      = edge_data[edge]['demand']

        if O_D != None:
            (s,t) = edge
            if (edge in p_e or (t,s) in p_e):
                d['classes'] = 'On_Path_Edge'

        # Edge size depends on input
        if(edge_size == 'constant'):
            d['data']['weight']  = 110
        elif(edge_size == 'demand'):
            d['data']['weight']  = edge_data[edge][edge_size] * (660/21818124.17040012)
        else:
            d['data']['weight']  = edge_data[edge][edge_size]

        nodes_and_edges += [d]

    return nodes_and_edges

# graph_to_elements(G,'constant','constant',O = '143')


## Graph Statistics

# G.number_of_nodes()
# G.number_of_edges()

def min_max_mean_degree(G):
    d = list(dict(G.degree(list(G.nodes))).values())
    return np.min(d),np.max(d),np.round(np.mean(d),2)

def avg_ridership(G):
    r = []
    for node in G.nodes():
        r += [node_data[node]['ridership']]
    return int(np.round(np.mean(r)))

def avg_num_routes(G):
    r = []
    for node in G.nodes():
        r += [node_data[node]['number of routes']]
    return np.round(np.mean(r),2)

def avg_travel_time(G):
    t = []
    for edge in G.edges():
        t += [edge_data[edge]['travel time']]
    return int(np.round(np.mean(t)))

def avg_demand(G):
    t = []
    for edge in G.edges():
        t += [edge_data[edge]['demand']]
    return int(np.round(np.mean(t)))

# nx.number_connected_components(G)

def num_basic_loops(G):
    return len(nx.cycle_basis(G))

# nx.is_planar(G)

def get_diameter(G):
    try:
        return nx.diameter(G)
    except:
        return "inf"

def get_density(G):
    return str(np.round(nx.density(G)*100,3)) + " %"


## Path Statistics
def path_travel_time(p):
    total_travel_time = 0
    for i in range(len(p)-1):
        try:
            total_travel_time += travel_times[(int(p[i]),int(p[i+1]))]
        except:
            total_travel_time += travel_times[(int(p[i+1]),int(p[i]))]
    return total_travel_time

def path_demand(p):
    total_demand = 0
    for i in range(len(p)-1):
        try:
            total_demand += demand[(int(p[i]),int(p[i+1]))]
        except:
            total_demand += demand[(int(p[i+1]),int(p[i]))]
    return int(np.round(total_demand))


# Returns String for Graph and Path Stats
def graph_stats_string(G,O_D=None):
    if G.number_of_nodes() == 0:
        return "No Nodes in Graph"

    s = ""
    s += "Nodes:\t\t" + str(G.number_of_nodes()) + "\n"
    s += "Edges:\t\t" + str(G.number_of_edges()) + "\n"
    min,max,avg = min_max_mean_degree(G)
    s += "Min Degree:\t" + str(min) + "\n"
    s += "Max Degree:\t" + str(max) + "\n"
    s += "Avg Degree:\t" + str(avg) + "\n"
    s += "Avg Ridership:\t"        + str(avg_ridership(G))   + "\n"
    s += "Avg Num Routes:\t" + str(avg_num_routes(G))  + "\n"
    s += "Avg Travel Time:"        + str(avg_travel_time(G)) + "\n"
    s += "Avg Demand:\t"           + str(avg_demand(G))      + "\n"
    s += "Components:\t"  + str(nx.number_connected_components(G)) + "\n"
    s += "Basic Loops:\t" + str(num_basic_loops(G))                + "\n"
    s += "Planar:\t\t"      + str(nx.is_planar(G))                   + "\n"
    s += "Diameter:\t"    + str(get_diameter(G))                   + "\n"
    s += "Density:\t"     + str(get_density(G))                    + "\n"

    return s

def path_stats_string(G,O_D):
    s = ""
    try:
        (o,d) = O_D
        p = nx.shortest_path(G,o,d,weight=travel_times_fn)

        s += "Length:\t\t"    + str(len(p))              + "\n"
        s += "Travel Time:\t" + str(path_travel_time(p)) + "\n"
        s += "Demand:\t\t"    + str(path_demand(p))      + "\n"
        return s
    except:
        return "No Path Exists"
    

## Graph Stylesheet
cyto_stylesheet = [
            # Group selectors
            {
                'selector': 'node',
                'style': {
                    'width' : "mapData(size,0,12,5,20)",
                    'height' : "mapData(size,0,12,5,20)",
                    'content': "data(label)",
                    'font-size': "6px",
                    'text-valign': "center",
                    'text-halign': "center",
                }
            },
            {
                "selector": 'edge',
                "style": {
                    "curve-style": "haystack",
                    # "haystack-radius": "0.5",
                    "opacity": "0.85",
                    "line-color": "#bbb",
                    "width": "mapData(weight, 0, 660, 1, 8)",
                    "overlay-padding": "3px",
                    # "content": "data(weight)",
                    # "font-size": "8px",
                    # "text-valign": "center",
                    # "text-halign": "center",
                },
            },
            # Class selectors
            {
                'selector': '.On_Path_Edge',
                'style': {
                    'line-color': 'red'
                }
            },
            {
                'selector': '.Path_Start',
                'style': {
                    'background-color': 'green',
                    'line-color': 'green'
                }
            },
            {
                'selector': '.Path_End',
                'style': {
                    'background-color': 'black',
                    'line-color': 'black'
                }
            },
            {
                'selector': '.On_Path',
                'style': {
                    'background-color': 'red',
                    'line-color': 'red'
                }
            },
            {
                'selector': '.Manhattan',
                'style': {
                    'background-color': 'grey',
                    'line-color': 'grey'
                }
            },
            {
                'selector': '.Brooklyn',
                'style': {
                    'background-color': 'orange',
                    'line-color': 'orange'
                }
            },
            {
                'selector': '.Queens',
                'style': {
                    'background-color': 'yellow',
                    'line-color': 'yellow'
                }
            },
            {
                'selector': '.Bronx',
                'style': {
                    'background-color': 'blue',
                    'line-color': 'blue'
                }
            }
        ]


## Define the App
app = Dash(__name__)

app.layout = html.Div(style={"font-family" : "courier new", 'whiteSpace': "pre-wrap",}, children=[

    html.H1(style={"font-weight" : "bold",'textAlign': 'center'},
           children="NYC Subway Network Graph Interactive Vizualization"),

    html.Div([

        html.Div(
            children=[
                html.P('Filter on Routes:',style={"font-weight" : "bold"}),
                html.Div([dcc.Input(id='text-box-routes', type='text',value="",style={'width' : "85%", 'height' : '24px'}),
                          html.P("Use '|'as OR\nUse '~' as NOT\nEx: A|R~M \nEx: 1|2|3|4|5|6|7",
                                 style={'width' : "100%", 'height' : "100px", 'font-style' : "italic", 'color' : "#bbb"})],
                         style={'width' : "100%", 'height' : "130px"}),
                html.P("Select Layout:",style={'height' : "24px","font-weight" : "bold"}),
                html.Div([
                    html.Button('Preset', id='btn-preset',style={'width' : "33%"}),
                    html.Button('Cose',   id='btn-cose'  ,style={'width' : "33%"}),
                    html.Button('Circle', id='btn-circle',style={'width' : "33%"}),
                ],style={'height' : "30px"}),
                html.Div(children=[
                    html.P("Node Size:",style={"font-weight" : "bold"}),
                    dcc.Dropdown(
                                id='dropdown-update-node-size',
                                value='constant',
                                clearable=False,
                                options=[
                                    {'label': name.capitalize(), 'value': name}
                                    for name in ['constant','ridership', 'number of routes']
                                        ]
                                ),
                    html.P("Edge Size:",style={"font-weight" : "bold"}),
                    dcc.Dropdown(
                        id='dropdown-update-edge-size',
                        value='constant',
                        clearable=False,
                        options=[
                            {'label': name.capitalize(), 'value': name}
                            for name in ['constant','travel time', 'demand']
                                ]
                                ),
                    ]),
                    html.Div([html.P("Station Names ",style={'display' : 'inline-block',"font-weight" : "bold"}),
                              dcc.Checklist(id='toggle-labels', options=[""], value=[""],style={'display' : 'inline-block'})]),
            ],
            style = {
                'height' : "600px",
                'margin-left' : "10px",
                'width' : "20%",
                'display' : "inline-block",
                'vertical-align' : "top"
            }
        ),


        html.Div(
            cyto.Cytoscape(
                id='cytoscape',
                elements=graph_to_elements(G,"constant","constant",reset=True),
                layout={'name': 'preset', "animate": True, "animationDuration": 1000},
                autoRefreshLayout=False,
                style={'width': '100%', 'height': '600px'},
                stylesheet=cyto_stylesheet
            ),
            style={
                'height' : "600px",
                'margin-left' : "10px",
                'width' : "50%",
                'display' : 'inline-block',
                'vertical-align' : "top"
                }
        ),


        html.Div(
            children=[
                html.P(id='cytoscape-selectedNodes',children="Click Node to Select Path Origin", 
                       style={'height' : "150px"}),
                html.P("Graph Statistics:", style={'font-weight' : 'bold', 'height' : '2px'}),
                html.P(id='cytoscape-graphData',children=graph_stats_string(parse_lines(G,""))),
                html.P("Mouseover Data:", style={'font-weight' : 'bold', 'height' : '2px'}),
                html.P(id='cytoscape-mouseoverNodeData',children=""),
                html.P(id='cytoscape-mouseoverEdgeData',children=""),
            ],
            style={
                'height' : "600px",
                'margin-left' : "10px",
                'width' : "25%",
                'display' : "inline-block",
                'vertical-align' : "top",
                'font-size' : "14px"
                }
        ),
    ])
])


## Callbacks
@app.callback(Output('cytoscape-mouseoverEdgeData', 'children'),
              Input('cytoscape', 'mouseoverEdgeData')
)
def mouse_over(data):
    if data == None:
        return
    data_string = "O: " + name_map[int(data['source'])] + "\nD: " + name_map[int(data['target'])] + "\n"
    data_string += "Demand: " + str(int(data['demand'])) + "\n"
    data_string += "Travel Time: " + str(data['travel time'])
    return data_string

@app.callback(Output('cytoscape-mouseoverNodeData', 'children'),
              Input('cytoscape', 'mouseoverNodeData')
)
def mouse_over(data):
    if data == None:
        return
    data_string = data['name'] + "\n"
    data_string += "Ridership: " + str(int(data['ridership'] * (80000/12))) + "\n"
    data_string += "Number of Routes: " + str(data['number of routes'])
    return data_string

@app.callback(
    [Output('cytoscape',               'layout'  ),
     Output('cytoscape',               'elements'),
     Output('btn-preset',              'n_clicks'),
     Output('btn-cose',                'n_clicks'),
     Output('btn-circle',              'n_clicks'),
     Output('cytoscape-graphData',     'children'),
     Output('cytoscape-selectedNodes', 'children')],

    Input('dropdown-update-node-size', 'value'           ),
    Input('dropdown-update-edge-size', 'value'           ),
    Input('text-box-routes',           'value'           ),
    Input('btn-preset',                'n_clicks'        ),
    Input('btn-cose',                  'n_clicks'        ),
    Input('btn-circle',                'n_clicks'        ),
    Input('toggle-labels',             'value'           ),
    Input('cytoscape',                 'selectedNodeData'),
    State('cytoscape',                 'layout'          ),
    State('cytoscape',                 'elements'        ),
    State('cytoscape-graphData',       'children'        )
    
)
def update_elements(node_size_factor,edge_size_factor,
                    lines,
                    preset_click,cose_click,circle_click,
                    label_on,
                    selected,
                    layout,elements,graph_stats
                   ):
    
    O_D = None
    O = None

    if selected == None or selected == []:
        selected_string = "Click Node to Select Path Origin"
    elif len(selected) == 1:
        selected_string = "Origin: " + "\n" + selected[0]['name'] + "\n\n" + "Ctrl Click Node to Select Destination"
        O = selected[0]['id']
    elif len(selected) > 1:
        selected_string = "Origin: " + "\n" + selected[0]['name'] + "\n" + "Destination: " + "\n" + selected[1]['name']
        O_D = (selected[0]['id'],selected[-1]['id'])
    
    if O_D != None:
        selected_string += "\n" + path_stats_string(parse_lines(G,lines),O_D)


    if preset_click != None:
        return [
                {'name': 'preset', "animate": True, "animationDuration": 1000},
                graph_to_elements(parse_lines(G,lines),node_size_factor,edge_size_factor,reset=True,label_on=label_on), 
                None,None,None,
                graph_stats_string(parse_lines(G,lines)),
                selected_string
               ]

    if cose_click != None:
        return [{'name': 'cose', "animate": True, "animationDuration": 1000},elements,None,None,None,graph_stats,selected_string]
    
    if circle_click != None:
        return [{'name': 'circle', "animate": True, "animationDuration": 1000},elements,None,None,None,graph_stats,selected_string]
    
    try:
        return [
                layout,
                graph_to_elements(parse_lines(G,lines),node_size_factor,edge_size_factor,label_on=label_on,O=O,O_D=O_D),
                None,None,None,
                graph_stats_string(parse_lines(G,lines),O_D=O_D),
                selected_string
               ]
    except:
        return [
                layout,
                graph_to_elements(parse_lines(G,lines),node_size_factor,edge_size_factor,label_on=label_on),
                None,None,None,
                graph_stats_string(parse_lines(G,lines)),
                selected_string
               ]
    

# Run Server
app.run_server(debug=False)