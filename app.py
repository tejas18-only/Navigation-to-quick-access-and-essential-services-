from flask import Flask, render_template, request
import osmnx as ox
import networkx as nx
import folium
from geopy.distance import geodesic
from shapely import wkt
import pandas as pd
import heapq

app = Flask(__name__)

# ------------------------------
# Load data from CSV files
# ------------------------------
hospital_file = 'pune_hospitals.csv'
police_file = 'pune_policestation.csv'
highways_file = 'pune_highways.csv'  # if needed for additional filtering

# Load hospital data and filter emergency hospitals
hospital_data = pd.read_csv(hospital_file)
hospital_data['geometry'] = hospital_data['geometry'].apply(wkt.loads)
emergency_hospitals = hospital_data[hospital_data['emergency'] == 'yes'].copy()
# Extract latitude and longitude from geometry
emergency_hospitals['lat'] = emergency_hospitals['geometry'].apply(lambda x: x.centroid.y if x.geom_type == 'Polygon' else x.y)
emergency_hospitals['lon'] = emergency_hospitals['geometry'].apply(lambda x: x.centroid.x if x.geom_type == 'Polygon' else x.x)
if 'name' not in emergency_hospitals.columns:
    emergency_hospitals['name'] = 'Hospital'

# Load police station data
police_data = pd.read_csv(police_file)
police_data['geometry'] = police_data['geometry'].apply(wkt.loads)
police_data['lat'] = police_data['geometry'].apply(lambda x: x.centroid.y if x.geom_type == 'Polygon' else x.y)
police_data['lon'] = police_data['geometry'].apply(lambda x: x.centroid.x if x.geom_type == 'Polygon' else x.x)
if 'name' not in police_data.columns:
    police_data['name'] = 'Police Station'

# ------------------------------
# Predefined locations in Pune
# ------------------------------
def get_user_locations():
    locations = {
        1: ('Shivajinagar', (18.5300, 73.8476)),
        2: ('Hadapsar', (18.5089, 73.9259)),
        3: ('Kothrud', (18.5074, 73.8077)),
        4: ('Aundh', (18.5590, 73.8078)),
        5: ('Viman Nagar', (18.5679, 73.9143)),
        6: ('Baner', (18.5603, 73.7890)),
        7: ('Wakad', (18.5971, 73.7707)),
        8: ('Magarpatta', (18.5139, 73.9317)),
        9: ('Pimpri', (18.6298, 73.7997)),
        10: ('Chinchwad', (18.6446, 73.7639))
        # Add more locations as needed...
    }
    return locations

# ------------------------------
# Heuristic function (using geodesic distance)
# ------------------------------
def heuristic(node, goal):
    return geodesic((node[1], node[0]), (goal[1], goal[0])).meters

# ------------------------------
# A* algorithm using the road network
# ------------------------------
def a_star(graph, start_node, goal_node):
    pq = []  # Priority queue: (priority, node)
    heapq.heappush(pq, (0, start_node))
    came_from = {start_node: None}
    cost_so_far = {start_node: 0}
    
    while pq:
        current_cost, current_node = heapq.heappop(pq)
        if current_node == goal_node:
            break
        for neighbor in graph.neighbors(current_node):
            road_length = graph.edges[current_node, neighbor, 0]['length']
            new_cost = cost_so_far[current_node] + road_length
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(
                    (graph.nodes[neighbor]['x'], graph.nodes[neighbor]['y']),
                    (graph.nodes[goal_node]['x'], graph.nodes[goal_node]['y'])
                )
                heapq.heappush(pq, (priority, neighbor))
                came_from[neighbor] = current_node
                
    # Reconstruct the path from start to goal
    path = []
    current = goal_node
    while current != start_node:
        path.append(current)
        current = came_from[current]
    path.append(start_node)
    path.reverse()
    return path

# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/')
def index():
    locations = get_user_locations()
    return render_template('index.html', locations=locations)

@app.route('/get_path', methods=['POST'])
def get_path():
    selected_location_key = int(request.form['location'])
    locations = get_user_locations()
    selected_location = locations[selected_location_key][1]

    # Fetch the road network for driving in Pune
    place_name = "Pune, India"
    graph = ox.graph_from_place(place_name, network_type='drive')

    # --- Nearest Emergency Hospital Route ---
    # Compute distances from selected location to each emergency hospital
    emergency_hospitals['distance'] = emergency_hospitals.apply(
        lambda row: geodesic(selected_location, (row['lat'], row['lon'])).meters, axis=1
    )
    nearest_hospital = emergency_hospitals.loc[emergency_hospitals['distance'].idxmin()]
    nearest_hospital_location = (nearest_hospital['lat'], nearest_hospital['lon'])
    distance_to_hospital = nearest_hospital['distance']

    # Get nearest nodes on the graph for start and hospital
    start_node = ox.distance.nearest_nodes(graph, selected_location[1], selected_location[0])
    hospital_goal_node = ox.distance.nearest_nodes(graph, nearest_hospital_location[1], nearest_hospital_location[0])
    shortest_path_hospital = a_star(graph, start_node, hospital_goal_node)

    # --- Nearest Police Station Route ---
    police_data['distance'] = police_data.apply(
        lambda row: geodesic(selected_location, (row['lat'], row['lon'])).meters, axis=1
    )
    nearest_police = police_data.loc[police_data['distance'].idxmin()]
    nearest_police_location = (nearest_police['lat'], nearest_police['lon'])
    distance_to_police = nearest_police['distance']
    police_goal_node = ox.distance.nearest_nodes(graph, nearest_police_location[1], nearest_police_location[0])
    shortest_path_police = a_star(graph, start_node, police_goal_node)

    # --- Create a Folium Map ---
    map_pune = folium.Map(location=[selected_location[0], selected_location[1]], zoom_start=12)

    # Plot all emergency hospitals on the map
    for _, hospital in emergency_hospitals.iterrows():
        folium.Marker([hospital['lat'], hospital['lon']],
                      popup=hospital['name'],
                      icon=folium.Icon(color='red', icon='info-sign')).add_to(map_pune)

    # Optionally, plot all police stations on the map
    for _, police in police_data.iterrows():
        folium.Marker([police['lat'], police['lon']],
                      popup=police['name'],
                      icon=folium.Icon(color='green', icon='shield', prefix='fa')).add_to(map_pune)

    # Plot the shortest path to the emergency hospital (blue polyline)
    route_coords_hospital = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in shortest_path_hospital]
    folium.PolyLine(route_coords_hospital, color='blue', weight=5, opacity=0.8).add_to(map_pune)

    # Plot the shortest path to the police station (orange polyline)
    route_coords_police = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in shortest_path_police]
    folium.PolyLine(route_coords_police, color='orange', weight=5, opacity=0.8).add_to(map_pune)

    # Convert the map to HTML for embedding in the template
    map_html = map_pune._repr_html_()

    return render_template('index.html', 
                           locations=locations, 
                           map_html=map_html, 
                           hospital_name=nearest_hospital['name'],
                           hospital_distance=round(distance_to_hospital, 2),
                           police_name=nearest_police['name'],
                           police_distance=round(distance_to_police, 2))

if __name__ == "__main__":
    app.run(debug=True)
