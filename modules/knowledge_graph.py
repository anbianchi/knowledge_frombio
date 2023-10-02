import networkx as nx
import os
from pyvis.network import Network

def construct_knowledge_graph(report, relations, color):
    
    # Create a new knowledge graph
    G = nx.Graph()
    
    # Add nodes
    for entity in report[0]["entities"]:
        G.add_node(entity["entity_id"], label=entity["entity"], color=color)
    
    # Add edges
    for relation in relations:
        if relation["entity1_id"] != relation["entity2_id"]:
            G.add_edge(relation["entity1_id"], relation["entity2_id"], label=relation["relation_type"], color=color)
        #else:
        #    print(f"Skipped adding self-relationship for entity: {relation['entity1_id']}")
    return G

def visualize_knowledge_graph(G, filename, bitext=False):
    
    if bitext:
        output_dir = 'merged_outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = 'temp_outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    if filename is None:
        new_filename = 'merged_kg'
    else:
        # Extract the patient number from the filename
        patient_number = filename.split('_')[0].replace('#', '')
        # Construct the new filename
        new_filename = f"#{patient_number}_kg"
        
    # Visualize the knowledge graph
    nt = Network(height='1000px', width='100%', font_color='black', directed=False, notebook=False)
    
    for node, data in G.nodes(data=True):
        nt.add_node(node, label=data['label'], color=data['color'])
    for source, target, data in G.edges(data=True):
        nt.add_edge(source, target, title=data['label'])
    #html = nt.show('notebook.html')
    #display(HTML(html)) 
    
    # Apply force atlas 2-based layout
    nt.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)
    
    # Save the graph to an HTML file
    nt.show_buttons(filter_=['physics'])
        
    # Save the graph to an HTML file
    nt.save_graph(os.path.join(output_dir, f'{new_filename}.html'))
    
    return nt

