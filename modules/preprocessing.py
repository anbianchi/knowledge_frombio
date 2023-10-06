import re
import os
from collections import Counter
import networkx as nx
from modules.relation_extraction import load_Bio_ClinicalBERT_model, extract_relations
from modules.entity_extraction import extract_entities_bern2
from modules.knowledge_graph import construct_knowledge_graph, visualize_knowledge_graph

def clear_reports_folder():
    # Remove existing files in the 'diagnostic_reports' folder
    for file in os.listdir('diagnostic_reports'):
        os.remove(os.path.join('diagnostic_reports', file))
        
def preprocess_medical_knowledge():
    # Load the medical knowledge as a list of strings
    medical_knowledge = []
    preprocessed_reports = []
    files = os.listdir("diagnostic_reports")
    if not files:
        return [], None
    for filename in files:
        with open(os.path.join("diagnostic_reports", filename), "r") as f:
            text = f.read()
            # Pre-process the medical knowledge by lowercasing and removing special characters
            text = re.sub(r'[^a-zA-Z0-9.\s]', '', text).lower()
            medical_knowledge.append(text)
            preprocessed_reports.append(text)
    
    #print("preprocessed reports: ", preprocessed_reports, "\n\n")
    filename = os.path.basename(filename)
    return preprocessed_reports, filename

def process_reports(preprocessed_reports, filename):
    re_model, re_tokenizer = load_Bio_ClinicalBERT_model()

    #preprocessed_reports, filename = preprocess_medical_knowledge()
    # Extracting entities using biobern2
    entity_lists_bern2 = []

    #print("Length of preprocessed reports: ", len(preprocessed_reports))
    #print("\n\n")
    entity_lists_bern2 = []
    for report in preprocessed_reports:
        extracted_entities = extract_entities_bern2(report)
        print("Extracted entities: ", extracted_entities)
        if extracted_entities is None:
            continue
        if extracted_entities[0].get("entities"):
            entity_lists_bern2.append(extracted_entities)
    #print("Length of entity_lists_bern2: ", len(entity_lists_bern2))
    #print("\n\n")   

    # Extract relations between entities in each report in entity_lists_bern2
    predicted_rels = []
    for entity_list in entity_lists_bern2:
        for report in entity_list:
            #print("Report: ", report)
            relations = extract_relations(report, re_model, re_tokenizer)
            predicted_rels.append(relations)
    #print("Length of predicted_rels: ", len(predicted_rels))
            
    
    all_graphs = []
    i = 0 
    colors = ['#ADD8E6', '#90EE90', '#FFDAB9', '#E6E6FA', '#FFC0CB', '#F4A460', '#D3D3D3', '#AFEEEE']


    for entity_list, rel_list in zip(entity_lists_bern2, predicted_rels):
            i = i + 1
            G = construct_knowledge_graph(entity_list, rel_list, colors[i % len(colors)])
            filename_i = f"report{i}.{filename}"
            kg = visualize_knowledge_graph(G, filename_i, False)
            # check if G is None or empty
            if (G is not None) or (len(G) > 0) or (nx.is_empty(G)) == False:
                all_graphs.append(G)
            
     
    # Merge all the knowledge graphs into a single graph, preserving the colors in the original graphs
    G_all = nx.compose_all(all_graphs)
    
    # Count occurrences of each label across all graphs
    label_counter = Counter()
    for G in all_graphs:
        for _, data in G.nodes(data=True):
            if 'label' in data:
                label_counter[data['label']] += 1

    # Identify labels that are common to all graphs
    common_labels = {label for label, count in label_counter.items() if count > 1}
    #print("Common labels: ", common_labels)

    # Highlight nodes with common labels in the merged graph
    for node, data in G_all.nodes(data=True):
        if 'label' in data and data['label'] in common_labels:
            # Check if the node is present in more than one graph
            node_count = sum(node in G.nodes() for G in all_graphs)
            if node_count > 1:
                data['color'] = 'red'
            else:
                data['color'] = colors[i % len(colors)]
            
    # update the knowledge graph color to red

    print("\n\n")
    visualize_knowledge_graph(G_all, filename, True) 
    return 1