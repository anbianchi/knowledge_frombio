# %%
import re
import os
import requests
import hashlib
import random
import pandas as pd
import argparse

#visualization
from pyvis.network import Network
import networkx as nx
from collections import defaultdict
from matplotlib import pyplot as plt
import itertools
from IPython.display import Image
from IPython.display import HTML

#scispacy
import scispacy
import spacy
import en_core_sci_scibert   
#import en_ner_bionlp13cg_md
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.linking import EntityLinker

#relation extraction
from transformers import AutoTokenizer, AutoModel
from zero_shot_re import RelTaggerModel, RelationExtractor

import torch
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%

# Define a dictionary to store all the knowledge graphs
knowledge_graphs = {}

# Define a function to construct an interactive knowledge graph
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
        else:
            print(f"Skipped adding self-relationship for entity: {relation['entity1_id']}")
    return G
    
# %%
def load_scispacy_model():
    nlp = spacy.load("en_core_sci_scibert")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    return nlp

def load_bern2_model(preprocessed_reports):
    entity_list = []
    try:
        entity_list.append(requests.post("http://bern2.korea.ac.kr/plain", json={'text': preprocessed_reports}).json())
        #entity_list[0]['annotations'].extend(entity_list['annotations'])
    except:
        print('invalid sentence')
        
    return entity_list

def load_relation_extraction_model():
    model = RelTaggerModel.from_pretrained("fractalego/fewrel-zero-shot")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    relations = ['associated', 'interacts']
    extractor = RelationExtractor(model, tokenizer, relations)
    return extractor

def extract_entities_scispacy(preprocessed_reports):
    nlp = load_scispacy_model()
    doc = nlp(preprocessed_reports)
           
    """ entity = doc.ents[1]

    print("Name: ", entity)
    >>> Name: bulbar muscular atrophy

    # Each entity is linked to UMLS with a score
    # (currently just char-3gram matching).
    linker = nlp.get_pipe("scispacy_linker")
    for umls_ent in entity._.kb_ents:
        print(linker.kb.cui_to_entity[umls_ent[0]])          
               """
          
    entities = []
    for ent in doc.ents:
        entity = {
            'entity_id': ent.text,
            'other_ids': [],
            'entity_type': ent.label_,
            'entity': ent.text
        }
        entities.append(entity)

    candidates = {
    'entities': entities,
    'text': doc.text,
    'text_sha256': hashlib.sha256(doc.text.encode()).hexdigest()
    }
    return candidates

def extract_entities_bern2(preprocessed_reports):
    entity_list = load_bern2_model(preprocessed_reports)
    parsed_entities = []
    #print(preprocessed_reports)
    for entities in entity_list:
        e = []
        if not entities.get('annotations'):
            parsed_entities.append({'text':entities['text'], 'text_sha256': hashlib.sha256(entities['text'].encode('utf-8')).hexdigest()})
            continue
        for entity in entities['annotations']:
            other_ids = [id for id in entity['id'] if not id.startswith("BERN")]
            entity_type = entity['obj']   
            entity_prob = entity['prob']                                                          
            entity_name = entities['text'][entity['span']['begin']:entity['span']['end']]
            try:
                entity_id = [id for id in entity['id'] if id.startswith("BERN")][0]
            except IndexError:
                entity_id = entity_name
            e.append({'entity_id': entity_id, 'other_ids': other_ids, 'entity_type': entity_type, 'entity': entity_name, 'entity_prob': entity_prob})

    parsed_entities.append({'entities':e, 'text':entities['text'], 'text_sha256': hashlib.sha256(entities['text'].encode('utf-8')).hexdigest()})
    
    #return entity_list
    return parsed_entities


# %%
def load_Bio_ClinicalBERT_model():
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return model, tokenizer

# Define a function to extract relations between entities in a report
def extract_relations(report, model, tokenizer):
    # Tokenize the report
    inputs = tokenizer(report["text"], return_tensors="pt")
    # Generate embeddings for each entity in the report
    entity_embeddings = []
    for entity in report["entities"]:
        entity_text = entity["entity"]
        entity_inputs = tokenizer(entity_text, return_tensors="pt")
        with torch.no_grad():
            entity_output = model(**entity_inputs)[0][:, 0, :]
        entity_embeddings.append(entity_output)
    # Compute the similarity between each pair of entity embeddings
    relations = []
    for i, e1 in enumerate(entity_embeddings):
        for j, e2 in enumerate(entity_embeddings):
            if i >= j:
                continue
            with torch.no_grad():
                similarity = torch.cosine_similarity(e1, e2).item()
            # If the similarity is above a threshold, predict a relation
            if similarity > 0.85:
                relation = {
                    "entity1_id": report["entities"][i]["entity_id"],
                    "entity2_id": report["entities"][j]["entity_id"],
                    "relation_type": "associated",
                    "confidence": similarity,
                }
                relations.append(relation)
    return relations

def visualize_knowledge_graph(G, filename):
    
    if filename is None:
        new_filename = 'merged_kg'
    else:
        # Extract the patient number from the filename
        patient_number = filename.split('_')[0].replace('#', '')
        # Construct the new filename
        new_filename = f"#{patient_number}_kg"
        
    # Visualize the knowledge graph
    nt = Network(height='800px', width='100%', font_color='black')
    for node, data in G.nodes(data=True):
        nt.add_node(node, label=data['label'], color=data['color'])
    for source, target, data in G.edges(data=True):
        nt.add_edge(source, target, title=data['label'])
    #html = nt.show('notebook.html')
    #display(HTML(html)) 
    
    # Save the graph to an HTML file
    nt.save_graph(f'{new_filename}.html')
    
    return nt

# %%
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
    for report in preprocessed_reports:
        entity_lists_bern2.append(extract_entities_bern2(report))
    #print("Length of entity_lists_bern2: ", len(entity_lists_bern2))   

    # Extract relations between entities in each report in entity_lists_bern2
    predicted_rels = []
    for entity_list in entity_lists_bern2:
        for report in entity_list:
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
            kg = visualize_knowledge_graph(G, filename_i)
            all_graphs.append(G)
            # Add common nodes to set
            
     
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
    print("Common labels: ", common_labels)

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
    visualize_knowledge_graph(G_all, filename) 
    return 1

# %%
def clear_reports_folder():
    # Remove existing files in the 'diagnostic_reports' folder
    for file in os.listdir('diagnostic_reports'):
        os.remove(os.path.join('diagnostic_reports', file))

def generate_reports(input_file, num_patients=None):
    # Create the 'diagnostic_reports' folder if it doesn't exist
    if not os.path.exists('diagnostic_reports'):
        os.makedirs('diagnostic_reports')

    # Read the CSV file with no truncation
    df = pd.read_csv(input_file)

    # Limit the number of patients if num_patients is provided
    if num_patients:
        df = df.head(num_patients)

    # Generate .txt reports for each patient
    for idx, row in df.iterrows():
        # Use the new headers based on your dataset
        patient_id = row['subject_id']
        print("Patient:",patient_id)
        
        # For each illness_history column, create a separate file if the value is not NaN
        for col in df.columns:
            if "illness_history_" in col and pd.notna(row[col]):
                with open(f'diagnostic_reports/#{patient_id}_{col}.txt', 'w', encoding='utf-8') as f:
                    f.write(str(row[col]))
        #print the number of files in the diagnostic_reports folder
        print("Number of files in the diagnostic_reports folder: ", len(os.listdir('diagnostic_reports')))
        
        preprocessed_reports, filename = preprocess_medical_knowledge()
        process_reports(preprocessed_reports, filename=filename)

        # Clear the reports folder for the next patient
        clear_reports_folder()
        

    print(f"Reports processing completed.")


# %%
def process_reports_from_dataset(dataset_file):
    """
    Process reports from the specified dataset file.
    """
    generate_reports(input_file=dataset_file, num_patients=None)
    #preprocessed_reports, filename = preprocess_medical_knowledge()
    #process_reports(preprocessed_reports, filename)
    
def process_reports_from_folder():
    """
    Process reports that are manually inserted into the 'diagnostic_reports' folder.
    """
    preprocessed_reports = preprocess_medical_knowledge()
    process_reports(preprocessed_reports)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process medical reports in two modes.')
    parser.add_argument('--manual', action='store_true', help='Use manually inserted reports in the diagnostic_reports folder.')
    parser.add_argument('--dataset', type=str, help='Specify the path to the dataset file to process reports from.')

    args = parser.parse_args()

    if args.manual:
        process_reports_from_folder()
    elif args.dataset:
        process_reports_from_dataset(args.dataset)
    else:
        print("Please specify a mode: --manual for manually inserted reports or --dataset <path_to_dataset> for processing from a dataset.")




