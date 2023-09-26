# %%
import re
import os
import requests
import hashlib
import random

#visualization
from pyvis.network import Network
import networkx as nx
from collections import defaultdict
from matplotlib import pyplot as plt
import itertools
from IPython.display import Image


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


# %%
from IPython.display import HTML

# Define a dictionary to store all the knowledge graphs
knowledge_graphs = {}

# Define a function to construct an interactive knowledge graph
def construct_knowledge_graph(report, relations):
    # Choose a random color for the knowledge graph
    colors = ['blue', 'green', 'yellow', 'orange', 'purple', 'pink']
    color = random.choice([c for c in colors if c not in knowledge_graphs.values()])
    
    # Create a new knowledge graph
    G = nx.DiGraph()
    
    # Add nodes
    for entity in report[0]["entities"]:
        G.add_node(entity["entity_id"], label=entity["entity"], color=color)
    
    # Add edges
    for relation in relations:
        print(relation)
        print(relation.keys())
        G.add_edge(relation["entity1_id"], relation["entity2_id"], label=relation["relation_type"], color=color)
        
    return G
    
# Define a function to merge and solve conflicts in knowledge graphs
def merging_and_solving_conflicts():
    # Merge all the knowledge graphs
    merged_graph = nx.DiGraph()
    for graph in knowledge_graphs.values():
        merged_graph = nx.compose(merged_graph, graph)
    
    # Find all the common entities
    entity_counts = defaultdict(int)
    for node in merged_graph.nodes:
        entity_counts[node] += 1
    common_entities = [entity for entity, count in entity_counts.items() if count > 1]
    
    # Resolve conflicts in common entities
    for entity in common_entities:
        # Find all the nodes with the same entity
        nodes = [node for node in merged_graph.nodes if node == entity]
        
        # Find the color of the first node
        color = None
        for node in nodes:
            if node in knowledge_graphs:
                color = knowledge_graphs[node]
                break
        
        # Change the color of all the nodes to the first color
        for node in nodes:
            if node in knowledge_graphs:
                knowledge_graphs[node] = color
            merged_graph.nodes[node]['color'] = color
            




# %%
def preprocess_medical_knowledge():
    # Load the medical knowledge as a list of strings
    medical_knowledge = []
    preprocessed_reports = []
    for filename in os.listdir("diagnostic_reports"):
        with open(os.path.join("diagnostic_reports", filename), "r") as f:
            text = f.read()
            # Pre-process the medical knowledge by lowercasing and removing special characters
            text = re.sub(r'[^a-zA-Z0-9.\s]', '', text).lower()
            medical_knowledge.append(text)
            
            preprocessed_reports.append(text)
    
    return preprocessed_reports




# %%
def load_scispacy_model():
    nlp = spacy.load("en_core_sci_scibert")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    return nlp

def load_bern2_model(preprocessed_reports):
    entity_list = []
    try:
        entity_list.append(requests.post("http://bern2.korea.ac.kr/plain", json={'text': report}).json())
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
 
# Extracting entities using scispacy and biobert as a model
""" entity_lists_scispacy = []
for report in preprocessed_reports:
    entity_lists_scispacy.append(extract_entities_scispacy(report))
print(entity_lists_scispacy)
 """ 

# Extracting relations using zero-shot relation extraction model from the entity_lists_bern2




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


re_model, re_tokenizer = load_Bio_ClinicalBERT_model()

preprocessed_reports = preprocess_medical_knowledge()
# Extracting entities using biobern2
entity_lists_bern2 = []

for report in preprocessed_reports:
    entity_lists_bern2.append(extract_entities_bern2(report))
#print(entity_lists_bern2)   

# Extract relations between entities in each report in entity_lists_bern2
predicted_rels = []
for entity_list in entity_lists_bern2:
    for report in entity_list:
        relations = extract_relations(report, re_model, re_tokenizer)
        predicted_rels.append(relations)
        
#print(predicted_rels)

# Generate and merge knowledge graphs for all reports
G_all = nx.Graph()
for entity_list, rel_list in zip(entity_lists_bern2, predicted_rels):
    #for report, relations in zip(entity_list, rel_list):
        print(entity_list)
        print("\n")
        print(rel_list)
        G = construct_knowledge_graph(entity_list, rel_list)
        G_all = nx.disjoint_union(G_all, G)

def visualize_knowledge_graph(G):
    # Visualize the knowledge graph
    nt = Network(height='800px', width='100%', font_color='black')
    for node, data in G.nodes(data=True):
        nt.add_node(node, label=data['label'], color=data['color'])
    for source, target, data in G.edges(data=True):
        nt.add_edge(source, target, title=data['label'])
    html = nt.show('notebook.html')
    display(HTML(html)) 
 
""" def visualize_knowledge_graph(G):
    pos = nx.spring_layout(G)
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
    
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000, node_color=node_colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show() """   
    
visualize_knowledge_graph(G_all)    

# %%
""" candidates = [s for s in parsed_entities if (s.get('entities')) and (len(s['entities']) > 1)]
predicted_rels = []
print(candidates)
print("\n\n")
for c in candidates:
  combinations = itertools.combinations([{'name':x['entity'], 'id':x['entity_id']} for x in c['entities']], 2)
  for combination in list(combinations):
    try:
      ranked_rels = extractor.rank(text=c['text'].replace(",", " "), head=combination[0]['name'], tail=combination[1]['name'])
      if ranked_rels[0][1] > 0.85:
        predicted_rels.append({'head': combination[0]['id'], 'tail': combination[1]['id'], 'type':ranked_rels[0][0], 'source': c['text_sha256']})
    except:
      pass

print(predicted_rels) """


