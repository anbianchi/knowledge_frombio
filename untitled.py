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
from transformers import AutoTokenizer
from zero_shot_re import RelTaggerModel, RelationExtractor

# %%
# Define a dictionary to store all the knowledge graphs
knowledge_graphs = {}

# Define a function to construct an interactive knowledge graph
def construct_knowledge_graph(entities, relationships):
    # Choose a random color for the knowledge graph
    colors = ['blue', 'green', 'yellow', 'orange', 'purple', 'pink']
    color = random.choice([c for c in colors if c not in knowledge_graphs.values()])
    
    # Create a new knowledge graph
    G = nx.DiGraph()
    
    # Add nodes for all the entities
    for entity in entities:
        G.add_node(entity, color=color)
    
    # Add edges for all the relationships
    for relationship in relationships:
        G.add_edge(relationship[0], relationship[1], label=relationship[2])
    
    # Visualize the knowledge graph
    nt = Network(height='800px', width='100%', bgcolor='#222222', font_color='white')
    nt.from_nx(G)
    nt.show('knowledge_graph.html')
    
    # Store the knowledge graph in the dictionary
    knowledge_graphs[color] = G
    
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

def extract_relations(preprocessed_reports, extractor):
    relations = extractor.extract(preprocessed_reports)
    return relations

preprocessed_reports = preprocess_medical_knowledge()
# Extracting entities using biobern2
entity_lists_bern2 = []

for report in preprocessed_reports:
    entity_lists_bern2.append(extract_entities_bern2(report))
print(entity_lists_bern2)   
 
# Extracting entities using scispacy and biobert as a model
""" entity_lists_scispacy = []
for report in preprocessed_reports:
    entity_lists_scispacy.append(extract_entities_scispacy(report))
print(entity_lists_scispacy)
 """ 

# Extracting relations using zero-shot relation extraction model from the entity_lists_bern2
