import re
import requests
import hashlib

from transformers import AutoTokenizer
from zero_shot_re import RelTaggerModel, RelationExtractor

model1 = RelTaggerModel.from_pretrained("fractalego/fewrel-zero-shot")
tokenizer1 = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
relations = ['associated', 'interacts']
extractor = RelationExtractor(model1, tokenizer1, relations)

#tokenizer2 = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#model2 = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

#visualization
from pyvis.network import Network
import networkx as nx
from collections import defaultdict
from matplotlib import pyplot as plt
import itertools

#scispacy
import scispacy
import spacy
#import en_core_sci_scibert   
import en_ner_bionlp13cg_md
from spacy import displacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker

def remove_duplicates(predicted_rels):
    seen = set()
    unique_rels = []
    for rel in predicted_rels:
        if (rel['head'], rel['tail'], rel['type']) not in seen:
            unique_rels.append(rel)
            seen.add((rel['head'], rel['tail'], rel['type']))
    return unique_rels


def query_plain(text, url="http://bern2.korea.ac.kr/plain"):
    return requests.post(url, json={'text': text}).json()




if __name__ == "__main__":
    
    # Load the medical knowledge as a list of strings
    with open('medical_knowledge.txt', 'r') as f:
        medical_knowledge = f.readlines()

    medical_knowledge_preserved = medical_knowledge
    # Pre-process the medical knowledge by lowercasing and removing special characters
    medical_knowledge = [re.sub(r'[^a-zA-Z0-9.\s]', '', s).lower() for s in medical_knowledge]

    print(medical_knowledge)

    result = " ".join(medical_knowledge_preserved)
    
    text1 = result
    #print(text1)

    result = " ".join(medical_knowledge)
    text2 = result
    #print(text2)

    entity_list = []

    # Not cleaned string (in case of mutations in the form of character.something, so not removing character)
    try:
        entity_list.append(query_plain(text1))
    except:
        print('invalid sentence')
        
    # Cleaned string
    try:
        result = query_plain(text2)
        #print(result)
    except:
        print('invalid sentence')    
    
    #print(len(entity_list[0]['annotations']))
    #print("\n\n\n")    
    #print(len(result['annotations']))
    #print("\n\n\n")    
    
    entity_list[0]['annotations'].extend(result['annotations'])

    #print(entity_list[0]['annotations'])    
    #print("\n\n\n")
        
    parsed_entities = []


    #relations = ['associated', 'interacts']
    #extractor = RelationExtractor(model2, tokenizer2, relations)

    #print(entity_list)

    for entities in entity_list:
        e = []
        if not entities.get('annotations'):
            parsed_entities.append({'text':entities['text'], 'text_sha256': hashlib.sha256(entities['text'].encode('utf-8')).hexdigest()})
            continue
        for entity in entities['annotations']:
            other_ids = [id for id in entity['id'] if not id.startswith("BERN")]
            entity_type = entity['obj']                                                             # maybe filter by the correct id (in case of duplicates)
            entity_name = entities['text'][entity['span']['begin']:entity['span']['end']]
            try:
                entity_id = [id for id in entity['id'] if id.startswith("BERN")][0]
            except IndexError:
                entity_id = entity_name
            e.append({'entity_id': entity_id, 'other_ids': other_ids, 'entity_type': entity_type, 'entity': entity_name})

    parsed_entities.append({'entities':e, 'text':entities['text'], 'text_sha256': hashlib.sha256(entities['text'].encode('utf-8')).hexdigest()})
    print(len(parsed_entities))  
    print(len(parsed_entities[0]['entities']))

    candidates = [s for s in parsed_entities if (s.get('entities')) and (len(s['entities']) > 1)]
    predicted_rels = []
    for c in candidates:
        combinations = itertools.combinations([{'name':x['entity'], 'id':x['entity_id']} for x in c['entities']], 2)
        for combination in list(combinations):
            try:
                ranked_rels = extractor.rank(text=c['text'].replace(",", " "), head=combination[0]['name'], tail=combination[1]['name'])
                if ranked_rels[0][1] > 0.85:
                    predicted_rels.append({'head': combination[0]['id'], 'tail': combination[1]['id'], 'type':ranked_rels[0][0], 'source': c['text_sha256']})
            except:
                pass

    print(predicted_rels)


    #print(medical_knowledge_preserved[0])

    nlp = spacy.load("en_core_sci_scibert")
    #nlp_spacy = spacy.load("en_ner_bionlp13cg_md")
    doc = nlp(medical_knowledge_preserved[0]) 
    print(list(doc.sents))
    print("\n")
    print(doc.ents)

    #print(len(predicted_rels))
    #print("\n\n\n")
    #print(predicted_rels)
    #print("\n\n\n")
    # Remove duplicates from predicted_rels
    predicted_rels = remove_duplicates(predicted_rels)
    #print(len(predicted_rels))
    #print("\n\n\n")


    g = Network(notebook=False)

    for rel in predicted_rels:
        g.add_node(rel['head'])
        g.add_node(rel['tail'])
        g.add_edge(rel['head'], rel['tail'], label=rel['type'])

    g.set_options("""
        var options = {
        "physics": {
            "enabled": false
        },
        "edges": {
            "font": {
                "size": 20
            }
        },
        "nodes": {
            "font": {
                "size": 20
            }
        }
    }""")

    g.show("knowledge_graph_python.html")
