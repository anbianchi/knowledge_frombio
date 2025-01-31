import requests
import hashlib

def load_bern2_model(preprocessed_reports):
    entity_list = []
    try:
        entity_list.append(requests.post("http://bern2.korea.ac.kr/plain", json={'text': preprocessed_reports}).json())
    except:
        print('invalid sentence: ', preprocessed_reports)
        print("\n\n")
        entity_list = None
        
    return entity_list

def extract_entities_bern2(preprocessed_reports):
    entity_list = load_bern2_model(preprocessed_reports)
    print("preprocessed reports: ", preprocessed_reports, "\n\n")
    if entity_list is None:
        parsed_entities = None
        return parsed_entities
    
    parsed_entities = []
    #print(preprocessed_reports)
    print(entity_list)
    print("\n\n")
    for entities in entity_list:
        e = []
        if not entities.get('annotations') or not entities.get('text'):
                parsed_entities=None
                #parsed_entities.append({'text':entities['text'], 'text_sha256': hashlib.sha256(entities['text'].encode('utf-8')).hexdigest()})
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

    return parsed_entities

