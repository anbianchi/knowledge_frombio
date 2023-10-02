from transformers import AutoTokenizer, AutoModel
import torch

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
