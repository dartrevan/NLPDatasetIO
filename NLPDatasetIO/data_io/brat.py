from NLPDatasetIO.document import Document, Entity, Relation
from typing import List
from NLPDatasetIO.data_io.utils import find_offset, read_file
from glob import glob
import os
import re

BRAT_FORMAT = r'(?P<entity_id>^T[0-9]+)\t(?P<type>[a-zA-Z\_]+) (?P<positions>[0-9; ]+)\t(?P<text>.*)'
BRAT_FORMAT = r'(?P<entity_id>^T[0-9\_a-z]+)\t(?P<type>[a-zA-Z\_\-]+) (?P<positions>[0-9; ]+)\t(?P<text>.*)'
# N1      Reference T1 GeneOntology:900   CCNG1
ANNOTATION = r'(?P<id>^N[0-9]+)\tReference (?P<entity_id>T[0-9]+) (?P<ontology_name>[a-zA-Z\_]+)\:(?P<concept_id>[a-zA-Z\_0-9]+)\t(?P<concept_name>.*)'
NOTE = r'#\d+\tAnnotatorNotes (?P<entity_id>T[0-9]+)\t(?P<note>.*)'


class AnnFilesIterator(object):

    def __init__(self, directory):
        self.directory = directory
        txt_files_pattern = os.path.join(directory, '*.txt')
        ann_files_pattern = os.path.join(directory, '*.ann')
        txt_files = [txt_file for txt_file in glob(txt_files_pattern)]
        ann_files = [ann_file for ann_file in glob(ann_files_pattern)]
        self.txt_files = list(sorted(txt_files))
        self.ann_files = list(sorted(ann_files))

    def __iter__(self):
        return iter(zip(self.txt_files, self.ann_files))


def parse_annotation(annotation_raw):
    # print(annotation_raw)
    annotation = re.search(BRAT_FORMAT, annotation_raw).groupdict()
    positions = re.findall(r'\d+', annotation['positions'])
    positions = [int(pos) for pos in positions]
    annotation['start'] = min(positions)
    annotation['end'] = max(positions)
    return annotation


def parse_label_annotation(annotation_raw):
    try:
        annotation = re.search(ANNOTATION, annotation_raw).groupdict()
        return annotation['entity_id'], annotation['concept_id']
    except:
        return None, None

def parse_note(annotation_raw):
    annotation = re.search(NOTE, annotation_raw).groupdict()
    return annotation['entity_id'], annotation['note']


def extract_entities_from_brat(annotations_raw: str, text: str) -> List[Entity]:
    entities = {}
    for annotation_raw in annotations_raw.split('\n'):
        if not annotation_raw.startswith('T'): continue
        try:
            annotation = parse_annotation(annotation_raw)
        except:
            continue
        start = annotation['start']
        end = annotation['end']
        entity_text = annotation['text']
        if text[start:end] != annotation['text']:
            start, end = find_offset(text, entity_text, start, end)
            entity_text = text[start:end]
        entities[annotation['entity_id']] = Entity(entity_id=annotation['entity_id'],
                                                   text=entity_text,
                                                   start=start,
                                                   end=end,
                                                   type=annotation['type'])
    return entities


def extract_entity_labels(annotations_raw: str):
    entity_labels = {}
    for annotation_raw in annotations_raw.split('\n'):
        if not annotation_raw.startswith('N'): continue
        entity_id, concept_id = parse_label_annotation(annotation_raw)
        entity_labels[entity_id] = concept_id
    return entity_labels


def extract_entity_notes(annotations_raw: str):
    entity_notes = {}
    for annotation_raw in annotations_raw.split('\n'):
        if not annotation_raw.startswith('#'): continue
        entity_id, note = parse_note(annotation_raw)
        entity_notes[entity_id] = note
    return entity_notes


def set_labels_and_notes(entities, entity_labels, notes):
    for entity_id, entity in entities.items():
        entity.label = entity_labels.get(entity.entity_id, None)
        entity.note = notes.get(entity.entity_id, None)


def get_entities_id(relation_line):
    relation_line_parts = relation_line.split(' ')
    return relation_line_parts[1].split(":")[1].strip(), \
           relation_line_parts[2].split(":")[1].strip()


def extract_relations_from_brat(annotations_raw: str):
    relations = []
    for annotation_raw in annotations_raw.split('\n'):
        if not annotation_raw.startswith('R'): continue
        line_parts = annotation_raw.split('\t')
        entity1_id, entity2_id = get_entities_id(line_parts[1])
        type = line_parts[1].split(' ')[0]
        id = line_parts[0]
        relations.append(Relation(relation_id=id, entity_id_1=entity1_id,
                                  entity_id_2=entity2_id, type=type))
    return relations


def read_from_brat(path_to_brat_folder):
    document_id = 0
    documents = []
    for text_file, ann_file in AnnFilesIterator(path_to_brat_folder):
        doc_id = os.path.basename(text_file).replace('.txt', '')
        text = read_file(text_file)
        annotations_raw = read_file(ann_file)
        entities = extract_entities_from_brat(annotations_raw, text)
        entity_labels = extract_entity_labels(annotations_raw)
        notes = extract_entity_notes(annotations_raw)
        set_labels_and_notes(entities, entity_labels, notes)
        relations = extract_relations_from_brat(annotations_raw)
        document = Document(doc_id=doc_id, text=text,
                            entities=entities, relations=relations)
        documents.append(document)
        document_id += 1
    return documents


def save_text_file(path_to_save: str, document: Document):
    with open(path_to_save, 'w', encoding='utf-8') as output_stream:
        output_stream.write(document.text)


def save_ann_file(path_to_save: str, document: Document):
    with open(path_to_save, 'w', encoding='utf-8') as output_stream:
        for entity in document.entities.values():
            if isinstance(entity.entity_id, str) and entity.entity_id.startswith('T'):
                output_stream.write(f'{entity.entity_id}\t{entity.type} {entity.start} {entity.end}\t{entity.text}\n')
            else:
                output_stream.write(f'T{entity.entity_id}\t{entity.type} {entity.start} {entity.end}\t{entity.text}\n')
        for relation in document.relations:
            if not isinstance(relation.entity_id_1, str) or not relation.entity_id_1.startswith('T'):
                relation.entity_id_1 = f'T{relation.entity_id_1}'
            if not isinstance(relation.entity_id_2, str) or not relation.entity_id_2.startswith('T'):
                relation.entity_id_2 = f'T{relation.entity_id_2}'
            output_stream.write(f'{relation.relation_id}\t{relation.type} Arg1:{relation.entity_id_1} Arg2:{relation.entity_id_2}\n')
        nid = 1
        for entity in document.entities.values():
             if entity.label is None: continue
             ontology_name = 'GeneOntology' if 'PRGE' in entity.type or 'target' in entity.type else 'DiseaseDB'
             output_stream.write(f'N{nid}\tReference {entity.entity_id} {ontology_name}:{entity.label}\t{entity.concept_name}\n')
             nid += 1



def save_brat(data, path_to_save: str):
    for document in data.documents:
        ann_file = os.path.join(path_to_save, f'{document.doc_id}.ann')
        txt_file = os.path.join(path_to_save, f'{document.doc_id}.txt')
        save_text_file(txt_file, document)
        save_ann_file(ann_file, document)
