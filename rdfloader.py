from pathlib import Path
from rdflib import RDF
from sklearn.model_selection import train_test_split

from graph import Vertex, KnowledgeGraph, extract_instance
from rdf2vec import RDF2VecTransformer

class RDFLoader:

    def __init__(self, data_path, test_size=0.2):
        self.data_path = Path(data_path).resolve() 

        self.triples = self.read_triples(self.data_path)
        self.functions = self.get_function_entities(self.triples)
        self.train_functions, self.test_functions = train_test_split(list(self.functions), test_size=test_size)

        self.kg = self.triples_to_kg(self.triples)

        self.train_subgraphs = [extract_instance(self.kg, func) for func in self.train_functions]
        self.test_subgraphs = [extract_instance(self.kg, func) for func in self.test_functions]

    def read_triples(self, file_path):
        triples = []
        with open(str(file_path), 'r', encoding='utf-8') as file:
            for line in file.readlines():
                s, p, o = line.split('\t')
                triples.append((s.strip(), p.strip(), o.strip()))
        return triples

    def get_function_entities(self, triples):
        entities = set()
        type_pred = str(RDF.type)
        for (s, p, o) in triples:
            if p == type_pred:
                entities.add(s)
        return entities

    def triples_to_kg(self, triples):
        kg = KnowledgeGraph()
        for (s, p, o) in triples:
            s_v, o_v = Vertex(s), Vertex(o)
            p_v = Vertex(p, predicate=True)
            kg.add_vertex(s_v)
            kg.add_vertex(p_v)
            kg.add_vertex(o_v)
            kg.add_edge(s_v, p_v)
            kg.add_edge(p_v, o_v)
        return kg

if __name__ == "__main__":
    loader = RDFLoader("./data/data.txt")

    transformer = RDF2VecTransformer(_type='walk', walks_per_graph=500)
    embeddings = transformer.fit_transform(loader.test_subgraphs+loader.train_subgraphs)
    transformer.save_model("rdf.model")

    import pdb; pdb.set_trace()
