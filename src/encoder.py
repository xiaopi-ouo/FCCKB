import abc
import tensorflow_hub as hub
from simcse import SimCSE
from sentence_transformers import SentenceTransformer

class SentenceEncoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, inputs):
        NotImplemented

class USESentenceEncoder(SentenceEncoder):
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        
    def encode(self, inputs):
        isString = isinstance(inputs, str)
        if isString:
            inputs = [inputs]
        elif not isinstance(inputs, list):
            raise Exception("Input should be a string or a list of string")

        embedding = self.model(inputs).numpy()

        if isString:
            embedding = embedding[0]

        return embedding

class SimCSESentenceEncoder(SentenceEncoder):
    def __init__(self):
        self.model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        
    def encode(self, inputs):
        embedding = self.model.encode(inputs)
        return embedding.numpy()

class SBERTSentenceEncoder(SentenceEncoder):
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    def encode(self, inputs):
        return self.model.encode(inputs, show_progress_bar=False)

class BM25SentenceEncoder(SentenceEncoder):
    
    def encode(self, inputs):
        if isinstance(inputs, str):
            return word_tokenize(inputs)
        elif isinstance(inputs, list):
            tokenized_corpus = [word_tokenize(s) for s in inputs]
            return BM25Okapi(tokenized_corpus)

class ClaimDatabase(abc.ABC):
    
    def __init__(self, claims):
        self.claims = claims

    @abc.abstractclassmethod
    def get_scores(self, query_obj):
        NotImplemented
        
class BM25Database(ClaimDatabase):
    
    def __init__(self, claims):
        super().__init__(claims)
        self.db = BM25Okapi([word_tokenize(claim) for claim in claims])
    
    def get_scores(self, query_obj):
        return self.db.get_scores(query_obj)
    
class NNEncoderDatabase(ClaimDatabase):
    
    def __init__(self, claims, encoder):
        super().__init__(claims)
        self.db = encoder.encode(claims)
        
    def get_scores(self, query_obj):

        if not isinstance(query_obj, np.ndarray):
            raise Exception("Type Error")

        return query_obj.dot(self.db.T)

class EncoderAndDatabaseFactory(abc.ABC):

    @abc.abstractclassmethod
    def get_encoder(self):
        NotImplemented
        
    @abc.abstractclassmethod
    def get_database(self):
        NotImplemented

class BM25EncoderAndDatabaseFactory(EncoderAndDatabaseFactory):
    
    def get_encoder(self):
        return BM25SentenceEncoder()

    def get_database(self, claims):
        return BM25Database(claims)

class NNEncoderAndDatabaseFactory(EncoderAndDatabaseFactory):
    
    @abc.abstractclassmethod
    def __init__(self):
        NotImplemented

    def get_encoder(self):
        return self.encoder
    
    def get_database(self, claims):
        return NNEncoderDatabase(claims, self.encoder)

class SBERTEncoderAndDatabaseFactory(NNEncoderAndDatabaseFactory):

    def __init__(self):
        self.encoder = SBERTSentenceEncoder()
        
class SimCSEEncoderAndDatabaseFactory(NNEncoderAndDatabaseFactory):

    def __init__(self):
        self.encoder = SimCSESentenceEncoder()

class USEEncoderAndDatabaseFactory(NNEncoderAndDatabaseFactory):
    def __init__(self):
        self.encoder = USESentenceEncoder()