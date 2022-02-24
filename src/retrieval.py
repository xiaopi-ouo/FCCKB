import abc
import numpy as np
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor

class SimilarityStrategy(abc.ABC):

    @abc.abstractmethod
    def compute(self, sample, claim_embeddings, k, encoder, *args):
        NotImplemented

    def get_top_claim_index(self, scores, k):

        # To get a set of indexes whose claims have the highest value.
        indexes = (-scores).argpartition(k)[:k]
        buffer = []
        for index in indexes:
            buffer.append((index, scores[index]))
        buffer = sorted(buffer, key = lambda x:-x[1])
        buffer = [x[0] for x in buffer]
        return buffer

class BaselineSimilarityStrategy(SimilarityStrategy):

    def compute(self, sample, claim_db, k, encoder, prepend_title_sentence=True, *args):

        if prepend_title_sentence and "title" in sample:
            sentence = sample["title"] + " " + sample["sentence"]
        else:
            sentence = sample["sentence"]

        sentence_embedding = encoder.encode(sentence)
        scores = claim_db.get_scores(sentence_embedding)
        top_claim_index = self.get_top_claim_index(scores, k)

        return top_claim_index

class SRLSimilarityStrategy(SimilarityStrategy):

    def __init__(self):
        self.srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        self.srl_predictor._model = self.srl_predictor._model.cuda()

    def compute(self, sample, claim_db, k, encoder, prepend_title_sentence=True, prepend_title_frame=True):
        

        if prepend_title_sentence and "title" in sample:
            sentence = sample["title"] + " " + sample["sentence"]
        else:
            sentence = sample["sentence"]

        sentence_embedding = encoder.encode(sentence)
        scores = claim_db.get_scores(sentence_embedding)

        # Get Frames
        if "frames" not in sample:
            sample["frames"] = self.get_frames(sample)

        frames = sample["frames"]
        for frame in frames:

            if prepend_title_frame and "title" in sample:
                frame = sample["title"] + " " + frame

            frame_embedding = encoder.encode(frame)
            similarity = claim_db.get_scores(frame_embedding)
            scores = np.maximum(scores, similarity)
        
        top_claim_index = self.get_top_claim_index(scores, k)

        return top_claim_index
    
    def get_frames(self, sample):
        frames = []
        try:
            res = self.srl_predictor.predict(
                sentence=sample["sentence"]
            )
            words = res["words"]
            for frame in res["verbs"]:
                buffer = []
                for word, tag in zip(words, frame["tags"]):
                    if tag != "O":
                        buffer.append(word)

                frames.append(" ".join(buffer))
        except Exception as e:
            print(e)

        return frames

        
class ClaimRetriever:
    
    def __init__(self, claims, encoder_and_database_factory, similarity_strategy):
        """Constructor

        Args:
            claims (`List[str]`):
                claims in the database.
            encoder_and_database_factory:
                factory to create the database and the encoder to encode query sentences.
            similarity_strategy:
                BaselineSimilarityStrategy or SRLSimilarityStrategy.
                
        Return:
            `List[List[int]]`: Indexes of retrieved claims for each sample.
            
        """
        self.encoder = encoder_and_database_factory.get_encoder()
        self.claim_db = encoder_and_database_factory.get_database(claims)
        self.similarity_strategy = similarity_strategy

    def retrieve(self, samples, k, prepend_title_sentence=True, prepend_title_frame=True):
        """Return a list of claim index

        Args:
            samples (`List[dict]`):
                each element has the following key
                    sentence (`str`):
                        Query sentence.
                    frames (`List[str]`, *optional*):
                        Frames of the query sentence
                    title (`str`, *optional*):
                        Title of the document where the sentence comes from.
            k:
                retrieve k claims.
        Return:
            `List[List[int]]`: Indexes of retrieved claims for each sample.
            
        """
        indexes = []
        for sample in tqdm(samples):
            indexes.append(self.similarity_strategy.compute(sample, self.claim_db, k, self.encoder, prepend_title_sentence, prepend_title_frame))

        return indexes