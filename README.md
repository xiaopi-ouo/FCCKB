# FCCKB
Source code for Fact-checking by Claim Knowledge Base

# Usage
Claim Retriever
```
from src.dataset import *
from src.encoder import *
from src.metric import *
from src.retrieval import *

with open("dataset/fever_preprocess/dev.json") as f:
    data = json.load(f)

dataset = FEVERDataset(data)
samples = dataset.get_evidence()
claims = dataset.get_claims()
golden_index = dataset.get_golden_index()

encoder_and_database_factory = SBERTEncoderAndDatabaseFactory()  # For the creation of the database and sentence encoder
similarity_strategy = SRLSimilarityStrategy()                    # For the computation of similarity

retriever = ClaimRetriever(claims, encoder_and_database_factory, similarity_strategy)
retrieval_result = retriever.retrieve(samples,                                            # query sentences 
                                      k=args.k,                                           # number of single claims to be retrieved for each sentence
                                      prepend_title_sentence=args.prepend_title_sentence, # whether to prepend title to each sentence (True/False)
                                      prepend_title_frame=args.prepend_title_frame)       # whether to prepend title to each frames (True/False)
```
Authenticity Predictor
```
import json
from src.dataset import *
from src.predictor import *

path_data = "dataset/fever_preprocess/dev.json"
with open(path_data) as f:
    data = json.load(f)

dataset = FEVERDataset(data)
samples = dataset.get_evidence()
claims = dataset.get_claims()
golden_index = dataset.get_golden_index()
claim_authenticity = data["claim-authenticity"]

# Load the retrieved claims of each sentence in the first stage.
with open("save/dev/sbert-baseline.pkl", "rb") as f:
    retrieved_result = pickle.load(f)

# Number of retrieved claims used for inference.
k = 20

baseline_authenticity_predictor = BasicAuthenticityPredictor(claims, claim_authenticity)
results = baseline_authenticity_predictor.predict(samples, retrieved_result, k=k)
```

# Data Format
The original data was from [FEVER](https://aclanthology.org/N18-1074/)
```javascript
{
  "evidence" : [
   {
     'evidence-id': 'Soul_Food_-LRB-film-RRB-_0',
     'sentence': "Soul Food is a 1997 American comedy-drama film produced by Kenneth `` Babyface '' Edmonds , Tracey Edmonds and Robert Teitel and released    by Fox 2000 Pictures .\tRobert Teitel\tRobert Teitel\tcomedy-drama film\tcomedy-drama film\tTracey Edmonds\tTracey Edmonds\tFox 2000 Pictures\tFox 2000 Pictures",
     'claim-id': [ 145451, 146225, 147051, 147270, 148389, 148837, 150400, 150822, 150931, 150982, 152336],
     'position': ['SUPPORTS', 'REFUTES', 'REFUTES', 'SUPPORTS', 'REFUTES', 'SUPPORTS', 'SUPPORTS', 'REFUTES', 'SUPPORTS', 'REFUTES', 'REFUTES'],
     'title': 'Soul Food -LRB-film-RRB-',
     'frames': [
                "Soul Food is a 1997 American comedy - drama film produced by Kenneth ` ` Babyface '' Edmonds , Tracey Edmonds and Robert Teitel and released by Fox 2000 Pictures . Robert Teitel Robert Teitel comedy - drama film comedy - drama film Tracey Edmonds Tracey Edmonds Fox 2000 Pictures Fox 2000 Pictures",
                "a 1997 American comedy - drama film produced by Kenneth ` ` Babyface '' Edmonds , Tracey Edmonds and Robert Teitel",
                "Babyface",
                "a 1997 American comedy - drama film released by Fox 2000 Pictures"
                ]
    }
  ...
  ]
  "claim" : 
  [
    'Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.',
    'Roman Atwood is a content creator.',
    'History of art includes architecture, dance, sculpture, music, painting, poetry literature, theatre, narrative, film, photography and graphic arts.',
    ...
  ]
}
```
All the `evidence` are the input query sentences in our problem. We use the term `evidence` since all of them are actually evidence in FEVER.
