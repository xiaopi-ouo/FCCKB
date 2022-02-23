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