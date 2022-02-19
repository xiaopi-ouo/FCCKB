class FEVERDataset:

    def __init__(self, data):
        self.data = data
        self.preprocess(data)

        self.golden_index = []

        claim_id_to_index = {}
        current_claim_index = 0
        for evidence in data["evidence"]:
            self.golden_index.append(evidence["claim-id"])
    
    def get_evidence(self):
        return self.data["evidence"]
    
    def get_claims(self):
        return self.data["claim"]

    def get_golden_index(self):
        return self.golden_index

    def preprocess(self, data):
        for sample in data["evidence"]:
            title, sentence_id = sample["evidence-id"].rsplit("_", 1)
            title = " ".join(title.split("_"))
            sample["title"] = title
