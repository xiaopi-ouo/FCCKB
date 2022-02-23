import abc
import json
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AuthenticityPredictor(abc.ABC):
    
    def __init__(self, claims, claim_authenticity):
        self.claims = claims
        self.claim_authenticity = claim_authenticity

    def predict(self, samples, retrieved_result, k=5, prepend_title_sentence=True):
        out = []
        for sample, retrieved_claims in tqdm(zip(samples, retrieved_result)):
            # consider k retrieved claims
            retrieved_claims = retrieved_claims[:k]
            out.append(self.predict_sample(sample, retrieved_claims, prepend_title_sentence))
        return out
    
    @abc.abstractclassmethod
    def predict_sample(self, sample, retrieved_claims, prepend_title_sentence):
        NotImplemented
        
class BasicAuthenticityPredictor(AuthenticityPredictor):
    
    def __init__(self, claims, claim_authenticity):
        super().__init__(claims, claim_authenticity)
        hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        self.tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
        self.cache_nli_result = []
        self.cache_claim_id = []

    def predict(self, samples, retrieved_result, k=5, prepend_title_sentence=True):
        self.cache_nli_result = []
        self.cache_claim_id = []
        return super().predict(samples, retrieved_result, k, prepend_title_sentence)

    def predict_sample(self, sample, retrieved_claims, prepend_title_sentence):

        if prepend_title_sentence:
            premise = sample["title"] + sample["sentence"]
        else:
            premise = sample["sentence"]
        
        nli_buffer = []
        claim_buffer = []
        authenticity = True
        for claim_id in retrieved_claims:
            hypothesis = self.claims[claim_id]
            claim_truth_value = self.claim_authenticity[claim_id]
            v = self.inference(premise, self.claims[claim_id])
            
            nli_buffer.append(v)
            claim_buffer.append(claim_id)
            
            if (v == 0 and claim_truth_value == False) or (v == 2 and claim_truth_value == True):
                # sentence supports a false claim or denies a true claim
                authenticity = False

        self.cache_nli_result.append(nli_buffer)
        self.cache_claim_id.append(claim_buffer)

        return authenticity
            
    def predict_by_cache(self, k=5):
        
        out = []

        for nli_buffer, claim_buffer in zip(self.cache_nli_result, self.cache_claim_id):
    
            authenticity = True
            for v, claim_id in zip(nli_buffer[:k], claim_buffer[:k]):
                claim_truth_value = self.claim_authenticity[claim_id]
                if (v == 0 and claim_truth_value == False) or (v == 2 and claim_truth_value == True):
                    # sentence supports a false claim or denies a true claim
                    authenticity = False
                    break
            out.append(authenticity)
        return out
    
    def dump(self, path):
        checkpoint = {
            "nli" : self.cache_nli_result,
            "claim_id" : self.cache_claim_id,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
            
    def load(self, path):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        self.cache_nli_result = checkpoint["nli"]
        self.cache_claim_id = checkpoint["claim_id"]

    def inference(self, premise, hypothesis):
        tokenized_input_seq_pair = self.tokenizer.encode_plus(premise, hypothesis,
                                                         max_length=max_length,
                                                         return_token_type_ids=True, truncation=True)

        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

        outputs = model(input_ids.cuda(),
                        attention_mask=attention_mask.cuda(),
                        token_type_ids=token_type_ids.cuda(),
                        labels=None)
        # Note:
        # "id2label": {
        #     "0": "entailment",
        #     "1": "neutral",
        #     "2": "contradiction"
        # },

        predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

        max_class = -1
        max_p = -1
        for c, p in enumerate(predicted_probability):
            if p > max_p:
                max_p = p
                max_class = c
        
        return max_class