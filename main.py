import os
import json
import pickle
import argparse
from src.dataset import *
from src.encoder import *
from src.metric import *
from src.retrieval import *

parser = argparse.ArgumentParser()
parser.add_argument("path_data", help="path of data. eg. ./dataset/train.json", type=str)
parser.add_argument("path_out", help="root directoy for saving result. eg. ./save", type=str)
parser.add_argument("-k", help="to retrieve K claims.", default=100, type=int)
parser.add_argument("-s", "--strategy", help="simialarity strategy", default="srl", type=str)
parser.add_argument("-e", "--encoder_name", help="sentence encoder for query sentence encoding", default="sbert", type=str)
parser.add_argument("-t", "--prepend_title_sentence", help="prepend title to the sentence or not", default=True, type=bool)
parser.add_argument("-f", "--prepend_title_frame", help="prepend title to frames or not", default=True, type=bool)
args = parser.parse_args()

if __name__ == "__main__":


    with open(args.path_data) as f:
        data = json.load(f)

    dataset = FEVERDataset(data)
    samples = dataset.get_evidence()
    claims = dataset.get_claims()
    golden_index = dataset.get_golden_index()

    if args.encoder_name == "sbert":
        encoder_and_database_factory = SBERTEncoderAndDatabaseFactory()
    elif args.encoder_name == "simcse":
        encoder_and_database_factory = SimCSEEncoderAndDatabaseFactory()
    elif args.encoder_name == "use":
        encoder_and_database_factory = USEEncoderAndDatabaseFactory()
    elif args.encoder_name == "bm25":
        encoder_and_database_factory = BM25EncoderAndDatabaseFactory()
    else:
        raise Exception("Encoder %s does not exist.".format(args.encoder_name))

    if args.strategy == "baseline":
        similarity_strategy = BaselineSimilarityStrategy()
    elif args.strategy == "srl":
        similarity_strategy = SRLSimilarityStrategy()
    else:
        raise Exception("Method %s does not exist.".format(arg_method))

    retriever = ClaimRetriever(claims, encoder_and_database_factory, similarity_strategy)
    retrieval_result = retriever.retrieve(samples,
                                          k=args.k, 
                                          prepend_title_sentence=args.prepend_title_sentence,
                                          prepend_title_frame=args.prepend_title_frame)

    with open(os.path.join(args.path_out, f"{args.encoder_name}-{args.strategy}.pkl"), "wb") as f:
        pickle.dump(retrieval_result, f)


