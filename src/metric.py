class ConfussionMatrix:
    
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def add_true_positive(self):
        self.TP += 1
        
    def add_true_negative(self):
        self.TN += 1
        
    def add_false_positive(self):
        self.FP += 1
        
    def add_false_negative(self):
        self.FN += 1
        
    def get_precision(self):
        return self.TP / (self.TP + self.FP)
    
    def get_recall(self):
        return self.TP / (self.TP + self.FN)

    def get_f1(self):
        precision = self.get_precision()
        recall = self.get_recall()

        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

class RetrieverEvaluator:
    
    def __init__(self):
        self.result = []

    def compute(self, retrieval_result, golden_index, k):
        self.result = []
        for rarry, garry in zip(retrieval_result, golden_index):
            rarry = rarry[:k]
            matric = ConfussionMatrix()
            for index in rarry:
                if index in garry:
                    matric.add_true_positive()
                else:
                    matric.add_false_positive()
            for index in garry:
                if index not in rarry:
                    matric.add_false_negative()
            self.result.append(matric)

    def get_precision(self):
        precision = 0
        for metric in self.result:
            precision += metric.get_precision()
        return precision / len(self.result)

    def get_recall(self):
        recall = 0
        for metric in self.result:
            recall += metric.get_recall()
        return recall / len(self.result)

    def get_f1(self):
        f1 = 0
        for metric in self.result:
            f1 += metric.get_f1()
        return f1 / len(self.result)