import csv
import math
import numpy as np
import pandas as pd

class DecisionTreeSimplified:
    _features = ["buying", "maint", "doors", "persons", "lug_boot", "safety","labels"]
    _labels = ["unacc", "acc"]
    _feature_value = {"buying": ["vhigh", "high", "med", "low"],
                      "maint": ["vhigh", "high", "med", "low"],
                      "doors": ["2", "3", "4", "5more"],
                      "persons": ["2", "4", "more"],
                      "lug_boot": ["small", "med", "big"],
                      "safety": ["low", "med", "high"]}
    train_data = pd.read_csv('car.training.csv',header =None, names = _features)
    test_data = pd.read_csv('car.test.csv',header =None, names = _features)
    print(train_data)
    print(test_data)

    #calculate gini index
    def calculate_gini_index(self, matrix):
        row_gini_index = []
        #find gini index for each row
        for row in matrix:
            row_gini = 1
            if sum(row)==0:
                row_gini=0
            else:
                for column_number in range(len(row)):
                    row_gini -= (row[column_number] / sum(row)) ** 2
            row_gini_index.append(row_gini*sum(row)/np.sum(matrix))
        return sum(row_gini_index)

    practice_data = [[1,3],
                     [8,0],
                     [1,7]]
    print(calculate_gini_index("DecisionTreeSimplified",practice_data))

    #find the best feature base on information gain
    def choose_best_feature(self, dataSet, available_features, algorithm_type = "gini"):
        information_gain = []
        gini_index = []
        for feature in available_features:
            if len(feature) > 1:
                index_of_feature = self._features.index(feature)
                feature_values = self._feature_value[feature]
                count_matrix = []
                for i in range (len(feature_values)):
                    count_matrix.append([])
                    for j in range (len(self._labels)):
                        count_matrix[i].append(0)
                for row in dataSet:
                    for value_index in range(len(feature_values)):
                        if row[index_of_feature] == feature_values[value_index]:
                            for label_index in range (len(self._labels)):
                                if row[-1] == self._labels[label_index]:
                                    count_matrix[value_index][label_index] += 1
                information_gain.append(self.calculate_information_gain(count_matrix))
                gini_index.append(self.calculate_gini_index(count_matrix))
            else:
                gini_index.append(1)
                information_gain.append(0)
        if algorithm_type == "gini":
            best_feature = self._features[gini_index.index(min(gini_index))]
        elif algorithm_type == "information":
            best_feature = self._features[information_gain.index(max(information_gain))]
        return best_feature