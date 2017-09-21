import csv
import math

class DecisionTree:
    _features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    _labels = ["unacc", "acc"]
    _feature_value = {"buying": ["vhigh", "high", "med", "low"],
                      "maint": ["vhigh", "high", "med", "low"],
                      "doors": ["2", "3", "4", "5more"],
                      "persons": ["2", "4", "more"],
                      "lug_boot": ["small", "med", "big"],
                      "safety": ["low", "med", "high"]}
    train_data = []
    test_data = []

    #retrieve training and test data sets
    with open('car.training.csv') as csvfile:
        csv_data = csv.reader(csvfile, delimiter = ",")
        for row in csv_data:
            train_data.append(row)
    with open('car.test.csv') as csvfile:
        csv_data = csv.reader(csvfile, delimiter=',')
        for row in csv_data:
            test_data.append(row)

    def __int__(self):
        _features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

    #def __int__(self, number_of_attributes, number_of_label_value, ):

    #calculate gini index
    def calculate_gini_index(self, matrix):
        total_sum = (sum(map(sum, matrix)))
        row_gini_index = []
        #find gini index for each row
        for row in matrix:
            row_gini = 1
            if sum(row)==0:
                row_gini=0
            else:
                for column_number in range(len(row)):
                    row_gini -= (row[column_number] / sum(row)) ** 2
            row_gini_index.append(row_gini*sum(row)/total_sum)
        return sum(row_gini_index)

    #calculate information gain
    def calculate_information_gain(self, matrix):
        total_sum = (sum(map(sum,matrix)))
        column_sums = []
        total_entropy = 0
        #find sum for each column
        for column in range (len(matrix[0])):
            column_sums.append(sum(row[column] for row in matrix))
        #find overall entropy
        for i in column_sums:
            if (i > 0):
                total_entropy += (-i/total_sum)*math.log(i/total_sum, 2)
            else:
                total_entropy += 0
        row_entropies = []
        for row in matrix:
            each_entropy = 0
            for i in range (len(row)):
                if (row[i] > 0):
                    each_entropy +=(sum(row)/total_sum)*(-1*row[i]/sum(row)) * math.log(row[i]/sum(row), 2)
                else:
                    each_entropy += 0
            row_entropies.append(each_entropy)
        return total_entropy - sum(row_entropies)

    #find the best feature base on information gain
    def choose_best_feature(self, dataSet, available_features):
        information_gain = []
        gini_index = []
        best_feature = ""
        algorithm_type = "information"
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

    #create the tree base on different method, gini or information gain
    def create_tree(self, tree, dataSet, total_feature):
        isLabel = False
        label = ""
        label_values = []
        remaining_feature = []
        for item in total_feature:
            remaining_feature.append(item)
        #check if all the labels are the same
        for row in dataSet:
            label_values.append(row[-1])
        if all(x == label_values[0] for x in label_values):
            isLabel = True
            label = dataSet[0][-1]
        #check is there is any data to catagorize
        if len(dataSet) == 0:
            label = "unknown"
        #check if there is any unsplited features
        number_of_remaining_attribute = 0
        for item in total_feature:
            if len(item) > 1:
                number_of_remaining_attribute+=1
        #print ("number_of_remaining_attribute", number_of_remaining_attribute)
        if number_of_remaining_attribute == 0:
            isLabel == True
            for row in dataSet:
                label_values = row[-1]
            if all(x == label_values[0] for x in label_values):
                label = dataSet[0][-1]
            elif len(dataSet) == 1:
                label = dataSet[0][-1]
            else:
                #print("total_feature", total_feature)
                print("this should not happen")
                print(dataSet)
                label = "unknown"
        #check if it reaches the leave node
        if isLabel:
            tree.update({label:{}})
        else:
            #find the best feature to split next
            best_feature = self.choose_best_feature(dataSet, remaining_feature)
            best_feature_index = self._features.index(best_feature)
            #build the tree
            tree.update({best_feature:{}})
            remaining_feature[best_feature_index] = ""
            for best_feature_value in self._feature_value[best_feature]:
                tree[best_feature].update({best_feature_value:{}})
                splited_data = self.split_dataSet(dataSet, best_feature, best_feature_value)
                self.create_tree(tree[best_feature][best_feature_value], splited_data, remaining_feature)

    #split the data set for different attribute values
    def split_dataSet(self, dataSet, feature, feature_value):
        updated_data = []
        feature_index = self._features.index(feature)
        for row_number in range(len(dataSet)):
            if dataSet[row_number][feature_index] == feature_value:
                updated_data.append(dataSet[row_number])
        return updated_data

    #print the tree
    def print_tree(self, tree,space):
        for i in range(space):
            print("   ",end="")
        if len(tree) > 0:
            for item in tree:
                print("node:", item)
                self.print_tree(tree[item], space + 1)

    #find estimated class for each row
    def find_label(self, tree, data):
        label = ["acc", "unacc"]
        for item in tree:
            if item in label:
                return item
            elif item in self._features:
                if len(data[self._features.index(item)] ) > 0:
                    data_value = data[self._features.index(item)]
                    data[self._features.index(item)] = ""
                return self.find_label(tree[item][data_value], data)
            else:
                print ("this should not print")

    #fit in dataSet and find the estimated result
    def fit(self, tree, dataSet):
        estimated_label = []
        for row in dataSet:
            estimated_label.append(self.find_label(tree, row))
        return estimated_label

if __name__=="__main__":
    dt = DecisionTree()
    tree = {}
    dt.create_tree(tree, dt.train_data, dt._features)
    dt.print_tree(tree,0)
    estimate_results = dt.fit(tree, dt.train_data)
    actual_results = []
    number_of_correct_result =0
    for row in dt.train_data:
        actual_results.append(row[-1])
    #find the number of correct answers
    for i in range (len(estimate_results)):
        if estimate_results[i] == actual_results[i]:
            number_of_correct_result+=1
    print("")
    print("*******************************************");
    print("***Decision Tree for [data/car.training]***");
    print("*******************************************");
    print("");

    print("");
    print("******************************************");
    print("***Test Results for [data/car.training]***");
    print("******************************************");
    print("");
    print("training_mode gini_index ");
    print("matches ", number_of_correct_result);
    print("test_rows ", len(estimate_results));
    print("overall ",number_of_correct_result / len(estimate_results))
