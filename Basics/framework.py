def createDataSet():


    def calculateError(self):

    def classify(self, inputTree, featureLabels, testVec):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        featureIndex = featureLabels.index(firstStr)
        key = testVec[featureIndex]
        valueOfFeature = secondDict[key]
        if isinstance(valueOfFeature, dict):
            classLabel = classify(valueOfFeature, featureLabels, testVec)
        else:
            classLabel = valueOfFeature
        return classLabel


    def splitDataSet(self, dataSet, bestFeature, value):

    def majorityCnt(self, classList):

    def calculateGiniIndex(self):

    def calculateInformationGain(self):

        '''
           def calculateEntropy(self, count, attributeIndex):
               label_frequency = {}
               data_entropy = 0
               attribute_frequency = {}

               for count in data:
                   if(label_frequency.has_key(count[-1])):
                       label_frequency[count[]]
               count1 = self._labels.count("unacc")
               count2 = self._labels.count("acc")
               totalEntropy =


           def createTree(self, dataSet, labels):
               #extract data
               classList = [example[-1] for example in dataSet]
               if classList.count(classList[0]) == len(classList):
                   return classList[0]
               if len(dataSet[0]) == 1:
                   return majorityCnt(classList)
               bestFeature = chooseBestFeature(dataSet)
               bestFeatureLabel = labels[bestFeature]

               #build a tree
               myTree = {bestFeatureLabel: {}}
               del (labels[bestFeature])
               featureValues = [example[bestFeature] for example in dataSet]
               uniqueVals = set(featureValues)
               for value in uniqueVals:
                   subLabels = labels[:]
                   myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature, value), subLabels)
               return myTree


           def bestFeatureBasedOnInformationGain(self, dataSet):
               for i in range(len(dataSet[0]))
                   eachColumn = dataSet[i] for row in dataSet

       '''

