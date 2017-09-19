import csv
import math
class Test:
    def __init__(self):
        self.foo = 11
        self._bar = 23
        self.__baz = 42

t = Test()
print(t)
print(dir(t))
print(t.foo)
print(t._bar)
print(t._Test__baz)

abc = [[1,2,3,4,1],["a","a","a"]]
bbb = len(abc) - 1
abc[0].insert(1, "b")
abc.insert(0, "b")
print("a number ")
print(abc)

dataSet = []
with open('car.training.csv') as csvfile:
    dataSet = csv.reader(csvfile, delimiter=',')
    #for row in dataSet:
        #print(row)

ccc = [1,2,3,4,1],[1,2,3,4,1],[1,2,3,4,1]
print (sum(map(sum,ccc)))
#print (0*math.log(5))

ddd = []
ddd.append([])
ddd[0].append(100)
#ddd[1] = 1000
print (ddd[0])

if "abc" == "abc":
    print("wtf")


count_matrix = []
number_of_features = 10
for i in range(10):
    count_matrix.append([])
    for j in range(2):
        count_matrix[i].append(0)
print(count_matrix)

dict = {1:2}
dict.update({"a":"b"})
print (dict[1])

_feature_value = {"buying": ["vhigh", "high", "med", "low"],
                  "maint": ["vhigh", "high", "med", "low"],
                  "doors": ["2", "3", "4", "5more"],
                  "persons": ["2", "4", "more"],
                  "lug_boot": ["small", "med", "big"],
                  "safety": ["low", "med", "high"]}
abc = "buying"
_feature_value[abc].append("omg")



aaa = ["aa"]
bbb = ["bb"]
ccc = ["cc"]
abc = []
abc.append(aaa)
aaa.append(bbb)
bbb.append(ccc)
print("what is abc ",abc)


dict = {1:{2:{3:4}}}
dict.update({"a":"b"})
print ("dict: ", dict)
print (len())


ggg = {}
aa = "aa"
ggg.update({aa:{}})
ggg.update({"not ggg":"not ggg"})

V = [{}]
path = {}

a = 10
for i in a:
    V