number_of_node_string = input()
number_of_node = int(number_of_node_string)
input_strings = []
#input_strings = ["1 2", "2 3", "3 4", "4 5"]
for i in range(number_of_node-1):
    input_strings.append(input())

input_matrix = []
for i in range(len(input_strings)):
    input_matrix.append([])
    for j in input_strings[i].split(" "):
        input_matrix[i].append(int(j))

pool_1 = []
pool_2 = []
unknown_pool = []
small_pools = []
for i in input_matrix:
    if i[0]


if (len(pool_1)*len(pool_2)-number_of_node+1)>0:
    result = len(pool_1)*len(pool_2)-number_of_node+1
else:
    result = 0
print(result)

