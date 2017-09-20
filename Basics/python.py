result_string = input()
number_string = input()

result_numbers = []
number_numbers = []
abc=[]
abc.append(result_string.split(" ")[:])

for i in result_string.split(" "):
    result_numbers.append(int(i))
for i in number_string.split(" "):
    number_numbers.append(int(i))

MEX = result_numbers[1]
number_add = 0
number_remove = 0
number_smaller = []
for i in number_numbers:
    if i < MEX:
        number_smaller.append(i)

for i in range(MEX):
    if i not in number_smaller:
        number_add+=1

if MEX in number_numbers:
    number_add +=1

print(number_add)
