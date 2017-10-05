
first_line = input()
#first_line = "6 2 1"
input1 = first_line.split(" ")
input1 = list(map(int, input1))
query = []

for i in range (input1[2]):
    query.append(input())
    pass
#query.append("8 12")

deg = 0
def degree(a):
    global deg
    if a < 10:
        return deg
    else:
        deg += 1
        return degree(a/10)

counter = 0
for i in query:
    i = list(map(int, i.split(" ")))
    for j in range(i[0], (i[1]+1)):
        #print(j)
        #print(int(math.log(j,10)), int(math.log(input1[0],10)), input1[1])
        b = degree(input1[0])
        c = degree(j)
        d = b - c
        #print("d", d)
        if d <= input1[1] and d >=0:
            counter+=1

print(counter)