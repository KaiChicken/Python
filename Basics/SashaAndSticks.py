input = input()
#input = "1000000000000000000 3"
a = int(input.split(" ")[0])
b = int(input.split(" ")[1])
if int(a/b)%2 == 1:
    print("Yes")
else:
    print("No")