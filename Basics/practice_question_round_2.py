user_input = int(input())
def GCD(c,d):
    if d == 0:
        return c
    else:
        return GCD(d, c%d)
first_part = []
def hahaha(local_input):
    for i in range(int(local_input/2),0,-1):
        for j in range(int(local_input/2), local_input):
            if i+j == local_input and i!=j and  GCD(i,j)==1:
                return [i,j]
result = hahaha(user_input)
print(result[0],result[1])