#b = int(input())
b=3
a=int(b/2)
prime =[]
result = []
if b >=3 and b <= 1000:
    for i in range(3,a):
        isPrime = True
        for j in range(2,i):
            if i%j == 0:
                isPrime = False
        if isPrime:
            prime.append(i)
    prime.insert(0,2)
    prime.insert(0,1)
    print(prime)
    for i in prime[::-1]:
        isPrime = True
        for j in range(2, (b-i)):
            if (b-i)%j == 0:
                isPrime = False
                #print("not prime", (b-i))
        #print(i,(b-i),isPrime)
        if isPrime:
            if i != (b-i):
                result.append(i)
                result.append(b-i)
                print(min(result),max(result))
                break

input = 34
def GCD(c,d):
    if d == 0:
        return c
    else:
        return GCD(d, c%d)
first_part = []
def hahaha():
    for i in range(int(input/2),0,-1):
        for j in range(int(input/2), input):
            if i+j == input and i!=j and  GCD(i,j)==1:
                return [i,j]
print(hahaha())