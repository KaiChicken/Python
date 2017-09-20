input_length = int(input())
#input_length = 3
#sequence_string = "1 3 5"
sequence_string = input()
sequence = []
for i in sequence_string.split(" "):
    sequence.append(int(i))

isRight = False
if input_length%2==1:
    if sequence[0]%2 == 1 and sequence[-1]%2 == 1:
        isRight = True

if isRight:
    print("Yes")
else:
    print("No")