import random
import sys
import os

def addNumber(fNum, lNum):
    sumNum = fNum + lNum
    return sumNum

print(addNumber(1,4))

string = addNumber(1,4)

print('What is your name')
#name = sys.stdin.readline()
#print('Hello', name)

long_string = "I'll catch you if you fall - The Floor"
print(long_string[0:4])
print(long_string[-5:])
print(long_string[:-5])
print(long_string[:4] + " be there")
print("%c is my %s letter and my number %d is %.5f" % ('X', 'favorite', 1, .14))

print(long_string.capitalize())
print(long_string.upper())
print(long_string.lower())
print(long_string.isalpha())
print(long_string.isalnum())
print(long_string.replace("Floor", "Ground"))
print(long_string.strip())
quote_list = long_string.split(" ")
print(quote_list)
