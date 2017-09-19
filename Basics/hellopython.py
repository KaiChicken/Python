import random
import sys
import os

grocery_list = ['Juice', 'Tomatoes', 'Potatoes', 'Bananas']

print('First item', grocery_list[0])

grocery_list[0] = "Green Juice"
print('First Item', grocery_list[0])

print(grocery_list[1:3])

other_events = ['Wash Car', 'Pick Up Kids', 'Cash Check']

to_do_list = [other_events, grocery_list]
print(to_do_list)

print((to_do_list[1][1]))

grocery_list.append('Onions')
print(to_do_list)

print("g list",grocery_list)
grocery_list.insert(1, "Pickle")


#grocery_list.remove("Pickle")

del grocery_list[4]
print("g list",grocery_list)
grocery_list.reverse()
to_do_list.sort()
print(to_do_list)



to_do_list2 = other_events + grocery_list
print(to_do_list2)
print(len(to_do_list2))
print(max(to_do_list2))
print(min(to_do_list2))

a = [["a", "b","c","d","e"],["f","g","h"]]
classL = [example[-1] for example in a]
print (classL)