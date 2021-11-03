import os

#clear = lambda: os.system("cls")
#clear()

remainder = lambda num: num % 2
print(type(remainder))
print(remainder(5))

product = lambda x,y: x *y
print(product(2,3))

def myfunction(num):
    return lambda x: x * num

result10 = myfunction(10)
#This would be the exact same:
#result10 = lambda x: x * 10
result100 = myfunction(100)


print(result10(9))
print(result100(9))


def myfunc(n):
    return lambda a: a*n

mydoubler = myfunc(2)       #this is n 
mytripler = myfunc(3)       #this is n

print(mydoubler(11))       #this is a
print(mytripler(11))       #this is a

numbers = [2,4,6,8,10,12,14,16]

filtered_list = list(filter(lambda num: (num > 7), numbers))
print(filtered_list)

mapped_list = list(map(lambda num: num %2, numbers))

print(mapped_list)

x = lambda a: a + 10

print(x(5))

x = lambda a,b,c: a + b + c
print(x(5,6,7))

def addition(n):
    return n + n

numbers = [1,2,3,4]

result = map(addition, numbers)
result = map(lambda num: num + num)
print(list(result))

