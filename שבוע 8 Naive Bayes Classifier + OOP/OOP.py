# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 22:46:04 2021

@author: Idan Tobis
"""

class Cars:
# Create and Set the class attributes using constructor (__init__) method
 def __init__(self, brand, speed, height, weight, colour):
  self.brand = brand
  self._speed = speed
  self._height = height
  self.__weight = weight
  self.__colour = colour
  
# Create the methods with simple "print" statement in them
 def activateHorn(self):
  print ( self.brand + " Says: Zambura")

 def _moveForward(self):
  print ( self.brand + " Moving Forward at the speed of: " + self._speed)

 def _moveBack(self):
  print ( self.__colour + " " + self.brand + "  is Moving Backward!")

 def __turnRight(self):
  print ("Turning Right!")

 def __turnLeft(self):
  print ("Turning  Left!")

 def zigzag(self):
     self.__turnRight()
     self.__turnLeft()
     self.__colour = 'scratched'
     print (self.__colour)

# Creates First object car1 and Assign the attributes value of car1 object
car1 = Cars("Toyota", "250km/hr", "1908mm", "2800kg", "Green")

# Accessing the attributes
print (car1.brand)
print (car1._speed)
print (car1._height)
# print (car1.__weight)
# print (car1.__colour)

# Accessing the methds
car1.activateHorn()
car1._moveForward()
car1._moveBack()
# car1.__turnRight()
#car1.__turnLeft()
car1.zigzag()