import os, sys

fp = open ('f','r')

s = 0
l1 = []

for line in fp:
	l = line.strip()
	l1 = l.split(',')

fp.close()

for t in l1: 
	if t != "":
		s = s + int(t)

print s
