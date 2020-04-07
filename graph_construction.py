#!/usr/bin/python
import sys

'''This module constructs a user-user network from the indexed rating data and represented in the form of edgelist.'''
def user_user_graph():
	#Provide indexed rating file name which is generated previously.
	file=open("results/indexed_data.txt",'r')
	useritem = {} 
	
	#Provide graph type here. If you wanted to generate a weighted edge list give "W".
	graph_type="UW"

	for line in file:
		line = line.split()
		try:
			useritem[line[0]].append(line[1])
		except Exception:
			useritem[line[0]] = [line[1]]

	#Here a linke between user1 and user2 is created if their item list have atleast one item in common.
	useruser = {}
	for user1 in useritem:
		for user2 in useritem:
			if user1 != user2:
				count = len( list(set(useritem[user1]).intersection(useritem[user2])) )
				if count != 0 :
					if user1 in useruser.keys():
						useruser[user1].append( (user2,count) )
					else:
						useruser[user1] = [(user2, count) ]
			
	if graph_type == "UW":
		f = open("results/user_user.edgelist",'w')
		for thing in useruser:
			for pthing in useruser[thing]:
				f.write("%s %s\n" % (thing, pthing[0]))
	
	#Here weight of the edge is the number of common items.			
	if graph_type == "W":
		f = open("results/user_user.edgelist",'w')
		for thing in useruser:
			for pthing in useruser[thing]:
				f.write( str(thing) +' '+str(pthing[0]) +' '+str(1) +'\n')
	f.close()
	
	return



