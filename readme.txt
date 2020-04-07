Here is the instructions to run the AESR code. Do not delete any of the folders in this file. If anyone of the folders are deleted make the neccessary change in the code.

Running the code

Use following command:

python aesr.py --ratingdataset sample --friendlist implicit
				or
python aesr.py --ratingdataset sample --friendlist explicit

Order of execution of the code:
By running aesr.py, the following modules will be executed in order with the given input data.
	Module 					Input/output file
1	loading_data.py			rating.txt, friends.txt
2	indexing.py				indexed_data.txt
3	graph_construction.py	user_user.edgelist
4	nrl_using_node2vec.py	out.emb
5	topkuser.py				similar.txt, sorted.txt
6	metrics.py

All these files are generated to verify the working of each module and its correctness

This code is made by combining several modules to understand the workflow. Thus it will take time to complete all the tasks. For large datasets(like epinion) you can split these modules and run them individually and parallely. For smaller datasets like github and movielens 100k this code can be used.

