from collections import namedtuple

# task specific params
TaskVRP = namedtuple('TaskVRP', ['task_name', 
						'input_dim',
						'n_nodes' ,
						'n_cust',
						'decode_len',
						'capacity',
						'demand_max'])


task_lst = {}

# VRP10
# vrp10 = TaskVRP(task_name = 'vrp',
# 			  input_dim=3,
# 			  n_nodes=11,
# 			  n_cust = 10,
# 			  decode_len=16,
# 			  capacity=20,
# 			  demand_max=9)
# task_lst['vrp10'] = vrp10


vrp100 = TaskVRP(task_name = 'vrp',
			  input_dim=6,
			  n_nodes=101,
			  n_cust = 100,
			  decode_len=160,
			  capacity=1000000000,
			  demand_max=50)
task_lst['vrp100'] = vrp100

# # VRP100
# vrp100 = TaskVRP(task_name = 'vrp',
# 			  input_dim=3,
# 			  n_nodes=101,
# 			  n_cust = 100,
# 			  decode_len=140,
# 			  capacity=50,
# 			  demand_max=9)
# task_lst['vrp100'] = vrp100