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

#测试
vrp50 = TaskVRP(task_name = 'vrp',
			  input_dim=6,
			  n_nodes=51,
			  n_cust = 50,
			  decode_len=100,
			  capacity=10000000,
			  demand_max=50)
task_lst['vrp50'] = vrp50

# VRP100
vrp100 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=101,
			  n_cust = 100,
			  decode_len=140,
			  capacity=10000000,
			  demand_max=9)
task_lst['vrp100'] = vrp100

# VRP200
vrp200 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=201,
			  n_cust = 200,
			  decode_len=300,
			  capacity=10000000,
			  demand_max=9)
task_lst['vrp200'] = vrp200

# VRP300
vrp300 = TaskVRP(task_name = 'vrp',
			  input_dim=3,
			  n_nodes=301,
			  n_cust = 300,
			  decode_len=500,
			  capacity=10000000,
			  demand_max=9)
task_lst['vrp300'] = vrp300