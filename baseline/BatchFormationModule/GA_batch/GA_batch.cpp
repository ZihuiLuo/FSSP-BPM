#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cstdlib>

using namespace std;
ofstream outfile;

#define vertex_num 300 //the number of clients
#define pop_size 200  //每一代种群的个体数
#define max_generation 200
//#define best_solution 1618.36 //the best solution in all generations
#define capacity 180 //the capacity of each truck
string input_data[vertex_num]; //用来存储从input.txt获取的所有数据
clock_t Start, Finish; //计算运行时间

struct Vertex {
	double num;
	double a;
	double b;
	double c;
	double d;
	double demand;
};

//int vertex_num;							//the number of clients
//int capacity;							//the capacity of each truck	

//int pop_size;							//the population size
double transform_probability = 0.200;	//the probability of individual transform
vector<Vertex> vertexs;					//the vector of clients vertexs	
vector<vector<double> > population;		//the set of individuals in the population

double cur_best_distance;				//the best solution in current generation
vector<double> cur_best_individual;		//the best individual in current generation 

double best_solution;					//the best solution	in all generations
vector<double> best_individual;			//the best individual in all generations

//randomly initialize the population, according to the pop_size
void initialize_population() {
	//truely random number
	srand((double)time(0) + rand());
	vector <double> sort_vector;
	for (int i = 1; i <= vertex_num; i++) {
		sort_vector.push_back(i);
	}
	for (int i = 0; i < pop_size; i++) {
		vector <double> temp = sort_vector;
		int cnt = 0;
		while (cnt < vertex_num * 2) {
			double first = rand() % vertex_num;
			double second = rand() % vertex_num;
			while (first == second) {
				second = rand() % vertex_num;
			}
			double exchange = temp[first];
			temp[first] = temp[second];
			temp[second] = exchange;

			cnt++;
		}
		population.push_back(temp);
	}
}

vector <double> add_depot_coordinate(vector <double> individual) {
	vector <double> add_depot;
	add_depot.push_back(0);
	double capacity_used = 0;

	//add zero depot in the individual vector, like: {0,3,2,0,1,0}
	for (int i = 0; i < individual.size(); i++) {
		if (capacity_used + vertexs[individual[i]].demand < capacity) {
			add_depot.push_back(individual[i]);
			capacity_used += vertexs[individual[i]].demand;
		}
		else {
			add_depot.push_back(0);
			add_depot.push_back(individual[i]);
			capacity_used = vertexs[individual[i]].demand;
		}
	}
	add_depot.push_back(0);
	return add_depot;
}

double caculate_individual_distance(vector <double> individual) {
	vector <double> add_depot = add_depot_coordinate(individual);

	//caculate the distance of the new vector with zero
	double distance = 0;
	for (int i = 1; i < add_depot.size(); i++) {
		double first_a = vertexs[add_depot[i - 1]].a;
		double first_b = vertexs[add_depot[i - 1]].b;
		double first_c = vertexs[add_depot[i - 1]].c;
		double first_d = vertexs[add_depot[i - 1]].d;
		double first_e = vertexs[add_depot[i - 1]].e;
		double first_f = vertexs[add_depot[i - 1]].f;
		double first_h = vertexs[add_depot[i - 1]].h;
		double first_i = vertexs[add_depot[i - 1]].i;
		double first_art = vertexs[add_depot[i - 1]].art;
		double first_thickness = vertexs[add_depot[i - 1]].thickness;
		double first_width = vertexs[add_depot[i - 1]].width;
		double first_date = vertexs[add_depot[i - 1]].date;
		double first_priority = vertexs[add_depot[i - 1]].priority;
		double second_a = vertexs[add_depot[i]].a;
		double second_b = vertexs[add_depot[i]].b;
		double second_c = vertexs[add_depot[i]].c;
		double second_d = vertexs[add_depot[i]].d;*/
		double second_e = vertexs[add_depot[i]].e;
		double second_f = vertexs[add_depot[i]].f;
		double second_h = vertexs[add_depot[i]].h;
		double second_i = vertexs[add_depot[i]].i;
		double second_art = vertexs[add_depot[i]].art;
		double second_thickness = vertexs[add_depot[i]].thickness;
		double second_width = vertexs[add_depot[i]].width;
		double second_date = vertexs[add_depot[i]].date;
		double second_priority = vertexs[add_depot[i]].priority;


		double edge_length = 0.2 * pow((first_a - second_a), 2) + 0.2 * pow((first_b - second_b), 2) + 0.2 * pow((first_c - second_c), 2)
			+ 0.2 * pow((first_d - second_d), 2) + 0.2 * pow((first_e - second_e), 2) + 0.2 * pow((first_f - second_f), 2)
			+ 0.2 * pow((first_h - second_h), 2) + 0.2 * pow((first_i - second_i), 2) + 0.2 * pow((first_art - second_art), 2)
			+ 0.2 * pow((first_thickness - second_thickness), 2) + 0.16 * pow((first_width - second_width), 2)
			+ 0.16 * pow((first_date - second_date), 2) + 0.08 * pow((first_priority - second_priority), 2);
		edge_length = pow(edge_length, 0.5);
		if ((add_depot[i] != 0) && (add_depot[i - 1] != 0)) {
			distance += edge_length;
		}
	}
	return distance * 1.0;
}

void choose_individuals() {
	double population_fitness = 0;
	vector <double> individuals_fitness;
	vector <double> accumulate_probability;

	//caculate all individuals' fitness, record them and caculate the fitness of whole population
	cur_best_distance = 99999999;
	for (int i = 0; i < pop_size; i++) {
		double cur_distance = caculate_individual_distance(population[i]);
		if (cur_distance < cur_best_distance) {
			cur_best_distance = cur_distance;
			cur_best_individual = population[i];
		}
		if (cur_distance < best_solution) {
			best_solution = cur_distance;
			best_individual = population[i];
		}

		//shorter the distance is, larger fitness the individual has.
		double cur_fitness = 1.0 / cur_distance;
		individuals_fitness.push_back(cur_fitness);
		population_fitness += cur_fitness;
	}

	//caculate the chosen probability of each individual, and accumulate them in 'accumulate_probability'
	for (int i = 0; i < pop_size; i++) {
		double probability = individuals_fitness[i] / population_fitness;
		if (i != 0) {
			accumulate_probability.push_back(accumulate_probability[i - 1] + probability);
		}
		else {
			accumulate_probability.push_back(probability);
		}
	}

	//accroding to the 'accumulate_probability', choose the number of 'pop_size' new individuals of population
	srand((double)time(0) + rand());
	vector <vector<double> > newPopulation;
	for (int i = 0; i < pop_size; i++) {
		double randomly = rand() % 10000 * 0.0001;
		//50% of chance push back the best individual in the population directly
		if (i == 0 && randomly < 0.5) {
			newPopulation.push_back(best_individual);
		}
		else {
			for (int j = 0; j < pop_size; j++) {
				if (randomly < accumulate_probability[j]) {
					newPopulation.push_back(population[j]);
					break;
				}
			}
		}
	}
	//use the new population to replace the old one
	population.clear();
	population = newPopulation;
}

void individuals_transform() {
	srand((double)time(0) + rand());
	//use the operation of 'Inver-Over'
	for (int i = 0; i < pop_size; i++) {
		double start_num = rand() % vertex_num + 1;
		double end_num;
		double start_index, end_index;
		for (int j = 0; j < vertex_num; j++) {
			if (population[i][j] == start_num) {
				start_index = j;
				break;
			}
		}

		//two method of operation
		double transform = rand() % 10000 * 0.0001;
		//likely mutation
		if (transform < transform_probability) {
			end_num = rand() % vertex_num + 1;
			while (start_num == end_num) {
				end_num = rand() % vertex_num + 1;
			}
			for (int j = 0; j < vertex_num; j++) {
				if (population[i][j] == end_num) {
					end_index = j;
					break;
				}
			}
		}
		//likely crossover
		else {
			double other = rand() % pop_size;
			while (other == i) {
				other = rand() % pop_size;
			}
			for (int j = 0; j < vertex_num; j++) {
				if (population[other][j] == start_num) {
					if (j == vertex_num - 1) {
						end_num = population[other][j - 1];
					}
					else {
						end_num = population[other][j + 1];
					}
					break;
				}
			}
			for (int j = 0; j < vertex_num; j++) {
				if (population[i][j] == end_num) {
					end_index = j;
					break;
				}
			}
		}
		if (start_index > end_index) {
			double temp = start_index;
			start_index = end_index;
			end_index = temp;
		}

		//reverse the vertexs in individual between start_index and end_index
		vector <double> reverse;
		for (int j = start_index + 1; j <= end_index; j++) {
			reverse.push_back(population[i][j]);
		}
		double reverse_index = reverse.size() - 1;
		for (int j = start_index + 1; j <= end_index; j++) {
			population[i][j] = reverse[reverse_index];
			reverse_index--;
		}
	}
}


int main() {
	Start = clock();
	ifstream input;
	input.open("../data/input_data_300.txt");
	outfile.open("../input_data_300_output.txt");
	if (input.is_open()) {
		best_solution = 9999999;

		//initialize the vector
		Vertex temp;
		for (int ii = 0; ii <= vertex_num; ii++) {
			vertexs.push_back(temp);
		}
		//depot coordinate (0,0,0,0)
		vertexs[0].a = 0;
		vertexs[0].b = 0;
		vertexs[0].c = 0;
		vertexs[0].d = 0;

		for (int ii = 1; ii <= vertex_num; ii++) {
			try {
				//input >> vertexs[i].num;
				vertexs[ii].num = ii;
				input >> vertexs[ii].a;
				input >> vertexs[ii].b;
				input >> vertexs[ii].c;
				input >> vertexs[ii].d;
				input >> vertexs[ii].demand;
			}
			catch (string & err) {
				cout << err << endl;
			}
		}
	}
	else {
		cout << "open file failed!" << endl;
	}

	initialize_population();

	int generation = 1;
	while (generation < max_generation) {
		choose_individuals();
		individuals_transform();

		//std::cout << "the best in current generation: " << cur_best_distance << endl;
		generation++;
	}

	std::cout << endl;
	std::cout << "The cost of the best solution is: " << best_solution << endl;
	std::cout << endl;
	std::cout << "The vertexs' order of the best solution is: " << endl;

	int iii = 0;
	vector <double> result = add_depot_coordinate(best_individual);
	for (int ii = 0; ii < result.size(); ii++) {
		std::cout << result[ii] << " ";
		if (result[ii] != 0) {
			outfile << "作业号：" << result[ii] << " " << vertexs[result[ii]].a << " " << vertexs[result[ii]].b << " " << vertexs[result[ii]].c << " " << vertexs[result[ii]].d << endl; /*" " << vertexs[result[ii]].demand <<*/
		}
		if ((ii != result.size() - 1) && (result[ii + 1] == 0)) {
			std::cout << endl;
			outfile << endl;
			iii++;
		}
	}

	int iiii = 0;
	std::cout << "[";
	outfile << "[";
	for (int ii = 0; ii < result.size(); ii++) {
		if (result[ii] != 0) {
			/*std::cout << "[" << vertexs[result[ii]].id << "], ";
			outfile << "[" << vertexs[result[ii]].id << "], ";*/
			std::cout << "[" << result[ii] << "], ";
			outfile << "[" << result[ii] << "], ";
		}
		if ((ii != result.size() - 1) && (result[ii + 1] == 0)) {
			iiii++;
			if (iiii != iii) {
				std::cout << "[0.0], ";
				outfile << "[0.0], ";
			}
			else {
				std::cout << "[0.0]";
				outfile << "[0.0]";
			}
		}
	}
	std::cout << "]";
	outfile << "]";
	std::cout << endl;
	outfile << endl;
	std::cout << "批次数量： " << iii << endl;
	outfile << "批次数量： " << iii << endl;
	std::cout << "difference value: " << best_solution << endl;
	outfile << "difference value: " << best_solution << endl;
	double a = 0.1, b = 0.5;
	std:; cout << "参数α = " << a << "     " << "β = " << b << endl;
	outfile << "参数α = " << a << "     " << "β = " << b << endl;
	std::cout << "reward: " << a * best_solution + b * 2 * iii << endl;
	outfile << "reward: " << a * best_solution + b * 2 * iii << endl;
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	outfile << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << endl;

	system("Pause");
	return 0;
}
