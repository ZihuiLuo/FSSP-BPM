#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <fstream>
#include <cmath>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <iterator>
#include <deque>
#include <cstdio>
#include <random>
using namespace std;
ofstream outfile;

int ordernumber;    //工序数
int parallel;       //每道工序的并行机个数
int machinenumber;  //机器的总数（等于每道工序的并行机个数×工序数）
int workpiecesnumber;  //订单总数

//GA算法相关参数
#define populationnumber 200  //每一代种群的个体数   群体大小：20~100
double crossoverrate = 0.6;            //交叉概率   交叉概率：0.4~0.99
double mutationrate = 0.05;             //变异概率  变异概率：0.0001~0.1
int G = 200;                        //循环代数200   遗传算法的终止进化代数：100~500

double fits[populationnumber];//存储每一代种群每一个个体的适应度，便于进行选择操作；
int makespan;    //总的流程加工时间；

//ACO算法相关参数
#define Ants 3 * workpiecesnumber / 2			    //蚂蚁的个数,最好为1.5倍订单数
#define I 200				//最大迭代次数
#define Alpha 1				//表征信息素重要程度的参数
#define Beta 5				//表征启发式因子重要程度的参数
#define Rho 0.1				//信息素蒸发系数
#define Q 100				//信息素增加强度系数

//TS算法
vector<vector<int> > Tabu;  //禁忌表，存储蚂蚁选择过的订单
vector<vector<double> > Dist; //存储每个订单的完成时间
vector<vector<double> > Eta;	 //表示启发式因子，为Dist[][]中距离的倒数
vector<vector<double> > Tau;		//表示两两节点间信息素浓度
vector<vector<double> > DeltaTau;		//表示两两节点间信息素的变化量
double T_best[I];			//存储每次迭代的路径的最短长度
double T_ave[I];			//存储每次迭代的路径的平均长度
vector<vector<int> > R_best;			//存储每次迭代的最佳路线
vector<double> T;    //存储每次迭代中各只蚂蚁选择的方案的makesapn

//ABC算法相关参数
double xmax = workpiecesnumber, xmin = 1;
std::default_random_engine random1((unsigned int)time(NULL));
std::uniform_real_distribution<double> u(0, 1); //随机数分布对象
std::uniform_real_distribution<double> u1(-1, 1); //随机数分布对象

//所有算法共有的与加工有关的参数
clock_t Start, Finish1, Finish; //计算运行时间
#define populationnumber1 1
#define display 1*24*2 //输出结果打印，目前定义以15minute为一个小间隔，两天为一个周期
vector< vector < vector<int> > > usetime; //第几个订单第几道工序的第几个机器的加工用时；
vector<vector<int> > machinetime;  //第几道工序的第几台并行机器的统计时间；
vector<vector<string> > machinetime_1;
vector<vector<vector<int> > > starttime;//第几个订单第几道工序在第几台并行机上开始加工的时间；
vector<vector<vector<string> > > starttime_2;
vector<string> deliverytime;//订单的交货时间
vector<vector<vector<int> > > finishtime;//第几个订单第几道工序在第几台并行机上完成加工的时间；
vector<vector<vector<string> > > finishtime_1;
int ttime[populationnumber];      //个体的makespan；
vector<vector<int> > a;//第几代的染色体顺序，即订单加工顺序；
vector<vector<int> > a1;
vector<string> times1; //用来存储从input.txt获取的所有数据
int times2[1]; //用来存储所有订单开始时间中的最小值
vector<int> times3; //用来存储所有订单的开始时间
string times4[1]; //用来存储最小开始时间的tm结构
vector<int> times5; //存储每个订单相对于最小开始时间的值
int flg7;   //暂时存储流程加工时间；
vector<int> finishtime1; //记录订单最后一道工序的完成时间
vector<int> deliverytime1; //计算订单交货时间与开始时间的差值


//GA函数实现
int initialization()   //初始化种群；
{
	for (int i = 0; i < populationnumber; i++)     //首先生成一个订单个数的全排列的个体；
		for (int j = 0; j < workpiecesnumber; j++)
		{
			a[i][j] = j + 1;
		}

	for (int i = 0; i < populationnumber; i++)     //将全排列的个体中随机选取两个基因位交换，重复订单个数次，以形成随机初始种群；
		for (int j = 0; j < workpiecesnumber; j++)
		{
			int flg1 = rand() % workpiecesnumber;
			int flg2 = rand() % workpiecesnumber;
			int flg3 = a[i][flg1];
			a[i][flg1] = a[i][flg2];
			a[i][flg2] = flg3;
		}
	return 0;
}

int fitness(int c)   //计算适应度函数，c代表某个体；
{
	int totaltime;      //总的加工流程时间（makespan）；
	vector<int> temp1(workpiecesnumber, 0);
	vector<int> temp2(workpiecesnumber, 0);
	vector<int> temp3(workpiecesnumber, 0);

	for (int j = 0; j < workpiecesnumber; j++)   //temp1暂时存储个体c的基因序列，以便进行不同流程之间的加工时记录工件加工先后顺序；
	{
		temp1[j] = a[c][j];
	}

	for (int i = 0; i < ordernumber; i++)
	{
		for (int j = 0; j < workpiecesnumber; j++)  //该循环的目的是通过比较所有机器的当前工作时间，找出最先空闲的机器，便于新的工件生产；
		{
			int m = machinetime[i][0];        //先记录第i道工序的第一台并行机器的当前工作时间；
			int n = 0;
			for (int p = 0; p < parallel; p++) //与其他并行机器进行比较，找出时间最小的机器；
			{
				if (usetime[j][i][p] > 0) {
					if (m > machinetime[i][p] && machinetime[i][p] >= 0)
					{
						m = machinetime[i][p];
						n = p;
					}
				}
			}
			int q = temp1[j]; //按顺序提取temp1中的工件号，对工件进行加工；
			starttime[q - 1][0][n] = times5[q - 1];
			int secondmax = max(starttime[q - 1][0][n], temp3[j]);
			starttime[q - 1][i][n] = max(secondmax, machinetime[i][n]);  //开始加工时间取该机器的当前时间和该工件上一道工序完工时间的最大值；
			machinetime[i][n] = starttime[q - 1][i][n] + usetime[q - 1][i][n]; //机器的累计加工时间等于机器开始加工的时刻，加上该工件加工所用的时间；
			finishtime[q - 1][i][n] = machinetime[i][n];                 //工件的完工时间就是该机器当前的累计加工时间；
			temp2[j] = finishtime[q - 1][i][n];       //将每个工件的完工时间赋予temp2，根据完工时间的快慢，便于决定下一道工序的工件加工顺序；
		}

		vector<int> flg2(workpiecesnumber, 0);  //生成暂时数组，便于将temp1和temp2中的工件重新排列；
		for (int s = 0; s < workpiecesnumber; s++)
		{
			flg2[s] = temp1[s];
		}

		for (int e = 0; e < workpiecesnumber - 1; e++)
		{
			for (int ee = 0; ee < workpiecesnumber - 1 - e; ee++) // 由于temp2存储工件上一道工序的完工时间，在进行下一道工序生产时，按照先完工先生产的
			{                                            //原则，因此，该循环的目的在于将temp2中按照加工时间从小到大排列，同时temp1相应进行变换来记录temp2中的工件号；
				if (temp2[ee] > temp2[ee + 1])
				{
					int flg5 = temp2[ee];
					int flg6 = flg2[ee];
					temp2[ee] = temp2[ee + 1];
					flg2[ee] = flg2[ee + 1];
					temp2[ee + 1] = flg5;
					flg2[ee + 1] = flg6;
				}
			}
		}
		for (int e = 0; e < workpiecesnumber; e++)    //更新temp1，temp2的数据，开始下一道工序；
		{
			temp1[e] = flg2[e];
			temp3[e] = temp2[e];
		}
	}
	totaltime = 0;
	for (int i = 0; i < parallel; i++) //比较最后一道工序机器的累计加工时间，最大时间就是该流程的加工时间；
		if (totaltime < machinetime[ordernumber - 1][i])
		{
			totaltime = machinetime[ordernumber - 1][i];
			//cout<<totaltime<<"   ";
		}
	for (int i = 0; i < workpiecesnumber; i++)  //将数组归零，便于下一个个体的加工时间统计；
		for (int j = 0; j < ordernumber; j++)
			for (int t = 0; t < parallel; t++)
			{
				starttime[i][j][t] = 0;
				finishtime[i][j][t] = 0;
				machinetime[j][t] = 0;
				starttime[i][0][t] = times5[i];
			}

	makespan = totaltime;
	fits[c] = 1.000 / makespan;          //将makespan取倒数作为适应度函数；
										 //cout<<fits[c]<<"    ";
	return 0;
}

int gant(int c)                   //该函数是为了将最后的结果便于清晰明朗的展示并做成甘特图，对问题的结果以及问题的解决并没有影响；
{
	int totaltime;
	vector<int> temp1(workpiecesnumber, 0); //加工顺序
	vector<int> temp2(workpiecesnumber, 0); //上一步骤的完工时间
	vector<int> temp3(workpiecesnumber, 0);

	for (int j = 0; j < workpiecesnumber; j++)
	{
		temp1[j] = a[c][j];
	}

	for (int i = 0; i < ordernumber; i++)
	{
		for (int j = 0; j < workpiecesnumber; j++)
		{
			int m = machinetime[i][0];
			int n = 0;
			for (int p = 0; p < parallel; p++) //找出时间最小的机器；
			{
				if (usetime[j][i][p] > 0) {
					if (m > machinetime[i][p] && machinetime[i][p] >= 0)
					{
						m = machinetime[i][p];
						n = p;
					}
				}
			}
			int q = temp1[j];
			starttime[q - 1][0][n] = times5[q - 1];
			int secondmax = max(starttime[q - 1][0][n], temp3[j]);
			starttime[q - 1][i][n] = max(secondmax, machinetime[i][n]);
			machinetime[i][n] = starttime[q - 1][i][n] + usetime[q - 1][i][n];
			finishtime[q - 1][i][n] = machinetime[i][n];
			temp2[j] = finishtime[q - 1][i][n];
		}

		vector<int> flg2(workpiecesnumber, 0);
		for (int s = 0; s < workpiecesnumber; s++)
		{
			flg2[s] = temp1[s];
		}
		for (int e = 0; e < workpiecesnumber - 1; e++)
		{
			for (int ee = 0; ee < workpiecesnumber - 1 - e; ee++)
			{
				if (temp2[ee] > temp2[ee + 1])
				{
					int flg5 = temp2[ee];
					int flg6 = flg2[ee];
					temp2[ee] = temp2[ee + 1];
					flg2[ee] = flg2[ee + 1];
					temp2[ee + 1] = flg5;
					flg2[ee + 1] = flg6;
				}
			}
		}

		for (int e = 0; e < workpiecesnumber; e++)
		{
			temp1[e] = flg2[e];
			temp3[e] = temp2[e];
		}
	}
	totaltime = 0;
	for (int i = 0; i < parallel; i++)
		if (totaltime < machinetime[ordernumber - 1][i])
		{
			totaltime = machinetime[ordernumber - 1][i];
		}
	cout << "Makespan = " << totaltime << endl;
	outfile << "Makespan = " << totaltime << endl;
	flg7 = totaltime;
	return 0;
}
/////////////////////////////////////////////////////////////

int select()
{
	double roulette[populationnumber + 1] = { 0.00 };	//记录轮盘赌的每一个概率区间；
	double pro_single[populationnumber];			//记录每个个体出现的概率，即个体的适应度除以总体适应度之和；
	double totalfitness = 0.00;                	//种群所有个体的适应度之和；
	vector<vector<int>> a11(populationnumber, vector<int>(workpiecesnumber)); //存储a中所有个体的染色体；

	for (int i = 0; i < populationnumber; i++)     		//计算所有个体适应度的总和；
	{
		totalfitness = totalfitness + fits[i];
	}

	for (int i = 0; i < populationnumber; i++)
	{
		pro_single[i] = fits[i] / totalfitness;   	//计算每个个体适应度与总体适应度之比；
		roulette[i + 1] = roulette[i] + pro_single[i]; //将每个个体的概率累加，构造轮盘赌；
	}

	for (int i = 0; i < populationnumber; i++)
	{
		for (int j = 0; j < workpiecesnumber; j++)
		{
			a11[i][j] = a[i][j];               //a11暂时存储a的值；
		}
	}

	for (int i = 0; i < populationnumber; i++)
	{
		int a2;   //当识别出所属区间之后，a2记录区间的序号；
		double p = rand() % (1000) / (double)(1000);
		for (int j = 0; j < populationnumber; j++)
		{
			if (p >= roulette[j] && p < roulette[j + 1])
				a2 = j;
		}
		for (int m = 0; m < workpiecesnumber; m++)
		{
			a[i][m] = a11[a2][m];
		}
	}
	return 0;
}

int crossover()
/*种群中的个体随机进行两两配对，配对成功的两个个体作为父代1和父代2进行交叉操作。
随机生成两个不同的基因点位，子代1继承父代2基因位之间的基因片段，其余基因按顺序集成父代1中未重复的基因；
子代2继承父代1基因位之间的基因片段，其余基因按顺序集成父代2中未重复的基因。*/
{
	for (int i = 0; i < populationnumber / 2; i++) //将所有个体平均分成两部分，一部分为交叉的父代1，一部分为进行交叉的父代2；
	{
		int n1 = 1 + rand() % workpiecesnumber / 2;    //该方法生成两个不同的基因位；
		int n2 = n1 + rand() % (workpiecesnumber - n1 - 1) + 1;
		int n3 = rand() % 10;
		if (n3 == 2)   //n3=2的概率为0.1；若满足0.1的概率，那么就进行交叉操作；
		{
			vector<int> temp1(workpiecesnumber, 0);
			vector<int> temp2(workpiecesnumber, 0);

			for (int j = 0; j < workpiecesnumber; j++)
			{
				int flg1 = 0; int flg2 = 0;
				for (int p = n1; p < n2; p++)          //将交叉点位之间的基因片段进行交叉，temp1和temp2记录没有发生重复的基因；
				{
					if (a[2 * i + 1][p] == a[2 * i][j])
						flg1 = 1;
				}
				if (flg1 == 0) { temp1[j] = a[2 * i][j]; }

				for (int p = n1; p < n2; p++)
				{
					if (a[2 * i][p] == a[2 * i + 1][j])
						flg2 = 1;
				}
				if (flg2 == 0) { temp2[j] = a[2 * i + 1][j]; }

			}


			for (int j = n1; j < n2; j++)             //子代1继承父代2交叉点位之间的基因；子代2继承父代1交叉点位之间的基因；
			{
				int n4 = 0;
				n4 = a[2 * i][j];
				a[2 * i][j] = a[2 * i + 1][j];
				a[2 * i + 1][j] = n4;
			}
			for (int p = 0; p < n1; p++)               //子代1第一交叉点之前的基因片段，按顺序依次继承父代1中未与子代1重复的基因；
			{
				for (int q = 0; q < workpiecesnumber; q++)
				{
					if (temp1[q] != 0)
					{
						a[2 * i][p] = temp1[q]; temp1[q] = 0;
						break;
					}
				}
			}
			for (int p = 0; p < n1; p++)               //子代2第一交叉点之前的基因片段，按顺序依次继承父代2中未与子代2重复的基因；
			{
				for (int m = 0; m < workpiecesnumber; m++)
				{
					if (temp2[m] != 0)
					{
						a[2 * i + 1][p] = temp2[m]; temp2[m] = 0;
						break;
					}
				}
			}
			for (int p = n2; p < workpiecesnumber; p++)             //子代1第2交叉点之后的基因片段，按顺序依次继承父代1中未与子代1重复的基因；
			{
				for (int q = 0; q < workpiecesnumber; q++)
				{
					if (temp1[q] != 0)
					{
						a[2 * i][p] = temp1[q]; temp1[q] = 0;
						break;
					}

				}
			}
			for (int p = n2; p < workpiecesnumber; p++)               //子代2第2交叉点之后的基因片段，按顺序依次继承父代2中未与子代2重复的基因；
			{
				for (int m = 0; m < workpiecesnumber; m++)
				{
					if (temp2[m] != 0)
					{
						a[2 * i + 1][p] = temp2[m]; temp2[m] = 0;
						break;
					}
				}
			}
		}
	}
	return 0;
}

int mutation()  //变异操作为两点变异，随机生成两个基因位，并交换两个基因的位置；
{
	int n3 = rand() % 20;
	if (n3 == 2)
	{
		for (int i = 0; i < populationnumber; i++)
		{
			int b1 = rand() % workpiecesnumber;
			int b2 = rand() % workpiecesnumber;
			int b3 = a[i][b1];
			a[i][b1] = a[i][b2];
			a[i][b2] = b3;
		}
	}
	return 0;
}
/////////////////////////////////////////////

//调度规则（Random，FIFO，LIFO，LPT，SPT）函数实现

int RANDOM() {
	for (int i = 0; i < populationnumber1; i++)     //首先生成一个订单个数的全排列的个体；
		for (int j = 0; j < workpiecesnumber; j++)
		{
			a1[i][j] = j + 1;
		}
	for (int i = 0; i < populationnumber1; i++)     //将全排列的个体中随机选取两个基因位交换，重复订单个数次，以形成随机初始种群；
		for (int j = 0; j < workpiecesnumber; j++)
		{
			int flg1 = rand() % workpiecesnumber;
			int flg2 = rand() % workpiecesnumber;
			int flg3 = a1[i][flg1];
			a1[i][flg1] = a1[i][flg2];
			a1[i][flg2] = flg3;
		}
	return 0;
}

int FIFO() {
	for (int i = 0; i < populationnumber1; i++)     //首先生成一个订单个数的全排列的个体；
		for (int j = 0; j < workpiecesnumber; j++)
		{
			a1[i][j] = j + 1;
		}
	return 0;
}

int LIFO() {
	int k = 0;
	for (int i = 0; i < populationnumber1; i++)     //首先生成一个订单个数的全排列的个体；
		for (int j = workpiecesnumber + 1; j > 1; j--)
		{
			a1[i][k] = j - 1;
			k++;
		}
	return 0;
}


bool compare(int a, int b) {
	return a < b;  //升序排列
}

bool compare1(int a, int b) {
	return a > b;  //降序排列
}


int LPT() {
	vector<int> sortusetime;
	for (int i = 0; i < workpiecesnumber; i++) {
		sortusetime.push_back(usetime[i][0][0]);
	}

	sort(sortusetime.begin(), sortusetime.end(), compare1);  //这里不需要对compare传参，这是规则
	auto n = unique(sortusetime.begin(), sortusetime.end());   // 去重(注：先排序再去重)
	sortusetime.erase(n, sortusetime.end());

	int k = 0;
	for (int i = 0; i < sortusetime.size(); i++) {
		for (int j = 0; j < workpiecesnumber; j++) {
			if (sortusetime[i] == usetime[j][0][0]) {
				a1[0][k] = j + 1;
				k++;
			}
		}
	}
	return 0;
}

int SPT() {
	vector<int> sortusetime;

	for (int i = 0; i < workpiecesnumber; i++) {
		sortusetime.push_back(usetime[i][0][0]);
	}

	sort(sortusetime.begin(), sortusetime.end(), compare);  //这里不需要对compare传参，这是规则
	auto n = unique(sortusetime.begin(), sortusetime.end());   // 去重(注：先排序再去重)
	sortusetime.erase(n, sortusetime.end());

	int k = 0;
	for (int i = 0; i < sortusetime.size(); i++) {
		for (int j = 0; j < workpiecesnumber; j++) {
			if (sortusetime[i] == usetime[j][0][0]) {
				a1[0][k] = j + 1;
				k++;
			}
		}
	}
	return 0;
}

int fitness1(int c)   //计算适应度函数，c代表某个体；
{
	int totaltime;      //总的加工流程时间（makespan）；
	vector<int> temp1(workpiecesnumber, 0); //加工顺序
	vector<int> temp2(workpiecesnumber, 0);; //上一步骤的完工时间
	vector<int> temp3(workpiecesnumber, 0);;
	vector<vector<vector<int> > > starttime1(workpiecesnumber, vector<vector<int> >(ordernumber, vector<int>(parallel, 0)));
	vector<vector<vector<int> > > finishtime1(workpiecesnumber, vector<vector<int> >(ordernumber, vector<int>(parallel, 0)));
	vector<vector<int> > machinetime1(ordernumber, vector<int>(parallel, 0));

	for (int j = 0; j < workpiecesnumber; j++)   //temp1暂时存储个体c的基因序列，以便进行不同流程之间的加工时记录工件加工先后顺序；
	{
		temp1[j] = a1[c][j];
	}

	for (int i = 0; i < ordernumber; i++)
	{
		for (int j = 0; j < workpiecesnumber; j++)  //该循环的目的是通过比较所有机器的当前工作时间，找出最先空闲的机器，便于新的工件生产；
		{
			int m = machinetime1[i][0];        //先记录第i道工序的第一台并行机器的当前工作时间；
			int n = 0;
			for (int p = 0; p < parallel; p++) //与其他并行机器进行比较，找出时间最小的机器；
			{
				if (usetime[j][i][p] > 0) {
					if (m > machinetime1[i][p] && machinetime1[i][p] >= 0)
					{
						m = machinetime1[i][p];
						n = p;
					}
				}
			}
			int q = temp1[j]; //按顺序提取temp1中的工件号，对工件进行加工；
			starttime1[q - 1][0][n] = times5[q - 1];
			int secondmax = max(starttime1[q - 1][0][n], temp3[j]);
			starttime1[q - 1][i][n] = max(secondmax, machinetime1[i][n]);  //开始加工时间取该机器的当前时间和该工件上一道工序完工时间的最大值；
			machinetime1[i][n] = starttime1[q - 1][i][n] + usetime[q - 1][i][n]; //机器的累计加工时间等于机器开始加工的时刻，加上该工件加工所用的时间；
			finishtime1[q - 1][i][n] = machinetime1[i][n];                 //工件的完工时间就是该机器当前的累计加工时间；
			temp2[j] = finishtime1[q - 1][i][n];       //将每个工件的完工时间赋予temp2，根据完工时间的快慢，便于决定下一道工序的工件加工顺序；
		}

		vector<int> flg2;
		for (int i = 0; i < workpiecesnumber; i++) {
			flg2.push_back(0);
		}

		for (int s = 0; s < workpiecesnumber; s++)
		{
			flg2[s] = temp1[s];
		}

		for (int e = 0; e < workpiecesnumber - 1; e++)
		{
			for (int ee = 0; ee < workpiecesnumber - 1 - e; ee++) // 由于temp2存储工件上一道工序的完工时间，在进行下一道工序生产时，按照先完工先生产的
			{                                            //原则，因此，该循环的目的在于将temp2中按照加工时间从小到大排列，同时temp1相应进行变换来记录temp2中的工件号；
				if (temp2[ee] > temp2[ee + 1])
				{
					int flg5 = temp2[ee];
					int flg6 = flg2[ee];
					temp2[ee] = temp2[ee + 1];
					flg2[ee] = flg2[ee + 1];
					temp2[ee + 1] = flg5;
					flg2[ee + 1] = flg6;
				}
			}
		}
		for (int e = 0; e < workpiecesnumber; e++)    //更新temp1，temp2的数据，开始下一道工序；
		{
			temp1[e] = flg2[e];
			temp3[e] = temp2[e];
		}
	}
	totaltime = 0;
	for (int i = 0; i < parallel; i++) //比较最后一道工序机器的累计加工时间，最大时间就是该流程的加工时间；
		if (totaltime < machinetime1[ordernumber - 1][i])
		{
			totaltime = machinetime1[ordernumber - 1][i];
			//cout<<totaltime<<"   ";
		}
	for (int i = 0; i < workpiecesnumber; i++)  //将数组归零，便于下一个个体的加工时间统计；
		for (int j = 0; j < ordernumber; j++)
			for (int t = 0; t < parallel; t++)
			{
				starttime1[i][j][t] = 0;
				finishtime1[i][j][t] = 0;
				machinetime1[j][t] = 0;
				starttime1[i][0][t] = times5[i];
			}

	std::cout << "Makespan = " << totaltime << endl;
	outfile << "Makespan = " << totaltime << endl;
	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//ACO-TS函数实现

void ValueInit() {  //初始化
	for (int i = 0; i < workpiecesnumber; i++) {
		for (int j = 0; j < workpiecesnumber; j++) {
			Dist[i][j] = 0;
			Eta[i][j] = 1.0 / Dist[i][j];
			Tau[i][j] = 1.0;
			DeltaTau[i][j] = 0;
		}
	}

	for (int i = 0; i < Ants; i++) {
		for (int j = 0; j < workpiecesnumber; j++) {
			Tabu[i][j] = 0;
		}
	}
}

void TargetValue(vector<int> visited1, int visited_size, vector<int> J1, int J_size) {  //计算每次迭代中每只蚂蚁在选定第一个订单的情况下选择下一个订单的概率（此概率为当前以选定订单和下一个将选定订单的makespan的倒数）
	int temp2[1];
	int visited_size1 = visited_size + 1;
	vector<vector<int>> temp1(J_size, vector<int>(visited_size1));    //存储第i只蚂蚁已经选择的订单和下一步可以选择的订单

	for (int i = 0; i < J_size; i++) {
		for (int j = 0; j < visited_size; j++) {
			temp1[i][j] = visited1[j];
		}
		temp1[i][visited_size] = J1[i];
	}


	for (int iii = 0; iii < temp1.size(); iii++) {
		vector<vector<vector<int>>> starttime1(workpiecesnumber, vector<vector<int>>(ordernumber, vector<int>(parallel, 0)));
		vector<vector<int>> machinetime1(ordernumber, vector<int>(parallel, 0));
		vector<vector<vector<int>>> finishtime1(workpiecesnumber, vector<vector<int>>(ordernumber, vector<int>(parallel, 0)));
		for (int i = 0; i < ordernumber; i++)
		{
			for (int j = 0; j < temp1[iii].size(); j++)  //该循环的目的是通过比较所有机器的当前工作时间，找出最先空闲的机器，便于新的工件生产；
			{
				int m = machinetime1[i][0];        //先记录第i道工序的第一台并行机器的当前工作时间；
				int n = 0;
				for (int p = 0; p < parallel; p++) //与其他并行机器进行比较，找出时间最小的机器；
				{

					if (usetime[j][i][p] > 0) {
						if (m > machinetime1[i][p] && machinetime1[i][p] >= 0)
						{
							m = machinetime1[i][p];
							n = p;
						}
					}
				}
				int q = temp1[iii][j]; //按顺序提取temp1中的工件号，对工件进行加工；
				starttime1[q][0][n] = times5[q];
				starttime1[q][i][n] = max(starttime1[q][0][n], machinetime1[i][n]);  //开始加工时间取该机器的当前时间和该工件上一道工序完工时间的最大值；
				machinetime1[i][n] = starttime1[q][i][n] + usetime[q][i][n]; //机器的累计加工时间等于机器开始加工的时刻，加上该工件加工所用的时间；
				finishtime1[q][i][n] = machinetime1[i][n];                 //工件的完工时间就是该机器当前的累计加工时间；
				temp2[0] = finishtime1[q][i][n];       //将每个工件的完工时间赋予temp2，根据完工时间的快慢，便于决定下一道工序的工件加工顺序；
			}
		}

		Dist[visited1.back()][temp1[iii].back()] = temp2[0];
		Eta[visited1.back()][temp1[iii].back()] = 1.0 / Dist[visited1.back()][temp1[iii].back()];

		starttime1.clear();
		machinetime1.clear();
		finishtime1.clear();
	}
	temp1.clear();
}

int fitness2(vector<vector<int>>& Tabu, vector<double> T)   //计算每次迭代所有蚂蚁的makespan；
{
	int totaltime = 0;      //总的加工流程时间（makespan）；
	vector<int> temp1(workpiecesnumber, 0); //加工顺序
	vector<int> temp2(workpiecesnumber, 0); //上一步骤的完工时间
	vector<int> temp3(workpiecesnumber, 0);

	vector<vector<vector<int>>> starttime1(workpiecesnumber, vector<vector<int>>(ordernumber, vector<int>(parallel, 0)));
	vector<vector<vector<int>>> finishtime1(workpiecesnumber, vector<vector<int>>(ordernumber, vector<int>(parallel, 0)));
	vector<vector<int>> machinetime1(ordernumber, vector<int>(parallel, 0));

	for (int ii = 0; ii < Ants; ii++) {
		for (int i = 0; i < ordernumber; i++)
		{
			for (int j = 0; j < workpiecesnumber; j++)  //该循环的目的是通过比较所有机器的当前工作时间，找出最先空闲的机器，便于新的工件生产；
			{
				int m = machinetime1[i][0];        //先记录第i道工序的第一台并行机器的当前工作时间；
				int n = 0;
				for (int p = 0; p < parallel; p++) //与其他并行机器进行比较，找出时间最小的机器；
				{
					if (usetime[j][i][p] > 0) {
						if (m > machinetime1[i][p] && machinetime1[i][p] >= 0)
						{
							m = machinetime1[i][p];
							n = p;
						}
					}
				}
				int q = Tabu[ii][j]; //按顺序提取temp1中的工件号，对工件进行加工；
				temp1[j] = q;
				starttime1[q][0][n] = times5[q];
				int secondmax = max(starttime1[q][0][n], temp3[j]);
				starttime1[q][i][n] = max(secondmax, machinetime1[i][n]);  //开始加工时间取该机器的当前时间和该工件上一道工序完工时间的最大值；
				machinetime1[i][n] = starttime1[q][i][n] + usetime[q][i][n]; //机器的累计加工时间等于机器开始加工的时刻，加上该工件加工所用的时间；
				finishtime1[q][i][n] = machinetime1[i][n];                 //工件的完工时间就是该机器当前的累计加工时间；
				temp2[j] = finishtime1[q][i][n];       //将每个工件的完工时间赋予temp2，根据完工时间的快慢，便于决定下一道工序的工件加工顺序；
			}

			vector<int> flg2(workpiecesnumber, 0);
			for (int s = 0; s < workpiecesnumber; s++)
			{
				flg2[s] = temp1[s];
			}

			for (int e = 0; e < workpiecesnumber - 1; e++)
			{
				for (int ee = 0; ee < workpiecesnumber - 1 - e; ee++) // 由于temp2存储工件上一道工序的完工时间，在进行下一道工序生产时，按照先完工先生产的
				{                                            //原则，因此，该循环的目的在于将temp2中按照加工时间从小到大排列，同时temp1相应进行变换来记录temp2中的工件号；
					if (temp2[ee] > temp2[ee + 1])
					{
						int flg5 = temp2[ee];
						int flg6 = flg2[ee];
						temp2[ee] = temp2[ee + 1];
						flg2[ee] = flg2[ee + 1];
						temp2[ee + 1] = flg5;
						flg2[ee + 1] = flg6;
					}
				}
			}
			for (int e = 0; e < workpiecesnumber; e++)    //更新temp1，temp2的数据，开始下一道工序；
			{
				temp1[e] = flg2[e];
				temp3[e] = temp2[e];
			}
		}
		totaltime = 0;
		for (int i = 0; i < parallel; i++) //比较最后一道工序机器的累计加工时间，最大时间就是该流程的加工时间；
			if (totaltime < machinetime1[ordernumber - 1][i])
			{
				totaltime = machinetime1[ordernumber - 1][i];
			}
		for (int i = 0; i < workpiecesnumber; i++)  //将数组归零，便于下一个个体的加工时间统计；
			for (int j = 0; j < ordernumber; j++)
				for (int t = 0; t < parallel; t++)
				{
					starttime1[i][j][t] = 0;
					finishtime1[i][j][t] = 0;
					machinetime1[j][t] = 0;
					starttime1[i][0][t] = times5[i];
					temp3[i] = 0;
				}
		T[ii] = totaltime;
	}

	return 0;
}

/*《函数Random_Num：生成lower和uper之间的一个double类型随机数》*/
double Random_Num(double lower, double uper)
{
	return  (rand() / (double)RAND_MAX) * (uper - lower) + lower;
}

int Rand(int i) { return rand() % i; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////ABC函数实现
class Graph
{
public:
	int Citycount;
	vector<vector<double>> distance;//城市间的距离矩阵

	void Readcoordinatetxt()//读取城市坐标文件的函数
	{
		distance.resize(workpiecesnumber, vector<double>(workpiecesnumber, 0));
		for (int i = 0; i < workpiecesnumber; i++) {
			for (int j = 0; j < workpiecesnumber; j++) {
				if (i != j)
					distance[i][j] = usetime[i][0][0] + usetime[j][0][0];
				else
					distance[i][j] = 0;
			}
		}
	}
	void shuchu()
	{
		cout << "距离矩阵： " << endl;
		for (int i = 0; i < workpiecesnumber; i++)
		{
			for (int j = 0; j < workpiecesnumber; j++)
			{
				if (j == workpiecesnumber - 1)
					std::cout << distance[i][j] << endl;
				else
					std::cout << distance[i][j] << "  ";
			}
		}
	}
};
Graph Map_City;//定义全局对象图,放在Graph类后
double round(double r) { return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5); }
void Adjuxt_validParticle(int* x)//调整蜜源路径有效性的函数，使得蜜源的位置符合TSP问题解的一个排列
{
	vector<int> route(workpiecesnumber, 0);//1-citycount
	vector<bool> flag(workpiecesnumber, 0);//对应route数组中是否在位置中存在的数组，参考数组为route
	vector<int> biaoji(workpiecesnumber, 0);//对每个元素进行标记的数组,参考数组为粒子位置x
	for (int j = 0; j < workpiecesnumber; j++)
	{
		route[j] = j + 1;
		flag[j] = false;
		biaoji[j] = 0;
	}

	for (int j = 0; j < workpiecesnumber; j++)
	{
		int num = 0;
		for (int k = 0; k < workpiecesnumber; k++)
		{
			if (x[k] == route[j])
			{
				biaoji[k] = 1;//说明k号元素对应的城市在route中，并且是第一次出现才进行标记
				num++; break;
			}
		}
		if (num == 0) flag[j] = false;//路线中没有route[j]这个城市
		else if (num == 1) flag[j] = true;//路线中有route[j]这个城市
	}
	for (int k = 0; k < workpiecesnumber; k++)
	{
		if (flag[k] == false)//路线中没有route[k]这个城市，需要将这个城市加入到路线中
		{
			int i = 0;
			for (; i < workpiecesnumber; i++)
			{
				if (biaoji[i] != 1)break;
			}
			x[i] = route[k];//对于标记为0的进行替换
			biaoji[i] = 1;
		}
	}
}
class Bee
{
public:
	int dimension;//变量的维数
	int* x;
	double fitdegreevalue = 0;
	double selectprobability = 0;
	int limitcishu = 0;
	void Init(int Dimension)
	{
		dimension = Dimension;
		x = new int[dimension];
		for (int j = 0; j < workpiecesnumber; j++)
		{
			*(x + j) = round(xmin + u(random1) * (xmax - xmin));//随机初始化可行解
		}
		Adjuxt_validParticle(x);//对蜜源路径进行有效性调整，确保是TSP问题的一个可行解
	}
	void shuchu()
	{
		for (int j = 0; j < dimension; j++)
		{
			if (j == dimension - 1)std::cout << x[j] << ")" << endl;
			else if (j == 0)
				std::cout << "(" << x[j] << ", ";
			else
				std::cout << x[j] << ", ";
		}
	}
};

class ABC
{
public:
	Bee* miyuan, bestmiyuan, * employedbee, * onlookerbee;//蜜源、最好蜜源、雇佣蜂、观察蜂
	int SN;//雇佣蜂的数量、蜜源的数量、观察蜂的数量相等
	int Dimension;//可行解的维数
	int MCN;//终止代数
	int Limit;//为防止算法陷入局部最优，蜜源最大改进次数
	void Init(int sn, int dimension, int mcn, int limit)
	{
		SN = sn;
		Dimension = dimension;
		MCN = mcn;
		Limit = limit;
		miyuan = new Bee[SN];
		employedbee = new Bee[SN];
		onlookerbee = new Bee[SN];
		for (int i = 0; i < SN; i++)
		{
			(miyuan + i)->Init(Dimension);
			double beefunvalue = CalculateFitValue(*(miyuan + i));
			(miyuan + i)->fitdegreevalue = CalculateFitDegree(beefunvalue);

			(employedbee + i)->Init(Dimension);
			double employedbeefunvalue = CalculateFitValue(*(employedbee + i));
			(employedbee + i)->fitdegreevalue = CalculateFitDegree(employedbeefunvalue);

			(onlookerbee + i)->Init(Dimension);
			double onlookerbeefunvalue = CalculateFitValue(*(onlookerbee + i));
			(onlookerbee + i)->fitdegreevalue = CalculateFitDegree(onlookerbeefunvalue);
		}
		bestmiyuan.Init(Dimension);//最好蜜源初始化
		for (int j = 0; j < Dimension; j++)
		{
			bestmiyuan.x[j] = miyuan->x[j];
		}
		bestmiyuan.fitdegreevalue = miyuan->fitdegreevalue;
	}

	double CalculateFitValue(Bee be)//适应值计算函数，即计算路径长度
	{
		int totaltime = 0;      //总的加工流程时间（makespan）；
		vector<int> temp1(workpiecesnumber, 0); //加工顺序
		vector<int> temp2(workpiecesnumber, 0); //上一步骤的完工时间
		vector<int> temp3(workpiecesnumber, 0);

		vector<vector<vector<int>>> starttime1(workpiecesnumber, vector<vector<int>>(ordernumber, vector<int>(parallel, 0)));
		vector<vector<vector<int>>> finishtime1(workpiecesnumber, vector<vector<int>>(ordernumber, vector<int>(parallel, 0)));
		vector<vector<int>> machinetime1(ordernumber, vector<int>(parallel, 0));

		double funvalue = 0;

		for (int i = 0; i < ordernumber; i++)
		{
			for (int j = 0; j < workpiecesnumber; j++)  //该循环的目的是通过比较所有机器的当前工作时间，找出最先空闲的机器，便于新的工件生产；
			{
				int m = machinetime1[i][0];        //先记录第i道工序的第一台并行机器的当前工作时间；
				int n = 0;
				for (int p = 0; p < parallel; p++) //与其他并行机器进行比较，找出时间最小的机器；
				{
					if (usetime[j][i][p] > 0) {
						if (m > machinetime1[i][p] && machinetime1[i][p] >= 0)
						{
							m = machinetime1[i][p];
							n = p;
						}
					}
				}
				int q = be.x[j] - 1; //按顺序提取temp1中的工件号，对工件进行加工；
				starttime1[q][0][n] = times5[q];
				int secondmax = max(starttime1[q][0][n], temp3[j]);
				starttime1[q][i][n] = max(secondmax, machinetime1[i][n]);  //开始加工时间取该机器的当前时间和该工件上一道工序完工时间的最大值；
				machinetime1[i][n] = starttime1[q][i][n] + usetime[q][i][n]; //机器的累计加工时间等于机器开始加工的时刻，加上该工件加工所用的时间；
				finishtime1[q][i][n] = machinetime1[i][n];                 //工件的完工时间就是该机器当前的累计加工时间；
				temp2[j] = finishtime1[q][i][n];       //将每个工件的完工时间赋予temp2，根据完工时间的快慢，便于决定下一道工序的工件加工顺序；
			}

			vector<int> flg2;  //生成暂时数组，便于将temp1和temp2中的工件重新排列；
			for (int i = 0; i < workpiecesnumber; i++) {
				flg2.push_back(0);
			}
			for (int s = 0; s < workpiecesnumber; s++)
			{
				flg2[s] = temp1[s];
			}

			for (int e = 0; e < workpiecesnumber - 1; e++)
			{
				for (int ee = 0; ee < workpiecesnumber - 1 - e; ee++) // 由于temp2存储工件上一道工序的完工时间，在进行下一道工序生产时，按照先完工先生产的
				{                                            //原则，因此，该循环的目的在于将temp2中按照加工时间从小到大排列，同时temp1相应进行变换来记录temp2中的工件号；
					if (temp2[ee] > temp2[ee + 1])
					{
						int flg5 = temp2[ee];
						int flg6 = flg2[ee];
						temp2[ee] = temp2[ee + 1];
						flg2[ee] = flg2[ee + 1];
						temp2[ee + 1] = flg5;
						flg2[ee + 1] = flg6;
					}
				}
			}
			for (int e = 0; e < workpiecesnumber; e++)    //更新temp1，temp2的数据，开始下一道工序；
			{
				temp1[e] = flg2[e];
				temp3[e] = temp2[e];
			}
		}
		totaltime = 0;
		for (int i = 0; i < parallel; i++) //比较最后一道工序机器的累计加工时间，最大时间就是该流程的加工时间；
			if (totaltime < machinetime1[ordernumber - 1][i])
			{
				totaltime = machinetime1[ordernumber - 1][i];
			}
		for (int i = 0; i < workpiecesnumber; i++)  //将数组归零，便于下一个个体的加工时间统计；
			for (int j = 0; j < ordernumber; j++)
				for (int t = 0; t < parallel; t++)
				{
					starttime1[i][j][t] = 0;
					finishtime1[i][j][t] = 0;
					machinetime1[j][t] = 0;
					starttime1[i][0][t] = times5[i];
					temp3[i] = 0;
				}
		funvalue = totaltime;
		return funvalue;
	}
	double CalculateFitDegree(double fi)//计算适应度的函数
	{
		double fitnessdu = 0;
		if (fi > 0)
			fitnessdu = 1.0 / (1 + fi);
		else
			fitnessdu = 1 + abs(fi);
		return fitnessdu;
	}
	void EmployedBeeOperate()//雇佣蜂操作函数
	{
		std::uniform_int_distribution<int> uD(1, Dimension); //随机数分布对象
		std::uniform_int_distribution<int> uSN(1, SN); //随机数分布对象
		for (int i = 0; i < SN; i++)
		{
			for (int j = 0; j < Dimension; j++)//雇佣峰利用先前的蜜源信息寻找新的蜜源
			{
				(employedbee + i)->x[j] = (miyuan + i)->x[j];
			}
			int k, jie;//随机生成且 k≠i k∈[1,SN]
			jie = uD(random1) - 1;//jie为【1，Dimension】上的随机数
			double random_fi = u1(random1);//random_fi表示[-1,1]之间的随机数
			while (true)
			{
				k = uSN(random1) - 1;
				if (k != i)break;
			}
			(employedbee + i)->x[jie] = (miyuan + i)->x[jie] + random_fi * ((miyuan + i)->x[jie] - (miyuan + k)->x[jie]);//搜索新蜜源
			//保证蜜源位置不越界
			if ((employedbee + i)->x[jie] > xmax) (employedbee + i)->x[jie] = xmax;
			else if ((employedbee + i)->x[jie] < xmin) (employedbee + i)->x[jie] = xmin;
			Adjuxt_validParticle((employedbee + i)->x);//对蜜源路径进行有效性调整，确保是TSP问题的一个可行解
			(employedbee + i)->fitdegreevalue = CalculateFitDegree(CalculateFitValue(*(employedbee + i)));//计算适应度值																							  //雇佣蜂根据贪心策略选择蜜源
			if (CalculateFitValue(*(employedbee + i)) < CalculateFitValue(*(miyuan + i)))
			{
				for (int j = 0; j < Dimension; j++)
					(miyuan + i)->x[j] = (employedbee + i)->x[j];
				(miyuan + i)->limitcishu = 0;
				(miyuan + i)->fitdegreevalue = (employedbee + i)->fitdegreevalue;
			}
			else
				(miyuan + i)->limitcishu++;//蜜源未改进次数加一
		}
	}
	void CalculateProbability()//计算蜜源概率的函数
	{
		for (int i = 0; i < SN; i++)
		{
			(miyuan + i)->fitdegreevalue = CalculateFitDegree(CalculateFitValue(*(miyuan + i)));
		}
		double sumfitdegreevalue = 0;
		for (int i = 0; i < SN; i++)
		{
			sumfitdegreevalue += (miyuan + i)->fitdegreevalue;
		}
		for (int i = 0; i < SN; i++)
		{
			(miyuan + i)->selectprobability = (miyuan + i)->fitdegreevalue / sumfitdegreevalue;
		}
	}
	void OnLookerBeeOperate()//观察蜂操作
	{
		std::uniform_int_distribution<int> uD(1, Dimension); //随机数分布对象
		std::uniform_int_distribution<int> uSN(1, SN); //随机数分布对象
		int m = 0;
		while (m < SN)//为所有的观察蜂按照概率选择蜜源并搜索新蜜源，计算新蜜源适应值
		{
			double m_choosedprobability = u(random1);//0~1之间的随机数
			for (int i = 0; i < SN; i++)
			{
				if (m_choosedprobability < (miyuan + i)->selectprobability)
				{
					int k, jie;//随机生成且 k≠i k∈[1,SN]
					jie = uD(random1) - 1;//jie为【1，Dimension】上的随机数
					double random_fi = u1(random1);//random_fi表示[-1,1]之间的随机数
					while (true)
					{
						k = uSN(random1) - 1;
						if (k != i)break;
					}
					for (int j = 0; j < Dimension; j++)
						(onlookerbee + m)->x[j] = (miyuan + i)->x[j];
					(onlookerbee + m)->x[jie] = (miyuan + i)->x[jie] + random_fi * ((miyuan + i)->x[jie] - (miyuan + k)->x[jie]);//搜索新蜜源
					if ((onlookerbee + m)->x[jie] > xmax) (onlookerbee + m)->x[jie] = xmax;
					else if ((onlookerbee + m)->x[jie] < xmin) (onlookerbee + m)->x[jie] = xmin;//判断是否越界
					Adjuxt_validParticle((onlookerbee + m)->x);//对蜜源路径进行有效性调整，确保是TSP问题的一个可行解
					(onlookerbee + m)->fitdegreevalue = CalculateFitDegree(CalculateFitValue(*(onlookerbee + m)));
					//贪心策略选择蜜源
					if (CalculateFitValue(*(onlookerbee + m)) < CalculateFitValue(*(miyuan + i)))
					{
						for (int j = 0; j < Dimension; j++)
							(miyuan + i)->x[j] = (onlookerbee + m)->x[j];
						(miyuan + i)->fitdegreevalue = (onlookerbee + m)->fitdegreevalue;
					}
					else
						(miyuan + i)->limitcishu++;
					m++;
				}break;
			}
		}
	}
	void ScoutBeeOperate()//侦查蜂操作,决定蜜源是否放弃，并随机产生新位置替代原蜜源
	{
		for (int i = 0; i < SN; i++)
		{
			if ((miyuan + i)->limitcishu >= Limit)
			{
				for (int j = 0; j < Dimension; j++)
				{
					(miyuan + i)->x[j] = round(xmin + u(random1) * (xmax - xmin));//随机初始化可行解;//随机产生可行解
				}
				Adjuxt_validParticle((miyuan + i)->x);
				(miyuan + i)->limitcishu = 0;
				(miyuan + i)->fitdegreevalue = CalculateFitDegree(CalculateFitValue(*(miyuan + i)));
			}
		}
	}
	void SaveBestMiyuan()//记录最优解的函数
	{
		for (int i = 0; i < SN; i++)
		{
			if (CalculateFitValue(*(miyuan + i)) < CalculateFitValue(bestmiyuan))
			{
				for (int j = 0; j < Dimension; j++)
					bestmiyuan.x[j] = (miyuan + i)->x[j];
			}
		}
	}
	void shuchumiyuan()
	{
		for (int i = 0; i < SN; i++)
		{
			std::cout << "蜜源" << i + 1 << "->";
			for (int j = 0; j < Dimension; j++)
			{
				if (j == Dimension - 1)
					std::cout << (miyuan + i)->x[j] << ")对应的距离为： " << CalculateFitValue(*(miyuan + i)) << endl;
				else if (j == 0)
					std::cout << "(" << (miyuan + i)->x[j] << ", ";
				else
					std::cout << (miyuan + i)->x[j] << ", ";
			}
		}
	}
	void ShuchuBestmiyuan()
	{
		for (int c = 0; c < populationnumber1; c++) {
			for (int i = 0; i < workpiecesnumber; i++) {
				a1[c][i] = bestmiyuan.x[i];
			}
			fitness1(c);
		}
	}
	void DoABC(int sn, int dimension, int mcn, int limit)//运行人工蜂群算法的函数
	{
		Map_City.Readcoordinatetxt();
		Init(sn, dimension, mcn, limit);//初始化
		SaveBestMiyuan();//保存最好蜜源
		for (int k = 0; k < MCN; k++)
		{
			EmployedBeeOperate();
			CalculateProbability();
			OnLookerBeeOperate();
			SaveBestMiyuan();
			ScoutBeeOperate();
			SaveBestMiyuan();
		}
		ShuchuBestmiyuan();
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



int main()
{
	string lujing = "../FSSP-BPM/baseline/SchedulingModule/PDR_Meta-heuristic/data/FSP/20_5_10_new";
	ifstream ifs(lujing + ".txt");
	ifstream ifs1(lujing + ".txt");
	outfile.open(lujing + "_output.txt");
	if (!ifs)
	{
		cout << "打开文件失败！请检查文件路径或名称是否存在错误！" << endl;
	}

	std::string line;
	while (std::getline(ifs, line)) { // 逐行读取文件
		workpiecesnumber++;
	}
	ifs.close();

	int l = 0; 
	string data;
	while (ifs1 >> data)
	{
		times1.push_back(data);
		l++;
	}
	ifs1.close();  //读入已知的开始时间和加工时间；

	ordernumber = l / workpiecesnumber;
	parallel = 1;
	machinenumber = ordernumber * parallel;

	//定义vector数组大小
	Tabu.resize(Ants, vector<int>(workpiecesnumber));  //禁忌表，存储蚂蚁选择过的订单
	Dist.resize(workpiecesnumber, vector<double>(workpiecesnumber)); //存储每个订单的完成时间
	Eta.resize(workpiecesnumber, vector<double>(workpiecesnumber));	 //表示启发式因子，为Dist[][]中距离的倒数
	Tau.resize(workpiecesnumber, vector<double>(workpiecesnumber));		//表示两两节点间信息素浓度
	DeltaTau.resize(workpiecesnumber, vector<double>(workpiecesnumber));		//表示两两节点间信息素的变化量
	R_best.resize(I, vector<int>(workpiecesnumber));			//存储每次迭代的最佳路线
	T.resize(Ants);    //存储每次迭代中各只蚂蚁选择的方案的makesapn

	usetime.resize(workpiecesnumber, vector<vector<int> >(ordernumber, vector<int>(parallel, 0))); //第几个订单第几道工序的第几个机器的加工用时；
	machinetime.resize(ordernumber, vector<int>(parallel, 0));  //第几道工序的第几台并行机器的统计时间；
	machinetime_1.resize(ordernumber, vector<string>(parallel));
	starttime.resize(workpiecesnumber, vector<vector<int>>(ordernumber, vector<int>(parallel, 0)));//第几个订单第几道工序在第几台并行机上开始加工的时间；
	starttime_2.resize(workpiecesnumber, vector<vector<string>>(ordernumber, vector<string>(parallel)));
	deliverytime.resize(workpiecesnumber);//订单的交货时间
	finishtime.resize(workpiecesnumber, vector<vector<int>>(ordernumber, vector<int>(parallel, 0)));//第几个订单第几道工序在第几台并行机上完成加工的时间；
	finishtime_1.resize(workpiecesnumber, vector<vector<string>>(ordernumber, vector<string>(parallel)));
	a.resize(populationnumber, vector<int>(workpiecesnumber));//第几代的染色体顺序，即订单加工顺序；
	a1.resize(populationnumber1, vector<int>(workpiecesnumber));
	times3.resize(workpiecesnumber); //用来存储所有订单的开始时间
	times5.resize(workpiecesnumber, 0); //存储每个订单相对于最小开始时间的值
	finishtime1.resize(workpiecesnumber); //记录订单最后一道工序的完成时间
	deliverytime1.resize(workpiecesnumber); //计算订单交货时间与开始时间的差值


	for (int i = 0; i < workpiecesnumber; i++)
	{
		for (int j = 0; j < ordernumber; j++)
		{
			for (int k = 0; k < parallel; k++) {
				usetime[i][j][k] = atoi(times1[machinenumber * i + parallel * j + k].c_str());
			}
		}
	}
	srand(time(NULL));

	//////////////////////////////////////////////////RANDOM
	Start = clock();
	RANDOM();
	std::cout << "RANDOM result: " << endl;
	outfile << "RANDOM result: " << endl;
	for (int c = 0; c < populationnumber1; c++) {
		fitness1(c);
	}
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << "//////////////////////////////////////////////////" << endl;

	////////////////////////////////////////////////FIFO
	Start = clock();
	FIFO();
	std::cout << "FIFO result: " << endl;
	outfile << "FIFO result: " << endl;
	for (int c = 0; c < populationnumber1; c++) {
		fitness1(c);
	}
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << "//////////////////////////////////////////////////" << endl;

	////////////////////////////////////////////////LIFO
	Start = clock();
	LIFO();
	std::cout << "LIFO result: " << endl;
	outfile << "LIFO result: " << endl;
	for (int c = 0; c < populationnumber1; c++) {
		fitness1(c);
	}
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << "//////////////////////////////////////////////////" << endl;

	////////////////////////////////////////////////LPT
	Start = clock();
	LPT();
	std::cout << "LPT result: " << endl;
	outfile << "LPT result: " << endl;
	for (int c = 0; c < populationnumber1; c++) {
		fitness1(c);
	}
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << "//////////////////////////////////////////////////" << endl;

	////////////////////////////////////////////////SPT
	Start = clock();
	SPT();
	std::cout << "SPT result: " << endl;
	outfile << "SPT result: " << endl;
	for (int c = 0; c < populationnumber1; c++) {
		fitness1(c);
	}
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << "//////////////////////////////////////////////////" << endl;

	///////////////////////遗传算法GA
	Start = clock();
	initialization();    //初始化种群；
	Finish1 = clock();
	for (int g = 0; g < G; g++) {
		if(((double)(Finish1 - Start) / CLOCKS_PER_SEC) <= 7200){
			for (int c = 0; c < populationnumber; c++)//计算每个个体适应度并存在ttime中；
			{
				fitness(c);
				ttime[c] = makespan;
			}
			select();     //选择操作；
			crossover();  //交叉操作；
			mutation();   //变异操作；
		}
		Finish1 = clock();
	}
	int flg8 = ttime[0];
	int flg9 = 0;
	for (int c = 0; c < populationnumber - 1; c++)  //计算最后一代每个个体的适应度，并找出最优个体；
	{
		if (ttime[c] < flg8) {
			flg8 = ttime[c];
			flg9 = c;
		}
	}
	std::cout << "GA result: " << endl;
	outfile << "GA result: " << endl;
	gant(flg9);   //画出简易的流程图；
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << "//////////////////////////////////////////////////" << endl;

	///////////////////////人工蜂群算法ABC
	Start = clock();
	ABC abc;
	std::cout << "ABC result: " << endl;
	outfile << "ABC result: " << endl;
	abc.DoABC(50, workpiecesnumber, 200, 20);
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << "//////////////////////////////////////////////////" << endl;

	///////////////////////蚁群优化算法ACO
	Start = clock();
	ValueInit();

	int iii = 0;
	while (iii < I) {
		vector<int> temp(workpiecesnumber, 0);
		for (int j = 0; j < workpiecesnumber; j++) {
			temp[j] = j;
		}
		srand(unsigned(time(0)));
		for (int i = 0; i < Ants; i++) {
			random_shuffle(temp.begin(), temp.end(), Rand);
			for (int j = 0; j < workpiecesnumber; j++) {
				Tabu[i][j] = temp[j];
			}
		}
		//Step4：计算本次迭代各蚂蚁的旅行数据（加工时间、时间等）
		//①记录本代每只蚂蚁走的总加工时间

		for (int i = 0; i < Ants; i++)
		{
			T[i] = 0.0;						//初始化
		}

		fitness2(Tabu, T); //计算本次迭代上面每只蚂蚁生成的路径的makesapn

		//②计算当前迭代所有蚂蚁路径中的最小蚂蚁路径（makespan）
		double min_value = T[0];					//声明求本代所有蚂蚁行走距离最小值的临时变量
		double sum_value = T[0];					//声明求本代所有蚂蚁行走距离总值的临时变量
		int min_index = 0;							//记录本代所有蚂蚁行走距离最小值的下标
		for (int i = 1; i < Ants; i++)
		{
			sum_value += T[i];
			if (T[i] < min_value)
			{
				min_value = T[i];
				min_index = i;
			}
		}

		//③将本次迭代的最小路径值、平均路径值、最短路径数据存在全局迭代数组中
		T_best[iii] = min_value;						//每代中路径的最短长度
		T_ave[iii] = sum_value / Ants;				//每代中路径的平均长度
		for (int i = 0; i < workpiecesnumber; i++)
		{
			R_best[iii][i] = Tabu[min_index][i];		//记录每代最短的路径数据
		}
		//④打印各代距离信息
		//cout << ii << ": T_best is " << T_best[ii] << "    " << "T_ave is " << T_ave[ii] << endl;

		//⑤迭代继续
		iii++;

		//Step5：更新两两节点间的信息素，为下一次迭代的概率选择下一访问节点作准备
		//①全部更新：累加所有蚂蚁，所有路径中，涉及到相同的两两节点间的信息素增量
		for (int i = 0; i < Ants; i++)
		{
			for (int j = 0; j < workpiecesnumber - 1; j++)
			{
				//（禁忌表从1开始，而DeltaTau[][]从0开始存储，所以要减一）
				DeltaTau[Tabu[i][j]][Tabu[i][j + 1]] += Q / T[i];
			}
			//蚁周算法，使用全局距离作更新
			DeltaTau[Tabu[i][workpiecesnumber - 1]][Tabu[i][0]] += Q / T[i];
		}
		//②基于以上两两节点间的信息素增量DeltaTau[][]，对Tau[][]作更新
		for (int i = 0; i < workpiecesnumber; i++)
		{
			for (int j = 0; j < workpiecesnumber; j++)
			{
				//考虑信息素挥发+信息素增量，更新信息素
				Tau[i][j] = (1 - Rho) * Tau[i][j] + DeltaTau[i][j];
			}
		}
	}
	//Step6：展示最终所有迭代中最好的结果
	double min_T1 = T_best[0];			//所有迭代中最短距离
	int min_T_index1 = 0;				//所有迭代中最优路径的迭代下标
	vector<int> Shortest_Route1(workpiecesnumber, 0);				//所有迭代中的最优路径
	for (int i = 0; i < iii; i++)
	{
		if (T_best[i] < min_T1)
		{
			min_T1 = T_best[i];
			min_T_index1 = i;
			//std::cout <<min_T1<<"  " <<min_T_index1 << "  ";
		}
	}
	for (int i = 0; i < workpiecesnumber; i++)		//所有迭代中的最优路径
	{
		Shortest_Route1[i] = R_best[min_T_index1][i] + 1;
		//std::cout << "  " << Shortest_Route1[i];
	}
	std::cout << "ACO result: " << endl;
	outfile << "ACO result: " << endl;
	for (int c = 0; c < populationnumber1; c++) {
		for (int i = 0; i < workpiecesnumber; i++) {
			a1[c][i] = Shortest_Route1[i];
		}
		fitness1(c);
	}
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << "//////////////////////////////////////////////////" << endl;

	///////////////////////禁忌搜索算法TS
	Start = clock();
	ValueInit();

	vector<int> all_jobs(workpiecesnumber, 0);
	int ii = 0;
	while (ii < I) {
		for (int i = 0; i < workpiecesnumber; i++) { //生成一个待加工订单的全序列
			all_jobs[i] = i;
		}

		vector<int> temp(workpiecesnumber, 0);
		for (int i = 0; i < ceil((double)Ants / (double)workpiecesnumber); i++) {
			for (int j = 0; j < workpiecesnumber; j++) {
				temp.push_back(j);
			}
		}
		srand(unsigned(time(0)));
		random_shuffle(temp.begin(), temp.end());
		for (int i = 0; i < Ants; i++)
		{
			Tabu[i][0] = temp[i];
		}

		
		//禁忌表已经有第一个加工订单，所以从第二个点开始概率选择加工
		for (int j = 1; j < workpiecesnumber; j++) {
			for (int i = 0; i < Ants; i++) {
				vector<int> visited;		//已访问过的订单的集合
				vector<int> J;				//待访问的订单的集合
				vector<double> V;				//待访问订单的竞争值的集合
				vector<double> P;			//待访问订单的被选概率的集合

				double Vsum = 0.0;			//竞争值和
				double Psum = 0.0;			//概率和
				double rand_rate = 0.0;		//随机概率
				double choose = 0.0;		//轮盘赌累加概率（用于和rand_rate比较）
				int to_visit = 0;			//下一个要去的订单

				for (int k = 0; k < j; k++)			//j控制当前已访问节点个数（包含出发点）
				{
					visited.push_back(Tabu[i][k]);	
				}

				//J，添加待访问节点
				for (int k = 0; k < workpiecesnumber; k++)
				{
					//若在visited中没有找到某节点all_jobs[k]，则说明还未被访问
					if (find(visited.begin(), visited.end(), all_jobs[k]) == visited.end())
					{
						J.push_back(all_jobs[k]);			//J初始化，添加到待访问订单集合
						P.push_back(0.0);					//P初始化，待访问订单被选概率占坑
						V.push_back(0.0);					//V初始化，待访问订单竞争值占坑
					}
				}

				//把待加工的订单传到TargetValue()计算makespan，即Dist
				int visited_size = visited.size();
				int J_size = J.size();
				TargetValue(visited, visited_size, J, J_size);

				//计算各待访订单的竞争值（已访问节点中最后一个订单—>各待访问各节点）
				for (int k = 0; k < V.size(); k++)
				{
					V[k] = pow(Tau[visited.back()][J[k]], Alpha) * pow(Eta[visited.back()][J[k]], Beta);
					Vsum += V[k];
				}

				//计算去下一个订单的各种概率（已访问节点中最后一个订单—>各待访问各节点）
				for (int k = 0; k < P.size(); k++)
				{
					P[k] = V[k] / Vsum;
					Psum += P[k];
				}

				//使用轮盘赌算法
				rand_rate = Random_Num(0.0, Psum);		//随机概率		
				for (int k = 0; k < P.size(); k++)
				{
					choose += P[k];						//概率小累计，用于与随机概率比较
					if (choose > rand_rate)
					{
						to_visit = J[k];
						break;
					}
				}

				//更新禁忌表（把刚选出的下一访问节点及时加入）
				Tabu[i][j] = to_visit;
			}
		}

		for (int i = 0; i < Ants; i++)
		{
			T[i] = 0.0;						//初始化
		}

		fitness2(Tabu, T); 

		double min_value = T[0];				
		double sum_value = T[0];	
		int min_index = 0;		
		for (int i = 1; i < Ants; i++)
		{
			sum_value += T[i];
			if (T[i] < min_value)
			{
				min_value = T[i];
				min_index = i;
			}
		}

		T_best[ii] = min_value;						//每代中路径的最短长度
		T_ave[ii] = sum_value / Ants;				//每代中路径的平均长度
		for (int i = 0; i < workpiecesnumber; i++)
		{
			R_best[ii][i] = Tabu[min_index][i];		//记录每代最短的路径数据
		}

		//迭代继续
		ii++;

		for (int i = 0; i < Ants; i++)
		{
			for (int j = 0; j < workpiecesnumber - 1; j++)
			{
				//（禁忌表从1开始，而DeltaTau[][]从0开始存储，所以要减一）
				DeltaTau[Tabu[i][j]][Tabu[i][j + 1]] += Q / T[i];
			}
			DeltaTau[Tabu[i][workpiecesnumber - 1]][Tabu[i][0]] += Q / T[i];
		}
		for (int i = 0; i < workpiecesnumber; i++)
		{
			for (int j = 0; j < workpiecesnumber; j++)
			{
				Tau[i][j] = (1 - Rho) * Tau[i][j] + DeltaTau[i][j];
			}
		}
		//禁忌表清零,进入下一次迭代!
		for (int i = 0; i < Ants; i++)
		{
			for (int j = 0; j < workpiecesnumber; j++)
			{
				Tabu[i][j] = 0;
			}
		}
	}
	//展示最终所有迭代中最好的结果
	double min_T = T_best[0];	
	int min_T_index = 0;	
	vector<int> Shortest_Route(workpiecesnumber, 0);	
	for (int i = 0; i < ii; i++)
	{
		if (T_best[i] < min_T)
		{
			min_T = T_best[i];
			min_T_index = i;
		}
	}
	for (int i = 0; i < workpiecesnumber; i++)		//所有迭代中的最优路径
	{
		Shortest_Route[i] = R_best[min_T_index][i] + 1;
		//cout << "  " << Shortest_Route1[i];
	}
	std::cout << "TS result: " << endl;
	outfile << "TS result: " << endl;
	for (int c = 0; c < populationnumber1; c++) {
		for (int i = 0; i < workpiecesnumber; i++) {
			a1[c][i] = Shortest_Route[i];
		}
		fitness1(c);
	}
	Finish = clock();
	std::cout << "Total Running Time = " << (double)(Finish - Start) / CLOCKS_PER_SEC << " s" << endl;
	std::cout << "//////////////////////////////////////////////////" << endl;

	system("Pause");
	return 0;
}
