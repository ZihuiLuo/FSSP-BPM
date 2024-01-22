#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>

using namespace std;

#define cin fin
#define cout fout
#define INF 0x3ffffff
#define Customer_Number 300//算例中除仓库以外的顾客节点个数
#define Capacity 180//车辆最大容量
#define Iter_Max 1000//搜索迭代次数
#define Tabu_tenure 0.2*Customer_Number//禁忌时长

ifstream fin("TS_300_3stage_batch.txt");
ofstream fout("TS_300_3stage_batch_Output.txt");

struct Customer_Type {
	int Number;//节点自身编号
	int R;//节点所属车辆路径编号
	//int id;
	double a, b, c, d, e, f, h, i, art, thickness, width, date, priority;//节点横纵坐标//节点属性
	double Begin, End, Service;//节点被访问的最早时间，最晚时间以及服务时长
	double Demand;//节点的需求
} Customer[Customer_Number + 10];//仓库节点编号为1，顾客节点编号为2-101

struct Route_Type {
	double Load;//单挑路径承载量
	double SubT;//单条路径违反各节点时间窗约束时长总和
	double Dis;//单条路径总长度
	vector<Customer_Type> V;//单条路径上顾客节点序列
} Route[Customer_Number + 10], Route_Ans[Customer_Number + 10];//车辆路径及搜索到最优的车辆路径

int Vehicle_Number = Customer_Number;//由于无车辆数量限制，因此将上限设为顾客总数
int Tabu[Customer_Number + 10][Customer_Number + 10];//禁忌表用于禁忌节点插入操作
int TabuCreate[Customer_Number + 10];//禁忌表用于紧急拓展新路径或使用新车辆

double Ans;
double Alpha = 1, Beta = 1, Sita = 0.5;
double Graph[Customer_Number + 10][Customer_Number + 10];
//************************************************************
double Distance(Customer_Type C1, Customer_Type C2) {//计算图上各节点见的距离
	return sqrt(0.2 * (C1.a - C2.a) * (C1.a - C2.a) + 0.2 * (C1.b - C2.b) * (C1.b - C2.b) + 0.2 * pow((C1.c - C2.c), 2) 
		+ 0.2 * pow((C1.d - C2.d), 2) + 0.2 * pow((C1.e - C2.e), 2) + 0.2 * pow((C1.f - C2.f), 2) + 0.2 * pow((C1.h - C2.h), 2) 
		+ 0.2 * pow((C1.i - C2.i), 2) + 0.2 * pow((C1.art - C2.art), 2) + 0.2 * pow((C1.thickness - C2.thickness), 2) 
		+ 0.16 * pow((C1.width - C2.width), 2) + 0.16 * pow((C1.date - C2.date), 2) + 0.08 * pow((C1.priority - C2.priority), 2));
}
//************************************************************
double Calculation(Route_Type R[], int Cus, int NewR) {//计算路径规划R的目标函数值
	//目标函数主要由三个部分组成：D路径总长度（优化目标），Q超出容量约束总量，T超出时间窗约束总量
	//目标函数结构为 f(R) = D + Alpha * Q + Beta * T, 第一项为问题最小化目标，后两项为惩罚部分
	//期中Alpha与Beta为变量，分别根据当前解是否满足两个约束进行变化（在Check函数中更新，由于Check针对每轮迭代得到的解）
	double Q = 0;
	double T = 0;
	double D = 0;


	//计算单条路径超出容量约束的总量
	for (int i = 1; i <= Vehicle_Number; ++i)
		if (R[i].V.size() > 2 && R[i].Load > Capacity)
			Q = Q + R[i].Load - Capacity;

	//计算单条路径上各个节点超出时间窗约束的总量（仅更新进行移除和插入节点操作的两条路径）
	double ArriveTime = 0;
	R[Customer[Cus].R].SubT = 0;
	for (int j = 1; j < R[Customer[Cus].R].V.size(); ++j) {
		ArriveTime = ArriveTime + R[Customer[Cus].R].V[j - 1].Service + Graph[R[Customer[Cus].R].V[j - 1].Number][R[Customer[Cus].R].V[j].Number];
		if (ArriveTime > R[Customer[Cus].R].V[j].End)
			R[Customer[Cus].R].SubT = R[Customer[Cus].R].SubT + ArriveTime - R[Customer[Cus].R].V[j].End;
		else if (ArriveTime < R[Customer[Cus].R].V[j].Begin)
			ArriveTime = R[Customer[Cus].R].V[j].Begin;
	}

	ArriveTime = 0;
	R[NewR].SubT = 0;
	for (int j = 1; j < R[NewR].V.size(); ++j) {
		ArriveTime = ArriveTime + R[NewR].V[j - 1].Service + Graph[R[NewR].V[j - 1].Number][R[NewR].V[j].Number];
		if (ArriveTime > R[NewR].V[j].End)
			R[NewR].SubT = R[NewR].SubT + ArriveTime - R[NewR].V[j].End;
		else if (ArriveTime < R[NewR].V[j].Begin)
			ArriveTime = R[NewR].V[j].Begin;
	}

	for (int i = 1; i <= Vehicle_Number; ++i)
		T += R[i].SubT;


	//计算路径总长度
	for (int i = 1; i <= Vehicle_Number; ++i)
		D += R[i].Dis;

	return D + Alpha * Q + Beta * T;
}
//************************************************************
bool Check(Route_Type R[]) {//检验路径规划R是否满足所有约束
	double Q = 0;
	double T = 0;
	double D = 0;

	//检查是否满足容量约束
	for (int i = 1; i <= Vehicle_Number; ++i)
		if (R[i].V.size() > 2 && R[i].Load > Capacity)
			Q = Q + R[i].Load - Capacity;

	//检查是否满足时间窗约束
	for (int i = 1; i <= Vehicle_Number; ++i)
		T += R[i].SubT;

	//分别根据约束满足的情况更新Alpha和Beta值
	if (Q == 0 && Alpha >= 0.001)
		Alpha /= (1 + Sita);
	else if (Q != 0 && Alpha <= 2000)
		Alpha *= (1 + Sita);

	if (T == 0 && Beta >= 0.001)
		Beta /= (1 + Sita);
	else if (T != 0 && Beta <= 2000)
		Beta *= (1 + Sita);

	if (T == 0 && Q == 0)
		return true;
	else
		return false;
}
//************************************************************
void Copy_Route() {//将路径规划Route的内容复制给路径规划Route_Ans
	for (int i = 1; i <= Vehicle_Number; ++i) {
		Route_Ans[i].Load = Route[i].Load;
		Route_Ans[i].V.clear();
		for (int j = 0; j < Route[i].V.size(); ++j)
			Route_Ans[i].V.push_back(Route[i].V[j]);
	}
}
//************************************************************
void Output(Route_Type R[]) {//结果输出
	cout << "************************************************************" << endl;
	cout << "The Minimum Total Distance = " << Ans << endl;
	cout << "Concrete Schedule of Each Route as Following : " << endl;
	cout << "注释：此处编号“1”为批次分隔符，所有作业号应减1才是对应作业" << endl;
	int M = 0;
	for (int i = 1; i <= Vehicle_Number; ++i)
		if (R[i].V.size() > 4) {
			M++;
			cout << "No." << M << " : ";
			for (int j = 0; j < R[i].V.size() - 1; ++j)
				cout << R[i].V[j].Number << " -> ";
			cout << R[i].V[R[i].V.size() - 1].Number << endl;
		}

	//检验距离计算是否正确
	double Check_Ans = 0;
	for (int i = 1; i <= Vehicle_Number; ++i)
		for (int j = 1; j < R[i].V.size(); ++j)
			Check_Ans += Graph[R[i].V[j - 1].Number][R[i].V[j].Number];

	cout << "Check_Ans = " << Check_Ans << endl;
	cout << "************************************************************" << endl;

	//cout << "[";
	for (int i = 1; i <= Vehicle_Number; ++i)
		if (R[i].V.size() > 4) {
			for (int j = 1; j < R[i].V.size() - 2; ++j) {
				/*cout << "[" << R[i].V[j].Number << "], ";*/
				if ((R[i].V[j].Number - 1) != 0) {
					cout << "作业号：" << R[i].V[j].Number - 1 << " ";
					cout << R[i].V[j].a << " ";
					cout << R[i].V[j].b << " ";
					cout << R[i].V[j].c << " ";
					cout << R[i].V[j].d << " ";
					cout << R[i].V[j].Demand << " ";
					cout << endl;
				}
				//cout << R[i].V[R[i].V.size() - 1].Number << endl;
			}
			cout << endl;
		}
	//cout << "]" << endl;
}
//************************************************************
void ReadIn_and_Initialization() {//数据读入及初始化
	for (int i = 1; i <= Customer_Number + 1; ++i)
		cin >> Customer[i].Number >> Customer[i].id >> Customer[i].a >> Customer[i].b >> Customer[i].c >> Customer[i].d 
		>> Customer[i].e >> Customer[i].f >> Customer[i].h >> Customer[i].i >> Customer[i].art >> Customer[i].thickness 
		>> Customer[i].width >> Customer[i].date >> Customer[i].priority >> Customer[i].Demand
		>> Customer[i].Begin >> Customer[i].End >> Customer[i].Service;

	//初始化每条路径，默认路径收尾为仓库，且首仓库最早最晚时间均为原仓库最早时间，尾仓库则均为原仓库最晚时间
	Customer[1].R = -1;
	for (int i = 1; i <= Vehicle_Number; ++i) {
		if (!Route[i].V.empty())
			Route[i].V.clear();
		Route[i].V.push_back(Customer[1]);
		Route[i].V.push_back(Customer[1]);
		Route[i].V.push_back(Customer[1]);
		Route[i].V.push_back(Customer[1]);
		Route[i].V[0].End = Route[i].V[0].Begin;
		Route[i].V[1].Begin = Route[i].V[1].End;
		Route[i].Load = 0;
	}

	Ans = INF;

	for (int i = 1; i <= Customer_Number + 1; ++i)
		for (int j = 1; j <= Customer_Number + 1; ++j)
			Graph[i][j] = Distance(Customer[i], Customer[j]);
}
//************************************************************
void Construction() {//构造初始路径规划
	int Customer_Set[Customer_Number + 10];
	for (int i = 1; i <= Customer_Number; ++i)
		Customer_Set[i] = i + 1;

	int Sizeof_Customer_Set = Customer_Number;
	int Current_Route = 1;

	//以满足容量约束为目的的随机初始化
	//即随机挑选一个节点插入到第m条路径中，若超过容量约束，则插入第m+1条路径
	//且插入路径的位置由该路径上已存在的各节点最早时间的升序决定
	while (Sizeof_Customer_Set > 0) {
		int K = rand() % Sizeof_Customer_Set + 1;
		int C = Customer_Set[K];
		Customer_Set[K] = Customer_Set[Sizeof_Customer_Set];
		Sizeof_Customer_Set--;

		/*int K = rand() % Sizeof_Customer_Set + 1;
		int C = Customer_Set[K];
		Customer_Set[K] = Customer_Set[Sizeof_Customer_Set];
		Sizeof_Customer_Set--;*/
		//将当前服务过的节点赋值为最末节点值,数组容量减1

		if (Route[Current_Route].Load + Customer[C].Demand > Capacity)
			Current_Route++;

		for (int i = 0; i < Route[Current_Route].V.size() - 1; i++)
			if ((Route[Current_Route].V[i].Begin <= Customer[C].Begin) && (Customer[C].Begin <= Route[Current_Route].V[i + 1].Begin)) {
				Route[Current_Route].V.insert(Route[Current_Route].V.begin() + i + 1, Customer[C]);
				Route[Current_Route].Load += Customer[C].Demand;
				Customer[C].R = Current_Route;
				break;
			}
	}

	//初始化计算超过容量约束的总量和超过时间窗约束的总量
	for (int i = 1; i <= Vehicle_Number; ++i) {
		double ArriveTime = Route[i].V[0].Begin;
		Route[i].SubT = 0;
		Route[i].Dis = 0;
		for (int j = 1; j < Route[i].V.size(); ++j) {
			ArriveTime = ArriveTime + Route[i].V[j - 1].Service + Graph[Route[i].V[j - 1].Number][Route[i].V[j].Number];
			Route[i].Dis += Graph[Route[i].V[j - 1].Number][Route[i].V[j].Number];
			if (ArriveTime > Route[i].V[j].End)
				Route[i].SubT = Route[i].SubT + ArriveTime - Route[i].V[j].End;
			else if (ArriveTime < Route[i].V[j].Begin)
				ArriveTime = Route[i].V[j].Begin;
		}
	}
}
//************************************************************
void Tabu_Search() {//禁忌搜索
	//禁忌搜索采取插入算子，即从一条路径中选择一点插入到另一条路径中
	//在该操作下形成的邻域中选取使目标函数最小化的解
	double Temp1;
	double Temp2;

	for (int i = 2; i <= Customer_Number + 1; ++i) {
		for (int j = 1; j <= Vehicle_Number; ++j)
			Tabu[i][j] = 0;
		TabuCreate[i] = 0;
	}

	int Iteration = 0;
	while (Iteration < Iter_Max) {
		Iteration++;
		int BestC = 0, BestR = 0, BestP = 0, P;
		double BestV = INF;


		for (int i = 2; i <= Customer_Number + 1; ++i) {
			for (int j = 1; j < Route[Customer[i].R].V.size(); ++j)
				if (Route[Customer[i].R].V[j].Number == i) {
					P = j;
					break;
				}
			//从节点原路径中去除该节点的需求
			Route[Customer[i].R].Load -= Customer[i].Demand;
			//从节点原路径中去除该节点所组成的路径并重组
			Route[Customer[i].R].Dis = Route[Customer[i].R].Dis - Graph[Route[Customer[i].R].V[P - 1].Number][Route[Customer[i].R].V[P].Number]
				- Graph[Route[Customer[i].R].V[P].Number][Route[Customer[i].R].V[P + 1].Number] + Graph[Route[Customer[i].R].V[P - 1].Number][Route[Customer[i].R].V[P + 1].Number];
			//从节点原路径中去除节点
			Route[Customer[i].R].V.erase(Route[Customer[i].R].V.begin() + P);

			for (int j = 1; j <= Vehicle_Number; ++j)
				//禁忌插入操作，后者为禁止使用新的车辆
				if ((Route[j].V.size() > 2 && Tabu[i][j] <= Iteration) || (Route[j].V.size() == 2 && TabuCreate[i] <= Iteration)) {
					for (int l = 1; l < Route[j].V.size(); ++l)
						if (Customer[i].R != j) {
							//在节点新路径中加上该节点的需求
							Route[j].Load += Customer[i].Demand;
							//在节点新路径中加上该节点插入后所组成的路径并断开原路径
							Route[j].Dis = Route[j].Dis - Graph[Route[j].V[l - 1].Number][Route[j].V[l].Number]
								+ Graph[Route[j].V[l - 1].Number][Customer[i].Number] + Graph[Route[j].V[l].Number][Customer[i].Number];
							//在节点新路径中插入节点
							Route[j].V.insert(Route[j].V.begin() + l, Customer[i]);
							Temp1 = Route[Customer[i].R].SubT;
							Temp2 = Route[j].SubT;

							double TempV = Calculation(Route, i, j);
							if (TempV < BestV) {
								BestV = TempV;
								BestC = i, BestR = j, BestP = l;
							}
							//节点新路径复原
							Route[Customer[i].R].SubT = Temp1;
							Route[j].SubT = Temp2;
							Route[j].V.erase(Route[j].V.begin() + l);
							Route[j].Load -= Customer[i].Demand;
							Route[j].Dis = Route[j].Dis + Graph[Route[j].V[l - 1].Number][Route[j].V[l].Number]
								- Graph[Route[j].V[l - 1].Number][Customer[i].Number] - Graph[Route[j].V[l].Number][Customer[i].Number];
						}
				}
			//节点原路径复原
			Route[Customer[i].R].V.insert(Route[Customer[i].R].V.begin() + P, Customer[i]);
			Route[Customer[i].R].Load += Customer[i].Demand;
			Route[Customer[i].R].Dis = Route[Customer[i].R].Dis + Graph[Route[Customer[i].R].V[P - 1].Number][Route[Customer[i].R].V[P].Number]
				+ Graph[Route[Customer[i].R].V[P].Number][Route[Customer[i].R].V[P + 1].Number] - Graph[Route[Customer[i].R].V[P - 1].Number][Route[Customer[i].R].V[P + 1].Number];
		}

		if (Route[BestR].V.size() == 2)
			TabuCreate[BestC] = Iteration + 2 * Tabu_tenure + rand() % 10;
		Tabu[BestC][Customer[BestC].R] = Iteration + Tabu_tenure + rand() % 10;
		for (int i = 1; i < Route[Customer[BestC].R].V.size(); ++i)
			if (Route[Customer[BestC].R].V[i].Number == BestC) {
				P = i;
				break;
			}

		//依据上述循环中挑选的结果，生成新的总体路径规划
		//更新改变过的各单挑路径的载重，距离长度，超出时间窗重量
		Route[Customer[BestC].R].Load -= Customer[BestC].Demand;
		Route[Customer[BestC].R].Dis = Route[Customer[BestC].R].Dis - Graph[Route[Customer[BestC].R].V[P - 1].Number][Route[Customer[BestC].R].V[P].Number]
			- Graph[Route[Customer[BestC].R].V[P].Number][Route[Customer[BestC].R].V[P + 1].Number] + Graph[Route[Customer[BestC].R].V[P - 1].Number][Route[Customer[BestC].R].V[P + 1].Number];

		Route[Customer[BestC].R].V.erase(Route[Customer[BestC].R].V.begin() + P);
		Route[BestR].Dis = Route[BestR].Dis - Graph[Route[BestR].V[BestP - 1].Number][Route[BestR].V[BestP].Number]
			+ Graph[Route[BestR].V[BestP - 1].Number][Customer[BestC].Number] + Graph[Route[BestR].V[BestP].Number][Customer[BestC].Number];
		Route[BestR].Load += Customer[BestC].Demand;
		Route[BestR].V.insert(Route[BestR].V.begin() + BestP, Customer[BestC]);

		double ArriveTime = 0;
		Route[BestR].SubT = 0;
		for (int j = 1; j < Route[BestR].V.size(); ++j) {
			ArriveTime = ArriveTime + Route[BestR].V[j - 1].Service + Graph[Route[BestR].V[j - 1].Number][Route[BestR].V[j].Number];
			if (ArriveTime > Route[BestR].V[j].End)
				Route[BestR].SubT = Route[BestR].SubT + ArriveTime - Route[BestR].V[j].End;
			else if (ArriveTime < Route[BestR].V[j].Begin)
				ArriveTime = Route[BestR].V[j].Begin;
		}
		ArriveTime = 0;
		Route[Customer[BestC].R].SubT = 0;
		for (int j = 1; j < Route[Customer[BestC].R].V.size(); ++j) {
			ArriveTime = ArriveTime + Route[Customer[BestC].R].V[j - 1].Service + Graph[Route[Customer[BestC].R].V[j - 1].Number][Route[Customer[BestC].R].V[j].Number];
			if (ArriveTime > Route[Customer[BestC].R].V[j].End)
				Route[Customer[BestC].R].SubT = Route[Customer[BestC].R].SubT + ArriveTime - Route[Customer[BestC].R].V[j].End;
			else if (ArriveTime < Route[Customer[BestC].R].V[j].Begin)
				ArriveTime = Route[Customer[BestC].R].V[j].Begin;
		}

		//更新被操作的节点所属路径编号
		Customer[BestC].R = BestR;

		//如果当前解合法且较优则更新存储结果
		if ((Check(Route) == true) && (Ans > BestV)) {
			Copy_Route();
			Ans = BestV;
		}
	}
}
//************************************************************
int main() {
	clock_t Start, Finish;
	Start = clock();

	srand((unsigned)time(NULL));
	ReadIn_and_Initialization();
	Construction();
	Tabu_Search();
	Output(Route_Ans);

	Finish = clock();
	cout << "Total Running Time = " << ( Finish - Start ) / 1000.0 << endl;
	return 0;
}
//************************************************************