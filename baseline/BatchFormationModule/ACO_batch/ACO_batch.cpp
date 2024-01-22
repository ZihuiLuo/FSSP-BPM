#pragma once
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>

using namespace std;

const int N_CITY_COUNT = 300;						//作业（订单）数量
const int N_ANT_COUNT = 3 * N_CITY_COUNT / 2;			//蚂蚁数量
const double Capacity = 180;                         //一个批次的容量
const double ALPHA = 1.0;							//启发因子，信息素的重要程度
const double BETA = 2.0;							//期望因子，城市间距离的重要程度
const double ROU = 0.5;								//信息素残留参数
const int N_IT_COUNT = 1000;						//迭代次数
const int virtualNode = 20;
const double DBQ = 100.0;							//总的信息素
const double DB_MAX = 10e9;							//一个标志数，10的9次方
double g_Trial[N_CITY_COUNT][N_CITY_COUNT];			//两两作业间信息素，就是环境信息素
double g_Distance[N_CITY_COUNT][N_CITY_COUNT];		//两两作业间difference
double demand[N_CITY_COUNT];
double g_sourse[N_CITY_COUNT];
clock_t start, finish;
ofstream outfile;
int id[N_CITY_COUNT];
double a[N_CITY_COUNT];
double b[N_CITY_COUNT];
double c[N_CITY_COUNT];
double d[N_CITY_COUNT];

//double minPathDist;
//double UnDeliveryRate;



//返回指定范围内的随机整数
int rnd(int nLow, int nUpper)
{
	return nLow + (nUpper - nLow) * rand() / (RAND_MAX + 1);
}
//返回指定范围内的随机浮点数
double rnd(double dbLow, double dbUpper)
{
	double dbTemp = rand() / ((double)RAND_MAX + 1.0);
	return dbLow + dbTemp * (dbUpper - dbLow);
}
//返回浮点数四舍五入取整后的浮点数
double ROUND(double dbA)
{
	return (double)((int)(dbA + 0.5));
}
//定义蚂蚁类
class CAnt
{
public:
	CAnt(void);
	~CAnt(void);
public:
	int m_nPath[N_CITY_COUNT];				//蚂蚁走的路径
	double m_dbPathLength;					//蚂蚁走过的路径长度
	int m_nAllowedCity[N_CITY_COUNT];		//待分批的作业
	int m_nCurCityNo;						//当前所分批作业编号
	int m_nMovedCityCount;					//已分批的作业数量
	double UnDeliveryRate;
public:
	int ChooseNextCity();					//选择下一个作业
	void Init();							//初始化
	void Move();							//蚂蚁在作业间移动
	void Search();							//搜索路径
	void CalPathLength();					//计算蚂蚁走过的路径长度
};
//构造函数
CAnt::CAnt(void)
{	}

//析构函数
CAnt::~CAnt(void)
{	}

//初始化函数，蚂蚁搜索前调用
void CAnt::Init()
{
	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		m_nAllowedCity[i] = 1;		//设置全部作业未被分批
		m_nPath[i] = 0;				//蚂蚁走的路径全部设置为0
	}

	m_dbPathLength = 0.0;							//蚂蚁走过的路径长度设置为0
	m_nCurCityNo = rnd(0, N_CITY_COUNT);			//随机选择一个作业
	m_nPath[0] = m_nCurCityNo;						//把选择的作业保存入路径数组中
	m_nAllowedCity[m_nCurCityNo] = 0;				//标识被选择作业为已被分批
	m_nMovedCityCount = 1;							//已分批的作业数量设置为1
}

//选择下一个作业
//返回值 为作业编号
int CAnt::ChooseNextCity()
{
	int nSelectedCity = -1;						//返回结果，先暂时把其设置为-1

	//==============================================================================
	//计算当前作业和没分批的作业之间的信息素总和

	double dbTotal = 0.0;
	double prob[N_CITY_COUNT];					//保存各个作业被选中的概率
	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		if (m_nAllowedCity[i] == 1)				//作业没被分批
		{
			prob[i] = pow(g_Trial[m_nCurCityNo][i], ALPHA) * pow(1.0 / g_Distance[m_nCurCityNo][i], BETA); //该作业和当前作业间的信息素
			dbTotal = dbTotal + prob[i];			//累加信息素，得到总和
		}
		else									//如果作业已被选择了，则其被选中的概率值为0
		{
			prob[i] = 0.0;
		}
	}
	//==============================================================================
	//进行轮盘选择
	double dbTemp = 0.0;
	if (dbTotal > 0.0)							//总的信息素值大于0
	{
		dbTemp = rnd(0.0, dbTotal);				//取一个随机数
		for (int i = 0; i < N_CITY_COUNT; i++)
		{
			if (m_nAllowedCity[i] == 1)			//作业未被分批
			{
				dbTemp = dbTemp - prob[i];			//这个操作相当于转动轮盘（轮盘赌）
				if (dbTemp < 0.0)				//轮盘停止转动，记下作业编号，直接跳出循环

				{
					nSelectedCity = i;
					break;
				}
			}
		}
	}
	//==============================================================================
	//如果作业间的信息素非常小 ( 小到比double能够表示的最小的数字还要小 )
	//那么由于浮点运算的误差原因，上面计算的概率总和可能为0
	//会出现经过上述操作，没有作业被选择出来
	//出现这种情况，就把第一个没被选择的作业作为返回结果

	if (nSelectedCity == -1)
	{
		for (int i = 0; i < N_CITY_COUNT; i++)
		{
			if (m_nAllowedCity[i] == 1)				//作业未被选择过
			{
				nSelectedCity = i;
				break;
			}
		}
	}
	//==============================================================================
	//返回结果，就是作业的编号
	return nSelectedCity;
}

//蚂蚁在作业间移动
void CAnt::Move()
{
	int nCityNo = ChooseNextCity();						//选择下一个作业
	m_nPath[m_nMovedCityCount] = nCityNo;				//保存蚂蚁走的路径
	m_nAllowedCity[nCityNo] = 0;						//把这个作业设置成已经被选择过了
	m_nCurCityNo = nCityNo;								//改变当前所在作业为选择的作业
	m_nMovedCityCount++;								//已经选择过的作业数量加1
}

//蚂蚁进行搜索一次
void CAnt::Search()
{
	Init();												//蚂蚁搜索前，先初始化
	//如果蚂蚁去过的作业数量小于城市数量，就继续移动
	while (m_nMovedCityCount < N_CITY_COUNT)
	{
		Move();
	}
	//完成搜索后计算走过的路径长度
	CalPathLength();
}

//计算蚂蚁走过的路径长度
void CAnt::CalPathLength()
{
	m_dbPathLength = 0.0;							// 先把路径长度置0
	//int m=0;
	//int n=0;
	double sum = 0;
	bool tag = 0;
	double unDelivery = 0;
	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		sum += demand[m_nPath[i]];
		if (sum < Capacity)
		{
			if (!tag)
			{
				m_dbPathLength += g_sourse[m_nPath[i]];
				tag = 1;
			}
			else
			{
				m_dbPathLength += g_Distance[m_nPath[i - 1]][m_nPath[i]];
			}
		}
		else
		{
			unDelivery += g_sourse[m_nPath[i - 1]];
			m_dbPathLength += g_sourse[m_nPath[i - 1]];
			i--;
			sum = 0;
			tag = 0;
		}
	}

	m_dbPathLength += g_sourse[m_nPath[N_CITY_COUNT - 1]];
	unDelivery += g_sourse[m_nPath[N_CITY_COUNT - 1]];

	UnDeliveryRate = unDelivery / m_dbPathLength;

	//加上从最后作业返回出发作业的距离

}

//tsp类
class CTsp
{
public:
	CTsp(void);
	~CTsp(void);
public:
	CAnt m_cAntAry[N_ANT_COUNT];				//蚂蚁数组
	CAnt m_cBestAnt;							//定义一个蚂蚁变量，用来保存搜索过程中的最优结果
	//该蚂蚁不参与搜索，只是用来保存最优结果
public:
	//初始化数据
	void InitData();
	//开始搜索
	void Search();
	//更新环境信息素
	void UpdateTrial();

};

//构造函数
CTsp::CTsp(void)
{	}
CTsp::~CTsp(void)
{	}

//初始化数据
void CTsp::InitData()
{
	//先把最优蚂蚁的路径长度设置成一个很大的值
	m_cBestAnt.m_dbPathLength = DB_MAX;

	double a[N_CITY_COUNT];
	double b[N_CITY_COUNT];
	double c[N_CITY_COUNT];
	double d[N_CITY_COUNT];
	double e[N_CITY_COUNT];
	double f[N_CITY_COUNT];
	double h[N_CITY_COUNT];
	double ii[N_CITY_COUNT];
	double art[N_CITY_COUNT];
	double thickness[N_CITY_COUNT];
	double width[N_CITY_COUNT];
	double date[N_CITY_COUNT];
	double priority[N_CITY_COUNT];

	/*fstream file_in("./graph.txt");

	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		file_in >> x_Ary[i] >> y_Ary[i];
	}

	fstream file_in_demand("./demand.txt");
	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		file_in_demand >> demand[i];
	}*/
	fstream file_in("../data/input_data_300.txt");
	outfile.open("../input_data_300.txt");
	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		file_in >> id[i] >> a[i] >> b[i] >> c[i] >> d[i] >> demand[i] >> e[i] >> f[i] >> h[i] >> ii[i] >> art[i] >> thickness[i] >> width[i] >> date[i] >> priority[i];
	}
	//计算两两作业间difference
	double dbTemp = 0.0;
	for (int i = 1; i < N_CITY_COUNT; i++)
	{
		dbTemp = pow((a[i] - 0), 2) + pow((b[i] - 0), 2) + pow((c[i] - 0), 2) + pow((d[i] - 0), 2) + pow((e[i] - 0), 2) + pow((f[i] - 0), 2) + pow((h[i] - 0), 2)
			+ pow((ii[i] - 0), 2) + pow((art[i] - 0), 2) + pow((thickness[i] - 0), 2) + pow((date[i] - 0), 2) + pow((priority[i] - 0), 2) + pow((width[i] - 0), 2);
		dbTemp = pow(dbTemp, 0.5);
		//g_sourse[i] = ROUND(dbTemp);
		g_sourse[i] = dbTemp;

		for (int j = 0; j < N_CITY_COUNT; j++)
		{
			dbTemp = pow((a[i] - a[j]), 2) + pow((b[i] - b[j]), 2) + pow((c[i] - c[j]), 2) + pow((d[i] - d[j]), 2) + pow((e[i] - e[j]), 2) + pow((f[i] - f[j]), 2) + pow((h[i] - h[j]), 2)
				+ pow((ii[i] - ii[j]), 2) + pow((art[i] - art[j]), 2) + pow((thickness[i] - thickness[j]), 2) + pow((width[i] - width[j]), 2) + pow((date[i] - date[j]), 2) + pow((priority[i] - priority[j]), 2);
				//dbTemp = pow((x_Ary[i-1] - x_Ary[i]), 2) + pow((y_Ary[i-1] - y_Ary[i]), 2) + pow((z_Ary[i-1] - z_Ary[i]), 2) + pow((m_Ary[i-1] - m_Ary[i]), 2);
			dbTemp = pow(dbTemp, 0.5);
			//g_Distance[i][j] = ROUND(dbTemp);
			g_Distance[i][j] = dbTemp;
		}
	}

	//初始化环境信息素，先把作业间的信息素设置成一样
	//这里设置成1.0，设置成多少对结果影响不是太大，对算法收敛速度有些影响
	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		for (int j = 0; j < N_CITY_COUNT; j++)
		{
			g_Trial[i][j] = 1.0;
		}
	}
}


//更新环境信息素
void CTsp::UpdateTrial()
{
	//临时数组，保存各只蚂蚁在两两作业间新留下的信息素
	double dbTempAry[N_CITY_COUNT][N_CITY_COUNT];
	memset(dbTempAry, 0, sizeof(dbTempAry));			//先全部设置为0
	//计算新增加的信息素,保存到临时数组里
	int m = 0;
	int n = 0;
	for (int i = 0; i < N_ANT_COUNT; i++)					//计算每只蚂蚁留下的信息素
	{
		for (int j = 1; j < N_CITY_COUNT; j++)
		{
			m = m_cAntAry[i].m_nPath[j];
			n = m_cAntAry[i].m_nPath[j - 1];
			dbTempAry[n][m] = dbTempAry[n][m] + DBQ / m_cAntAry[i].m_dbPathLength;
			dbTempAry[m][n] = dbTempAry[n][m];
		}
		//最后作业和开始作业之间的信息素
		n = m_cAntAry[i].m_nPath[0];
		dbTempAry[n][m] = dbTempAry[n][m] + DBQ / m_cAntAry[i].m_dbPathLength;
		dbTempAry[m][n] = dbTempAry[n][m];
	}
	//==================================================================
	//更新环境信息素
	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		for (int j = 0; j < N_CITY_COUNT; j++)
		{
			g_Trial[i][j] = g_Trial[i][j] * ROU + dbTempAry[i][j]; //最新的环境信息素 = 留存的信息素 + 新留下的信息素
		}
	}
}

void CTsp::Search()
{
	char cBuf[256];				//打印信息用

	//在迭代次数内进行循环
	for (int i = 0; i < N_IT_COUNT; i++)
	{
		//每只蚂蚁搜索一遍
		for (int j = 0; j < N_ANT_COUNT; j++)
		{
			m_cAntAry[j].Search();
		}

		//保存最佳结果
		for (int j = 0; j < N_ANT_COUNT; j++)
		{
			if (m_cAntAry[j].m_dbPathLength < m_cBestAnt.m_dbPathLength)
			{
				m_cBestAnt = m_cAntAry[j];
			}
		}

		//更新环境信息素
		UpdateTrial();
		//输出目前为止找到的最优路径的长度
		//sprintf(cBuf, "\n[%d] %.0f", i + 1, m_cBestAnt.m_dbPathLength);
		//printf(cBuf);
	}
}

int main()
{
	start = clock();
	//用当前时间点初始化随机种子，防止每次运行的结果都相同
	time_t tm;
	time(&tm);
	unsigned int nSeed = (unsigned int)tm;
	srand(nSeed);

	//开始搜索
	CTsp tsp;
	tsp.InitData();						//初始化
	tsp.Search();						//开始搜索
	printf("\n");
	//printf("空载率为：%f\n", tsp.m_cBestAnt.UnDeliveryRate);
	printf("total cost：%f\n", tsp.m_cBestAnt.m_dbPathLength);
	//输出结果
	printf("\n最佳分批方案为 :\n");
	char cBuf[128];
	outfile << "total cost：" << tsp.m_cBestAnt.m_dbPathLength << endl;
	outfile << ("\n最佳分批方案为 :\n");
	int k = 1;
	cout << "批次 1 : ";
	outfile << "批次 1 : ";
	double sum_demand = 0;
	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		sum_demand += demand[i];
		if (sum_demand <= Capacity)
		{
			cout << tsp.m_cBestAnt.m_nPath[i] + 1 << " ";
			outfile << tsp.m_cBestAnt.m_nPath[i] + 1 << " ";
		}
		else
		{
			i--;
			sum_demand = 0;
			cout << endl;
			outfile << endl;
			k++;
			cout << "批次 " << k << " : ";
			outfile << "批次 " << k << " : ";
		}
	}

	finish = clock();
	std::cout << endl;
	outfile << endl;
	std::cout << "Total Running Time = " << (double)(finish - start) / CLOCKS_PER_SEC << " s" << endl;
	outfile << "Total Running Time = " << (double)(finish - start) / CLOCKS_PER_SEC << " s" << endl;

	double sum_demand1 = 0;
	//outfile << "[";
	for (int i = 0; i < N_CITY_COUNT; i++)
	{
		sum_demand1 += demand[i];
		if (sum_demand1 <= Capacity)
		{
			//outfile << "[" << id[tsp.m_cBestAnt.m_nPath[i]] << "], ";
			outfile << "作业号：" << id[tsp.m_cBestAnt.m_nPath[i]] << " ";
			outfile << a[tsp.m_cBestAnt.m_nPath[i]] << " ";
			outfile << b[tsp.m_cBestAnt.m_nPath[i]] << " ";
			outfile << c[tsp.m_cBestAnt.m_nPath[i]] << " ";
			outfile << d[tsp.m_cBestAnt.m_nPath[i]] << " ";
			outfile << endl;
		}
		else
		{
			i--;
			sum_demand1 = 0;
			//outfile << "[0], ";
			outfile << endl;
		}
	}
	//outfile << "[0]]" << endl;

	printf("\n\nPress any key to exit!");
	getchar();
	return 0;
}