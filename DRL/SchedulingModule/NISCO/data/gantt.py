import time
import plotly as py
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import random

def randomcolor():

    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def random_color(num):
    return tuple(randomcolor() for _ in range(num))


order_num = 50

df = pd.read_csv("gantt_data.csv")
# 每道工序开始加工时间
n_start_time=np.array(df["Start"].astype('int')).tolist()
# length, 对应于每个图形在x轴方向的长度
# duration, time, of, every, task, , //每个工序的持续时间,以小时为单位
n_duration_time=np.array(df["Duration"].astype('int')).tolist()

# y轴, 对应于画图位置的起始坐标y
# bay, id, of, every, task, , ==工序数目，即在哪一行画线 //第几个machine
n_bay_start= np.array(df["Machine"]).tolist()
# 工序号，可以根据工序号选择使用哪一种颜色，相同作业用相同颜色表示
n_job_id = np.array(df["job_id"]).tolist()

# op = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] #工10个作业
op=np.arange(order_num).tolist()

print("n_job_id:",n_job_id)
print("start_time:",n_start_time)
# 10个作业用10种不同的颜色表示
colors = random_color(order_num)
# colors = ('rgb(46, 137, 205)',
#           'rgb(114, 44, 121)',
#           'rgb(198, 47, 105)',
#           'rgb(58, 149, 136)',
#           'rgb(107, 127, 135)',
#           'rgb(46, 180, 50)',
#           'rgb(150, 44, 50)',
#           'rgb(70, 47, 150)',
#           'rgb(58, 100, 180)',
#           'rgb(20, 127, 20)')


# millis_seconds_per_minutes = 1000 * 60
millis_seconds_per_minutes = 1000 * 60

st='2020-08-06 08:00:00' # 以这个作为当前时间
timeArray=time.strptime(st,"%Y-%m-%d %H:%M:%S")
timeStamp = int(time.mktime(timeArray))
start_time = timeStamp * 1000  #time.time()返回当前时间的时间戳（1970纪元后经过的浮点秒数）。

job_sumary = {}


# 获取工件对应的第几道工序
def get_op_num(job_num):
    index = job_sumary.get(str(job_num))
    new_index = 1
    if index:
        new_index = index + 1
    job_sumary[str(job_num)] = new_index
    return new_index


import random



def create_draw_defination():
    df = []
    for index in range(len(n_job_id)):
        operation = {}
        # 机器，纵坐标
        operation['Task'] = 'M' + str(n_bay_start.__getitem__(index)) #获取该工序的机器号
        #operation['Task'] = n_bay_start.__getitem__(index)
        operation['Start'] = start_time.__add__(n_start_time.__getitem__(index) * millis_seconds_per_minutes) #获取该工序的开始时间
        operation['Finish'] = start_time.__add__(
            (n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)) * millis_seconds_per_minutes)#获取该工序的完成时间
        # 工件，
        job_num = op.index(n_job_id.__getitem__(index)) + 1 #该工序属于第几个作业
        operation['Resource'] = 'J' + str(job_num)
        df.append(operation) # 放入一个dict{'Task':'M1','Start':0,'Finish':10,'Resource':'J1'}
    df.sort(key=lambda x: x["Task"], reverse=True) # 根据Task（机器编号）倒序排列
    print(df)
    return df


def draw_prepare():
    df = create_draw_defination()
    return ff.create_gantt(df, colors=colors, index_col='Resource',
                           title='最小化makespan', show_colorbar=True,
                           group_tasks=True, data=n_duration_time,
                           showgrid_x=True, showgrid_y=True)


def add_annotations(fig):
    y_pos = 0
    for index in range(len(n_job_id)):
        # 机器，纵坐标
        y_pos = n_bay_start.__getitem__(index)

        x_start = start_time.__add__(n_start_time.__getitem__(index) * millis_seconds_per_minutes)
        x_end = start_time.__add__(
            (n_start_time.__getitem__(index) + n_duration_time.__getitem__(index)) * millis_seconds_per_minutes)
        x_pos = (x_end - x_start) / 2 + x_start

        # 工件，
        job_num = op.index(n_job_id.__getitem__(index)) + 1
        text = 'J(' + str(job_num) + "," + str(get_op_num(job_num)) + ")=" + str(n_duration_time.__getitem__(index))
        # text = 'T' + str(job_num) + str(get_op_num(job_num))
        text_font = dict(size=14, color='black')
        fig['layout']['annotations'] += tuple(
            [dict(x=x_pos, y=y_pos, text=text, textangle=-30, showarrow=False, font=text_font)])


def draw_fjssp_gantt():
    fig = draw_prepare()
    add_annotations(fig)
    py.offline.plot(fig, filename='min_maxspan-gantt-picture.html')


if __name__ == '__main__':
    draw_fjssp_gantt()

