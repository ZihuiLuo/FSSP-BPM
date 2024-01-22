import numpy as np
import math
from data.result_evaluation import path2load, path2csv, print_reward

'''
# 完全没考虑容量
def K_mean_without_capacity(data, k=10, input_dim=5, decode_len=100):
    data = data[:, :-1]
    centers = np.array([data[i] for i in range(k)])
    while (True):
        # point_id : center_id
        center_p = [np.argmin([np.sum((point - center) ** 2) ** (1 / 2) for center in centers])
                    for point in data]
        # center_id : list of point_id
        plist_center = [[] for _ in range(k)]
        for i in range(len(center_p)):
            plist_center[center_p[i]].append(i)
        new_centers = np.array([np.mean([data[point] for point in row], axis=0) for row in plist_center])
        if (np.sum(np.abs(centers - new_centers)) < 1e-5):
            break
        centers = new_centers
    return plist_center
'''

'''
# 考虑了点的拆分，但有BUG
def K_mean_with_capacity_decompose(data, capacity=50, input_dim=5, decode_len=100):
    demand = np.array(data[:, -1])
    data = data[:, :-1]
    k = math.ceil(np.sum(demand) / capacity)
    demand_desc = np.argsort(-demand)
    centers = np.array([data[demand_desc[i]] for i in range(k)])
    for i in range(100):
        center_remain = np.array(k * [capacity], dtype=np.float)
        # point_id : list of center_id(一个点可以有多个类别，当被拆分时)
        center_p = [[] for _ in range(len(data))]
        have_to_give = np.copy(demand)
        while have_to_give.any() != 0:
            i = np.argsort(-have_to_give)[0]
            indexes = np.argsort([(np.sum((data[i] - center) ** 2) ** (1 / 2)) / have_to_give[i] for center in centers])
            # BUG:！！！遍历的顺序和转为path的顺序冲突
            for index in indexes:
                if center_remain[index] >= have_to_give[i]:
                    center_remain[index] -= have_to_give[i]
                    center_p[i].append(index)
                    have_to_give[i] = 0
                else:
                    have_to_give[i] -= center_remain[index]
                    center_p[i].append(index)
                    center_remain[index] = 0
                if have_to_give[i]<=0:
                    break
        # center_id : list of point_id
        plist_center = [[] for _ in range(k)]
        for i in range(len(center_p)):
            for index in center_p[i]:
                plist_center[index].append(i)
        new_centers = np.array([np.mean([data[point] for point in row], axis=0) for row in plist_center])
        if (np.sum(np.abs(centers - new_centers)) < 1e-5):
            break
        centers = new_centers
    return plist_center
'''


# 将分批结果的表示从批次类别转为VRP中的路径
def plist_center2path(plist_center, data, input_dim=5, decode_len=100):
    input_dim -= 1
    path = []
    for patch in [[list(data[point])[:input_dim] for point in row] for row in plist_center]:
        for point in patch:
            path.append(point)
        path.append(input_dim * [0.])
    for i in range(decode_len - len(path)):
        path.append(input_dim * [0.])
    return path


# 不考虑拆分
def K_mean_with_capacity(data, capacity=50, input_dim=5, decode_len=100):
    demand = np.array(data[:, -1])
    data = data[:, :-1]
    k = math.ceil(np.sum(demand) / capacity)
    demand_desc = np.argsort(-demand)
    centers = np.array([data[demand_desc[i]] for i in range(k)])
    for i in range(100):
        center_remain = np.array(k * [capacity], dtype=np.float)
        # point_id : center_id
        center_p = len(data) * [-1]
        have_to_give = np.copy(demand)
        while have_to_give.any() != 0:
            i = np.argsort(-have_to_give)[0]
            indexes = np.argsort([(np.sum((data[i] - center) ** 2) ** (1 / 2)) for center in centers])
            for index in indexes:
                if center_remain[index] >= have_to_give[i]:
                    center_remain[index] -= have_to_give[i]
                    center_p[i] = index
                    have_to_give[i] = 0
                    break
        # center_id : list of point_id
        plist_center = [[] for _ in range(k)]
        for i in range(len(center_p)):
            plist_center[center_p[i]].append(i)
        # print(plist_center)
        new_centers = np.array([np.mean([data[point] for point in row], axis=0) for row in plist_center])
        if (np.sum(np.abs(centers - new_centers)) < 1e-5):
            break
        centers = new_centers
    return plist_center

if __name__=='__main__':
    input_dim = 5
    file = open("test.txt", "r")
    file = np.array([list(map(float, line.strip().split('\t'))) for line in file])
    data = file[:50]
    plist_center = K_mean_with_capacity(data, input_dim=input_dim)
    path = np.array(plist_center2path(plist_center, data),dtype=np.float32)
    path2csv(path, path2load(data, path),'K-means_result.csv')
    print_reward(path)
