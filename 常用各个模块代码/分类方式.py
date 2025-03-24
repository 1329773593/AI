

def generate_class_labels_by_percentile(Y, num_classes=5):
    """
    根据目标 Y 的分位数生成类别标签。
    参数：
      Y: 一维 numpy 数组
      num_classes: 类别数（例如5）
    返回：
      labels: 对应的类别标签（0 ~ num_classes-1）
      boundaries: 用于说明类别划分的分位数边界
    """
    boundaries = np.percentile(Y, np.linspace(0, 100, num_classes + 1))
    labels = np.digitize(Y, boundaries[1:-1], right=True)  # 生成0~num_classes-1标签
    return labels, boundaries


def generate_class_labels_by2(Y):
    """
    根据2的次幂生成分类标签，确保最大边界是距离样本最大值最近的更大次幂。

    参数：
        Y: 一维 numpy 数组

    返回：
        labels: 对应的类别标签（0 ~ n_classes-1）
        boundaries: 按2次幂扩展的边界列表（包含0和终止值）
    """
    Y = np.asarray(Y)
    if Y.size == 0:
        return np.array([]), [0, 1]  # 空数据默认返回0-1边界

    Y_max = Y.max()

    # 生成2的次幂边界，直到覆盖最大值
    boundaries = [0]
    current_power = 1  # 初始为2^0=1

    # 生成所有<=Y_max的次幂
    while current_power <= Y_max:
        boundaries.append(current_power)
        current_power *= 2

    # 添加第一个超过Y_max的次幂作为终止边界
    boundaries.append(current_power)

    # 生成分类标签 (每个区间的左闭右开)
    bins = boundaries[1:-1]  # 忽略首尾的0和终止值
    labels = np.digitize(Y, bins=bins, right=False)  # right=False表示左闭右开

    return labels, boundaries


import numpy as np
from sklearn.cluster import KMeans


def generate_class_labels_byCluster(Y, n_clusters=4):
    """
    使用 K-Means 聚类生成分类标签，并确定类别边界。

    参数：
        Y: 一维 numpy 数组
        n_clusters: 聚类簇数，决定类别数量

    返回：
        labels: 聚类后的类别标签（0 ~ n_clusters-1）
        boundaries: 计算得到的类别边界（包含最小值和最大值）
    """
    Y = np.asarray(Y).reshape(-1, 1)  # 转换为列向量
    if Y.size == 0:
        return np.array([]), [0, 1]  # 空数据返回默认边界

    # 使用 K-Means 进行聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(Y)

    # 获取聚类中心，并按值排序
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

    # 计算类别边界（取均值作为分界点）
    boundaries = [Y.min()]  # 最小值作为起始边界
    for i in range(len(cluster_centers) - 1):
        boundary = (cluster_centers[i] + cluster_centers[i + 1]) / 2
        boundaries.append(boundary)
    boundaries.append(Y.max())  # 最大值作为终止边界

    # 生成分类标签
    labels = np.digitize(Y.flatten(), bins=boundaries[1:-1], right=False)

    return labels, boundaries
