from collections import defaultdict


class MetroStation:
    def __init__(self, parameters, env, id, name, dis2nextstation):
        # 初始化函数，用于创建 MetroStation 类的实例并设置初始属性值
        # 参数：parameters，包含系统参数的字典
        # 参数：env，仿真环境
        # 参数：id，地铁站点的唯一标识符
        # 参数：name，地铁站点的名称
        # 参数：dis2nextstation，到下一站的距离
        self.parameters = parameters
        self.env = env
        self.id = id
        self.name = name
        self.dis2nextstation = dis2nextstation
        self.info = defaultdict(dict)       # 存储地铁站点信息的字典

    def reset(self):
        self.info = defaultdict(dict)       # 重置站点信息
        # 通过重新创建一个空的 defaultdict 来清空 info 字典，从而清除之前存储的所有站点信息。
