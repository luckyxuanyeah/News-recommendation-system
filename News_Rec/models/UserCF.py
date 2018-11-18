# 计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
# 导入基本模块的basemodel
from models.basemodel import basemodel
# 压缩稀疏矩阵的行
from scipy.sparse import csr_matrix
# 对多维数组对象的支持 支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
import numpy as np

class userCF(basemodel):
    """基于用户的协同过滤方法，由于点击数据大多都只有0,1，直接用与当前用户最相似的K个用户的评分加权预测对新项目的评分"""
    def __init__(self, user_news_df, knn):
        basemodel.__init__(self, user_news_df) # 利用基本类模型进行初始化 得到用户号-新闻号点击矩阵
        self.user_sim = cosine_similarity(self.ui_mat)# 用余弦相似度计算用户相似性 得相似度矩阵 输入为用户号-新闻号点击矩阵
        self.knn = knn
    # 训练
    def train(self):
        # 将矩阵中每个用户与其他用户的相似度从大到小排序 返回数组按照行（axis=1）排序从大到小（-self.user_sim）的索引值
        self.sorted_sim = np.argsort(-self.user_sim, axis=1)
    # 预测方法（参数 要推荐给的user 要推荐的某篇新闻item）
    def predict(self, user, item):
        # 找出用户的K近邻用户集
        user_k = self.sorted_sim[user] #取出排好序的矩阵对应user的那一行
        user_topK = []
        k = 0
        for u in user_k: #对user_k中的每一个user循环
            if k >= self.knn: #如果k比knn大，那么就退出循环
                break
            if self.ui_mat[u, item] > 0: #如果在用户号-新闻号点击矩阵中的u对应item点击
                user_topK.append(u) #将user加入topK中
                k += 1 #加入了的k近邻数量加1
        if len(user_topK) == 0 or self.user_sim[user_topK[0], user] < 1e-10: #如果没有相似的用户点击了item或者和用户相似的第一个用户余弦相似度小于某个阈值，则返回0
            return 0

        prediction = 0 #打分值
        sim_sum = 0 #相似用户个数
        for uk in user_topK: #与user相似的k近邻用户依次循环
            sim_sum += self.user_sim[user, uk] #相似度矩阵中取出二者的相似度 sim_sum表示相似度之和
            uki = self.ui_mat[uk, item] #取出用户号-新闻号点击矩阵中的值
            prediction += self.user_sim[user, uk] * uki #预测结果为相似度乘点击值
        if sim_sum < 1e-10: #如果总的相似度之和小于某值 打分为0
            prediction = 0
        else: #否则打分为加权之和／相似度之和
            prediction = prediction / sim_sum 
        return prediction
