# 计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
from models.basemodel import basemodel
# 压缩稀疏矩阵的行
from scipy.sparse import csr_matrix
# 对多维数组对象的支持 支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
import numpy as np

class itemCF(basemodel):
    "基于项目内容的K近邻方法，计算相似项目使用新闻内容的tfidf值，而不采用评分计算 该用户对最近似K个项目的评分加权预测对该项目的评分"
    def __init__(self, user_news_df, knn):
        basemodel.__init__(self, user_news_df) # 利用基本类模型进行初始化 得到用户号-新闻号点击矩阵
        self.knn = knn # 近邻的个数
    # 训练
    def train(self):
        self.item_sim = np.loadtxt('Data/news_sim.mat') # 读入新闻相似度矩阵，每一行格式相同
        self.sorted_sim = np.argsort(-self.item_sim, axis=1) # 对相似度矩阵按行进行从大到小排序（结果为最相似到最不相似）
        print("Sort Complish")
    # 预测  找出K近邻项目集 参数 user item
    def predict(self, user, item):
        item_k = self.sorted_sim[item] # 从相似度矩阵中取出来这篇新闻的所在行
        item_topK = [] # 建立一个空列表
        k = 0 #计数
        for i in item_k: # 对每篇相似新闻循环
            if k >= self.knn: # 如果k比knn大，那么就退出循环
                break
            if self.ui_mat[user, i] > 0: # 如果点击矩阵中用户点击了这篇相似的文章
                item_topK.append(i) # 将新闻加入到topk中
                k += 1 # k加1
        if len(item_topK) == 0 or self.item_sim[item_topK[0], item] < 1e-10: # 如果没有相似的新闻或最相似新闻小于阈值
            return 0 # 返回0
        
        prediction = 0
        sim_sum = 0
        for ik in item_topK: # 对当前item的k近邻进行循环，取出一个循环值ik
            sim_sum += self.item_sim[item, ik] # sim_sum表示相似度之和
            uki = self.ui_mat[user, ik] # 取出user item关系矩阵中的这个user对这个新闻的打分值（1）
            prediction += self.item_sim[item, ik] * uki # 相似度*打分值相加
        return sim_sum
        if sim_sum < 1e-10: # 如果相似度之和小于阈值
            prediction = 0 # 返回打分0
        else:
            prediction = prediction / sim_sum # 打分值为打分加权和／相似度之和
        return prediction
# 计算新闻的余弦相似度
def cal_sim():
    tfidf_mat = np.loadtxt('Data/tfidf.mat') # 加载tfidf.mat稀疏矩阵（A：0 0 0.23 0.34 0.28 0.1 0.6）
    print(np.shape(tfidf_mat)) # 打印出矩阵的形状
    item_sim = cosine_similarity(tfidf_mat) # 计算新闻-新闻的余弦相似度 存item_sim中
    print("Sim Complish")
    np.savetxt('Data/item_sim.mat', item_sim) # 将相似度存入item_sim.mat
