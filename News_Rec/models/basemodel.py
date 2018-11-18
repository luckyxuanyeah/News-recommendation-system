"""Basemodel 由所有模型继承重写关键方法"""
# 对多维数组对象的支持 支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
import numpy as np
import math
# 压缩稀疏矩阵的行
from scipy.sparse import csr_matrix
import random
# 基本模型类
class basemodel():
    # 初始化
    def __init__(self, user_news_df):
        self.ui_df = user_news_df #原始数据按行标签索引 在preprocess取用户号-新闻号-时间 存user_news_df 这里存ui_df
        self.USER_NUM = 10000 #10000  ml : 943 用户数
        self.ITEM_NUM = 6183 #6183  ml : 1682 新闻数
        self.user_flag = [False for _ in range(self.USER_NUM)] #设置用户个数个标志false
        self.ui_mat = self.get_mat(user_news_df) #get_mat函数转化为矩阵（ui_mat 用户新闻点击矩阵）才能后面操作
        self.item_clicksum = np.sum(self.ui_mat, axis=0) #将矩阵按照列相加 得到每条新闻总的点击次数
        self.item_clicksum = np.argsort(-self.item_clicksum) #按总点击次数从大到小将新闻排序 key：num
    # 定义get_mat函数
    def get_mat(self, ui_df):
        read_sum = ui_df.shape[0] #取用户号-新闻号-时间的行数（一共的阅读数量）
        user_row = np.array([self.ui_df.iloc[i, 0] for i in range(read_sum)]) #循环每行 取第一列 Numpy提供ndarray对象：存储单一数据类型的多维数组（这里是read_sum维数组）得到用户行
        item_col = np.array([self.ui_df.iloc[i, 1] for i in range(read_sum)]) #循环每行 取第二列 read_sum维 得新闻列
        mat = np.zeros((self.USER_NUM, self.ITEM_NUM)) #返回给定形状和类型的矩阵（用户个数为行 新闻个数为列）
        for i in range(read_sum): #循环总的阅读数量
            self.user_flag[user_row[i]] = True #user_row[0 1]第一个用户对应用户号0 0 此位置置true
            mat[user_row[i], item_col[i]] = 1 #矩阵中user_row[0 1 2 3 4]的（0 4）处置1，用户0看过文章4
        return mat #返回矩阵（用户 新闻点击矩阵）
    # 训练（每种模型都有自己训练方法）
    def train(self):
        pass
    # 预测
    def predict(self, user, item):
        """预测user对item的评分"""
        prediction = self.item_clicksum[item] #选择这篇新闻对应的总点击次数即为预测（不同模型不一样）
        return prediction
    # 预测topK
    def predict_topK(self, user, K):
        """生成给用户user推荐的K个项目列表"""
        if self.user_flag[user] == False: #如果用户的标志为false
            return self.item_clicksum[0:K] #就返回给用户点击率最高的K篇新闻
            # return random.sample(list(self.item_clicksum[0: 100]), K)
        user_rating = self.ui_mat[user, :] #取出用户新闻点击矩阵中的user对应的那一行
        rec_list = dict() #新建一个字典
        # k = 0
        for item in range(self.ITEM_NUM): #循环每篇新闻
            # print(k)
            # k += 1
            if user_rating[item] == 0: #如果用户之前没有点击过这篇新闻 点击过什么都不做
                rec_list[item] = self.predict(user, item) #使用对应的预测方法预测 将新闻号对应预测打分值加入字典（0 8）
        rec_topK = sorted(rec_list.items(), key=lambda e: e[1], reverse=True) #返回新字典 选择打分进行排序 降序
        return [rec_topK[i][0] for i in range(K)] #返回字典中的K篇新闻
    # 评价指标
    def evaluation(self, test_df, topn = 10):
        read_sum = test_df.shape[0]
        user_row = np.array([test_df.iloc[i, 0] for i in range(read_sum)])
        item_col = np.array([test_df.iloc[i, 1] for i in range(read_sum)])
        read_score = np.array([1 for i in range(read_sum)])
        self.test_mat = csr_matrix((read_score, (user_row, item_col)), shape=(self.USER_NUM, self.ITEM_NUM))
        ui_dict = dict()
        for i in range(test_df.shape[0]):
            if test_df.iloc[i, 0] not in ui_dict.keys():
                ui_dict[test_df.iloc[i, 0]] = [test_df.iloc[i, 1]]
            else:
                ui_dict[test_df.iloc[i, 0]].append(test_df.iloc[i, 1])
        # 计算MAP和NDCG
        mAP = 0
        mPrecision = 0
        eval_user = 0
        user_sum = len(ui_dict)
        each = user_sum / 4
        start = each * 0
        end = each * 1
        for user, itemlist in ui_dict.items():
            # if self.user_flag[user] == False:
            #     user_sum -= 1
            #     continue
            eval_user += 1
            if eval_user % 1 == 0:
                print("Eval process: %d / %d" % (eval_user, user_sum))
            if eval_user > user_sum:
                break
            # if eval_user > end or eval_user <= start:
            #     continue
            predlist = self.predict_topK(user, topn)
            reclist = list(set(itemlist))
            mPrecision += self.cal_PN(predlist, reclist)
            mAP += self.cal_AP(predlist, reclist)
            # nDCG += self.cal_DCG(user, predlist, reclist)
        mPrecision /= user_sum
        mAP /= user_sum
        # nDCG /= user_sum
        print("Top%d Rec Result:" % topn)
        print("mPrecision: %g  mAP: %g" % (mPrecision, mAP))
    # 计算PN
    def cal_PN(self, predlist, reclist, n=10):
        p = 0
        for pred in predlist:
            if pred in reclist:
                p += 1
        p /= n
        return p
    # 计算AP
    def cal_AP(self, predlist, reclist):
        pos = 1
        rel = 1
        ap = 0
        for i in range(len(reclist)):
            if reclist[i] in predlist:
                ap += rel / pos
                rel += 1
            pos += 1
        ap /= len(reclist)
        return ap
    # 计算DCG
    def cal_DCG(self, user, predlist, reclist, n=10):
        pred_rank = [self.test_mat[user, item] for item in predlist]
        rec_rank = [self.test_mat[user, item] for item in reclist]
        dcg = pred_rank[0]
        idcg = rec_rank[0]
        for i in range(1, len(pred_rank)):
            dcg += pred_rank[i] / math.log2(i + 1)
        for i in range(1, len(rec_rank)):
            idcg += rec_rank[i] / math.log2(i + 1)
        ndcg = dcg / idcg
        return ndcg
