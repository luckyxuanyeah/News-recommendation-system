import numpy as np
from models.basemodel import basemodel
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

class NMF_model(basemodel):
    '''NMF非负矩阵分解，设置隐式空间维度'''
    def __init__(self, user_news_df, dimk):
        basemodel.__init__(self, user_news_df)
        self.dimk = dimk
        self.nmf = NMF(n_components=self.dimk) #非负矩阵分解 隐式空间维度为dimk

    def train(self):
        self.user_factors = self.nmf.fit_transform(self.ui_mat)
        self.item_factors = self.nmf.components_

    def predict(self, user, item):
        """"""
        u_f = self.user_factors[user, :]
        i_f = self.item_factors[:, item] 
        prediction = np.matmul(u_f, i_f)
        return prediction
