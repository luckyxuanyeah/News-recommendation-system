from models.basemodel import basemodel
from models.UserCF import userCF
from models.ItemCB import itemCB
from models.NMF_model import NMF_model
from models.RNN_Encoder import MF_RNN
from models.LDA_MF import lda_MF
import pandas as pd

# data_df = pd.read_csv("Data/ml_100k/ml_train.txt", sep=',', header=-1)
# test_df = pd.read_csv("Data/ml_100k/ml_test.txt", sep=',', header=-1)
data_df = pd.read_csv("Data/train_data.txt", sep='\t', header=-1)
test_df = pd.read_csv("Data/test_data.txt", sep='\t', header=-1)
# ctx_df = pd.read_csv("Data/news_context_id.txt", sep='\t', header=-1)
# model = NMF_model(data_df, 20)
# model = userCF(data_df, 5)
# model = itemCB(data_df, 20)
# model = MF_RNN(data_df, ctx_df, 10)
model = lda_MF(data_df, 20)
model.train(pairwise=True)
model.evaluation(test_df)
