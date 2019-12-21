import pandas as pd
from feature_engineering import calculate_feature

test_path = './jet_simple_data/simple_test_R04_jet.csv'
train_path = './jet_simple_data/simple_train_R04_jet.csv'

train = pd.read_csv(train_path,nrows=100)
test = pd.read_csv(test_path,nrows=100)

print('finish data read')

#### add features #####
train, test = calculate_feature(train, test)

# train.to_csv("./data_fea/train_fea_1.csv", index=False)
# test.to_csv("./data_fea/test_fea_1.csv", index=False)

