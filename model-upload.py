import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#import drop_feature
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_auc_score

#### select feature
params = {"eta":0.05,
         "num_class":4,
         "max_depth":8,
         "min_child_weight":1,
         "gamma":0.001,
         "subsample":0.8,
         "colsample_bytree":0.8,
         "objective":"multi:softmax",
         "scale_pos_weight":1,
         "seed":2019,
         "tree_method":"gpu_hist",
         "n_gpus":-1,
         "gpu_id":0,
         "n_jobs":4}

# clf = xgb.XGBClassifier(learning_rate=0.01, n_estimators=10, max_depth=8,
#                     min_child_weight=3, gamma=0.2, subsample=0.7, colsample_bytree=1.0,
#                     objective='multi:softmax', nthread=4, scale_pos_weight=1, seed=2019,
#                     tree_method='gpu_hist', n_gpus=-1, gpu_id=0)

label_dic = {1: 0, 4: 1, 5: 2, 21: 3}

do_predict = True
do_valid = True
do_select = False
do_submit= True

sub_path = "sub1221_3.csv"
keep_num = 100

def eval_func(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_one_hot = label_binarize(y_true, classes=np.arange(4))
    y_pred_one_hot = label_binarize(y_pred, classes=np.arange(4))
    score = roc_auc_score(y_one_hot, y_pred_one_hot, average='macro')
    # print('auc:', score)
    return "auc_score",score

def submit_result(test_id, results):
    dd = {0: 1, 1: 4, 2: 5, 3: 21}
    def sub_process(x):
        x = dd[x]
        return x
    sub = pd.DataFrame()
    # pred = results.argmax(axis=1)
    sub['label'] = list(results)
    print(sub['label'])
    sub['label'] = sub['label'].apply(sub_process)
    sub['id'] = list(test_id)
    sub_reverse = pd.DataFrame({'id': sub['id'].tolist(), 'label': sub['label'].tolist()})
    sub_reverse.to_csv(sub_path, index=False)


if __name__ == '__main__':
    train_path = "./data_fea/train_fea_1.csv"
    train = pd.read_csv(train_path)
    print('label' in list(train.columns))
    print(train.shape)

    train_y = train['label'].apply(lambda x: label_dic[x])
    train_y = np.array(list(train_y))
    train.pop('label')
    train.pop('jet_id')
    train.pop('event_id')

    ### delete useless features
    # for fea in drop_feature.drop_list:
    #     try:
    #         train.pop(fea)
    #     except:
    #         continue
    print("train shape now:",train.shape)

    if do_select:
        fea_path = "./imp.csv"
        fea = pd.read_csv(fea_path)
        global fea_list
        fea_list = list(fea['Features'])
        train = train[fea_list[:keep_num]]
    train_x = train.values

    # clf.fit(X_train, y_train, eval_metics='mlogloss', eval_set=[[X_train, y_train], [X_dev, y_dev]])
    X_train,X_dev,y_train,y_dev = train_test_split(train_x,train_y,test_size=0.2,stratify=train_y)
    dtrain = xgb.DMatrix(X_train,label=y_train,nthread=-1)
    ddev = xgb.DMatrix(X_dev,label=y_dev,nthread=-1)
    bst = xgb.train(params,dtrain,num_boost_round=5000,early_stopping_rounds=50,verbose_eval=True,
                    feval=eval_func,evals=[(dtrain,'train'),(ddev,'eval')],evals_result={"eval_metic":"auc_score"},
                    maximize=True)

    ### 全量数据训练
    # dtrain = xgb.DMatrix(train_x,label=train_y)
    # bst_cv = xgb.cv(params,dtrain,num_boost_round=10,
    #                 nfold=5,stratified=True,
    #                 feval=eval_func,early_stopping_rounds=50)
    # print(bst_cv)
    # print(list(bst_cv.columns))
    # print("test auc mean:",bst_cv["test-auc_score-mean"].mean())
    # print("train auc mean:",bst_cv["train-auc_score-mean"].mean())
    if do_valid:
        print("validating")
        # 均匀切分验证集-stratify
        # X_train, X_dev, y_train, y_dev = train_test_split(train_x, train_y, test_size=0.2, stratify=train_y)
        ddev = xgb.DMatrix(X_dev, label=y_dev)
        y_pred = bst.predict(ddev)

        cols = list(train.columns)
        valid_df = pd.DataFrame(columns=cols, data=X_dev)
        tmp_df = pd.DataFrame({"true_label": y_dev, "predict_label": y_pred})
        valid_df = pd.concat([valid_df, tmp_df], axis=1)
        valid_df.to_csv("validation.csv", index=False)

    if do_predict:
        print("predicting")
        test_path = "./data_fea/test_fea_1.csv"
        test = pd.read_csv(test_path)
        test.pop("event_id")
        test_id = test.pop("jet_id")

        # for fea in drop_feature.drop_list:
        #     try:
        #         test.pop(fea)
        #     except:
        #         continue

        if do_select:
            test = test[fea_list[:keep_num]]
        test_x = test.values
        dtest = xgb.DMatrix(test_x,nthread=-1)

        res = bst.predict(dtest)
        if do_submit:
            submit_result(test_id=test_id, results=res)
            print("submit done!")


        ### watch feature importance
        # feat = train.columns.tolist()
        # feat_imp = clf.feature_importances_
        # res_df = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=False)
        # print(res_df)
        # res_df = res_df.sort_values(by=["Importance"],ascending=False)
        # res_df.to_csv("./importance.csv",index=False)