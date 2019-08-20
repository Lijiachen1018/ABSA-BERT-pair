import numpy as np
import pandas as pd
from sklearn import metrics

def get_y_true():
    y_true = pd.read_csv(train_path,names = list(range(num_asp+1)))#第一列是text，后面20列才是aspects
    y_true = y_true.loc[:,range(1,num_asp+1)].values
    return y_true

def get_y_pred():
    with open(pred_path,"r", encoding="utf-8") as f:
        s = f.readlines()
    logits = [ float(i.strip().split()[2]) for i in s]
    reshaped_logits = np.reshape(logits,(num_pol,num_asp,-1),order = 'F')
    matrix_shape = reshaped_logits.shape
    num_smpl = matrix_shape[2]
    pred = []
    for n in range(num_smpl):
        smpl_logits = reshaped_logits[:,:,n]
        indexs = np.argmax(smpl_logits,axis = 0)
        pred += [polarity[i] for i in indexs]
    return np.reshape(pred, (num_smpl,num_asp))

if __name__ == '__main__':

    y_pred = get_y_pred()
    y_true = get_y_true()
    aspect_name_list = ['交通便利程度',
                        '与商圈的距离', '是否容易找到',
                        '排队等待时间', '服务人员的态度',
                        '是否容易停车', '点菜或上菜速度', '价格水平',
                        '性价比', '折扣力度', '店面装修',
                        '环境是否嘈杂', '环境空间大小', '环境是否整洁',
                        '分量', '味道', '菜品外观', '菜品推荐程度',
                        '本次消费感受', '再次消费意愿']
    train_path = '/Data_Center/美团评论ABSA/2018_zh_ABSA_dataset/meituan_absa_train_data.csv'
    pred_path = '/ABSA-BERT-pair/results/ch/QA_B/test_ep_6.txt'
    num_asp = len(aspect_name_list)
    num_pol = 4
    polarity = {0: -2, 1: 0, 2: 1, 3: -1}
    for aspect_idx in range(num_asp):
        print('\n\n',aspect_name_list[aspect_idx],'\n')
        print(metrics.classification_report(y_true[:,aspect_idx],y_pred[:,aspect_idx]))

