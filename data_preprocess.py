import pandas as pd
import torchtext.data as Tdata


class Preprocessor(object):
    """docstring for Preprocessor"""

    def __init__(self, datadir: str, train_path: str, test_path: str, col_names: list, aspect_name_list: list,
                 table_format='csv'):
        super(Preprocessor, self).__init__()
        self.datadir = datadir
        self.table_format = table_format
        self.train_path = train_path
        self.test_path = test_path
        self.col_names = col_names
        self.aspect_list = aspect_name_list

    def field_gen(self):
        assert 'text' in self.col_names, 'No text column in data'
        data_fields = []
        for i in self.col_names:
            data_fields.append((i, Tdata.Field()))
        return data_fields

    def read_raw_to_torchtext(self):
        train_dataset, test_dataset = Tdata.TabularDataset.splits(
            path=self.datadir,
            format=self.table_format,
            train=self.train_path,
            test=self.test_path,
            fields=self.field_gen()
        )
        return train_dataset, test_dataset

    def torchtext2df(self, torchtext_dataset):
        df = []
        for single_sample in torchtext_dataset:
            dic = {}
            dic['text'] = ' '.join(getattr(single_sample, 'text'))
            for aspect in self.aspect_list:
                dic[aspect] = getattr(single_sample, aspect)[0]
            df.append(dic)
        return pd.DataFrame(df)

    def process_data(self):
        train_dataset, test_dataset = self.read_raw_to_torchtext()
        train_dataset = self.torchtext2df(train_dataset)
        test_dataset = self.torchtext2df(test_dataset)
        return train_dataset, test_dataset


class ABSA_data_gen(object):
    '''
    guid
    label:
        Multi:  {-2:"unmentioned", 0:"neutural", 1:"positive", -1:"negative"}
        Binary: {0:'no',1:'yes'}
    text_a: synatic text
    text_b: original text

    '''

    def __init__(self, task_name, sentiment_label, train_df, test_df, aspect_list):
        self.task_name = task_name
        self.dataset = {'train': train_df, 'test': test_df}
        self.label = sentiment_label
        self.template = {
            'QA_M': "你认为在%s方面怎么样?",
            'QA_B': "你对于%s方面的评价是%s吗？",
            'NLI_M': '%s',
            'NLI_B': '%s-%s'}
        self.aspect_list = aspect_list

    def save_to_tsv(self, ABSA_tsv_DIR):
        for train_test in ['train', 'test']:
            df = pd.DataFrame(self.task_data_gen(self.dataset[train_test]))
            path = f'{ABSA_tsv_DIR}{train_test}_{self.task_name}.tsv'
            df.to_csv(path, sep='\t', index=False, header=False)

    def task_data_gen(self, df):
        result = []
        for index, row in df.iterrows():
            for aspect in self.aspect_list:
                if self.task_name[-1] == "M":
                    template = self.template[self.task_name] % aspect
                    result.append({'guid': index, 'label': self.label[row[aspect]],
                                   'text_a': template, 'text_b': row['text']})
                else:
                    for key, polarity in self.label.items():
                        template = self.template[self.task_name] % (' '.join(aspect.split('_')), polarity)
                        tag = 1 if key == row[aspect] else 0
                        result.append({'guid': index, 'label': tag,
                                       'text_a': template, 'text_b': row['text']})
        return result


if __name__ == '__main__':
    aspect_name_list = ['交通便利程度',
                        '与商圈的距离', '是否容易找到',
                        '排队等待时间', '服务人员的态度',
                        '是否容易停车', '点菜或上菜速度', '价格水平',
                        '性价比', '折扣力度', '店面装修',
                        '环境是否嘈杂', '环境空间大小', '环境是否整洁',
                        '分量', '味道', '菜品外观', '菜品推荐程度',
                        '本次消费感受', '再次消费意愿']

    # aspect_name_list= ['location_traffic_convenience',
    #                  'location_distance_from_business_district', 'location_easy_to_find',
    #                  'service_wait_time', 'service_waiters_attitude',
    #                  'service_parking_convenience', 'service_serving_speed', 'price_level',
    #                  'price_cost_effective', 'price_discount', 'environment_decoration',
    #                  'environment_noise', 'environment_space', 'environment_cleaness',
    #                  'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
    #                  'others_overall_experience', 'others_willing_to_consume_again']

    col_name_order = ['text'] + aspect_name_list
    DATA_DIR = "/Data_Center/美团评论ABSA/2018_zh_ABSA_dataset/"
    train_path = 'meituan_absa_train_data.csv'
    test_path = 'meituan_absa_dev_data.csv'
    data_reader = Preprocessor(DATA_DIR, train_path, test_path, col_name_order, aspect_name_list)
    print('Generating train and test sets...')
    train_dataset, test_dataset = data_reader.process_data()

    ABSA_tsv_DIR = '/bert_absa_data/'
    print(f'Saving to {ABSA_tsv_DIR} ...')
    task_name = 'QA_B'
    sentiment_label = {'-2': "未提及", '0': "中性", '1': "正面", '-1': "负面"}
    data_generator = ABSA_data_gen(task_name, sentiment_label, train_dataset, test_dataset, aspect_name_list)
    data_generator.save_to_tsv(ABSA_tsv_DIR)
    print('Success!')