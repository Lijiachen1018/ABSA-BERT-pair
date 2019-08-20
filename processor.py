# coding=utf-8

"""Processors for different tasks."""

class DataProcessor(object):

    def __init__(self,ABSA_tsv_DIR,task_name,sentiment_label):
        self.task_name = task_name
        self.ABSA_tsv_DIR = ABSA_tsv_DIR
        self.sentiment_label = sentiment_label
        self.train_path = f'{self.ABSA_tsv_DIR}train_{self.task_name}.tsv'
        self.test_path = f'{self.ABSA_tsv_DIR}test_{self.task_name}.tsv'

    def get_train_path(self):
        return self.train_path

    def get_test_path(self):
        return self.test_path

    def get_labels(self):
        if self.task_name[-1] == 'M':
            return  list(self.sentiment_label.values())
        else:
            return ['0','1']
