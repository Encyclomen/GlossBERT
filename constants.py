import os


project_root_dir = os.path.split(os.path.realpath(__file__))[0]

csv_relative_paths = {
        'train': 'Training_Corpora/SemCor/semcor_train_token_cls.csv',
        '2007': 'Evaluation_Datasets\\semeval2007\\semeval2007_test_token_cls.csv',
        '2013': 'Evaluation_Datasets/semeval2013/semeval2013_test_token_cls.csv',
        '2015': 'Evaluation_Datasets/semeval2015/semeval2015_test_token_cls.csv',
        '2': 'Evaluation_Datasets/senseval2/senseval2_test_token_cls.csv',
        '3': 'Evaluation_Datasets/senseval3/senseval3_test_token_cls.csv',
        'ALL': 'Evaluation_Datasets/ALL/ALL_test_token_cls.csv'
}

csv_paths = {
        'train': os.path.join(project_root_dir, csv_relative_paths['train']),
        'dev': os.path.join(project_root_dir, csv_relative_paths['2007']),
        '2007': os.path.join(project_root_dir, csv_relative_paths['2007']),
        '2013': os.path.join(project_root_dir, csv_relative_paths['2013']),
        '2015': os.path.join(project_root_dir, csv_relative_paths['2015']),
        '2': os.path.join(project_root_dir, csv_relative_paths['2']),
        '3': os.path.join(project_root_dir, csv_relative_paths['3']),
        'ALL': os.path.join(project_root_dir, csv_relative_paths['ALL']),
}

bert_dir = os.path.join(project_root_dir, 'bert-model')