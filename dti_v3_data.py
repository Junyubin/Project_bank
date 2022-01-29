from dti_v3_utils import *
from dti_v3_query import *
from datetime import date, timedelta
import pandas as pd
import numpy as np



class DataCreation():
    def __init__(self, mode, config_type, model_id, model_name):
        self.mode = mode
        self.config_type = config_type
        self.model_id = model_id
        self.model_name = model_name
        
        config = Params.get_config(self.config_type, self.model_id, self.model_name)
        
        self.today = date.today()
        self.normal_start_time = (self.today - timedelta(days = config['normal_days'])).strftime('%Y-%m-%d %H:%M:%S')
        self.attack_start_time = (self.today - timedelta(days = config['attack_days'])).strftime('%Y-%m-%d %H:%M:%S')
        self.pred_start_time = (self.today - timedelta(days = config['predict_time'])).strftime('%Y-%m-%d %H:%M:%S')
        self.end_time = self.today.strftime('%Y-%m-%d %H:%M:%S')
        self.data_limit = config['limit']
        self.interval = '1 minute'
        self.attack_array = ['BEACONING', 'CREDENTIAL', 'SQL_INJECTION', 'XSS']
        self.index_cols = config['index_cols'].split(', ')
        
        if self.mode == 'train':
            self.train_data_load()
        else:
            self.pred_data_load()
        
    
    def train_data_load(self):
        print('NORMAL DATA START DATETIME: ', self.normal_start_time)
        print('ATTACK DATA START DATETIME: ', self.attack_start_time)
        print('DATA END DATETIME: ', self.end_time)
        print('DATA LIMIT: ', self.data_limit)
        print('******* 정상 데이터 불러오기 *******')
        try:
            result, meta = execute_ch(normal_query(self.normal_start_time, self.end_time, self.data_limit, self.interval), with_column_types = True)
        except:
            print('ERROR: CHECK THE NORMAL DATA QUERY...')
            print(normal_query(self.normal_start_time, self.end_time, self.data_limit, self.interval))
            return                        
        if not result:
            print('ERROR: NORMAL DATA NOT FOUND. PLEASE CHECK YOUR DATETIME AND FILTER SETTINGS...')
            return
        feats = [m[0] for m in meta]
        normal_data = pd.DataFrame(result, columns = feats)
        print('NORMAL DATA PROPERTIES')
        print(normal_data.info())        
        
        print('******* 공격 데이터 불러오기 *******')        
        attack_data = pd.DataFrame()
        for attack in self.attack_array:
            sql = attack_query(attack, self.attack_start_time, self.end_time, self.data_limit, self.interval)
            try:
                result, meta = execute_ch(sql, with_column_types = True)
            except:
                print('ERROR: CHECK THE ATTACK DATA QUERY...')
                print(attack_query(attack, self.attack_start_time, self.end_time, self.data_limit, self.interval))
                return
            if not result:
                print('ERROR: ATTACK DATA {attack} NOT FOUND. PLEASE CHECK YOUR DATETIME AND FILTER SETTINGS...'.format(attack=attack))
                continue
            feats = [m[0] for m in meta]
            sql_data = pd.DataFrame(result, columns = feats)
            attack_data = pd.concat([attack_data, sql_data])
        print('ATTACK DATA PROPERTIES')
        print(attack_data.info())
        
        self.total_data = pd.concat([normal_data, attack_data]).convert_dtypes()
        
        ########## [datetime timezone  ->  datetime] ##########
        datetime_tz_col = list(self.total_data.select_dtypes('datetimetz'))
        for col in datetime_tz_col:
            self.total_data[col] = pd.to_datetime(self.total_data[col]).dt.tz_localize(None)
            
        train_data = self.__type_check(self.total_data)
        self.__fill_null()
        return train_data
            
    def pred_data_load(self):
        print('PREDICT DATA START DATETIME: ', self.pred_start_time)
        print('DATA END DATETIME: ', self.end_time)
        print('DATA LIMIT: ', self.data_limit)
        
        print('******* 예측 데이터 불러오기 *******')
        try:
            result, meta = execute_ch(predict_query(self.pred_start_time, self.end_time, self.data_limit, self.interval), with_column_types = True)
        except:
            print('ERROR: CHECK THE NORMAL DATA QUERY...')
            print(predict_query(self.pred_start_time, self.end_time, self.data_limit, self.interval))
            return
        
        if not result:
            print('ERROR: NORMAL DATA NOT FOUND. PLEASE CHECK YOUR DATETIME AND FILTER SETTINGS...')
            return
            
        feats = [m[0] for m in meta]
        self.pred_data = pd.DataFrame(result, columns = feats)
        print('PREDICT DATA PROPERTIES')
        print(self.pred_data.info())
        
        ########## [datetime timezone  ->  datetime] ##########
        datetime_tz_col = list(self.pred_data.select_dtypes('datetimetz'))
        for col in datetime_tz_col:
            self.pred_data[col] = pd.to_datetime(self.pred_data[col]).dt.tz_localize(None)
            
        self.pred_data = self.pred_data.convert_dtypes()
        
        self.__type_check(self.pred_data)
        self.__fill_null()
    
    def __type_check(self, df, cat_threshold=10):
        df.reset_index(drop = True, inplace = True)
        if self.mode == 'train':
            ########## [label 데이터 분리] ##########
            X_data = df.drop('label', axis = 1)
            self.label_data = df[['label']]
        else:
            X_data = df.copy()
        
        ########## [datetime 데이터 분리] ##########
        self.index_data = X_data[self.index_cols]
        
        ########## [유니크값이 2개 이상 100개 이하인 데이터 -> 카테고리 데이터] ##########
        X_data.drop(list(self.index_data), axis = 1, inplace=True)
        for i in list(X_data) :
            if X_data[i].nunique() >= 2 and X_data[i].nunique() <= cat_threshold:
                X_data[i] = X_data[i].astype('category')
                
        ########## [num, str, category 데이터 분리] ##########
        self.num_data = X_data.select_dtypes('number')
        self.str_data = X_data.select_dtypes('string')
        self.cat_data = X_data.select_dtypes('category')

        print('INDEXES : ',list(self.index_data))
        print('NUMBER : ',list(self.num_data))
        print('STRING : ',list(self.str_data))
        print('CATEGORY : ',list(self.cat_data))
        
        return X_data

    def __fill_null(self):
        ########## [데이터 유형별 null 값 채우기] ##########
#         self.dt_data = self.dt_data.fillna('')
        self.num_data = self.num_data.fillna('-1')
        self.str_data = self.str_data.fillna('-')
        for col in self.cat_data:
            self.cat_data[col] = self.cat_data[col].cat.add_categories("empty").fillna("empty")