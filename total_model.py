import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import time
import operator
import os, pathlib, sys
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.backend import set_session
from graphviz import Source
pwd = os.getcwd()
sys.path.insert(0, pwd+'/../')
from sql_query import *
# from ssql_query_v2_backup_0216 import *
from total_base import *
from optparse import OptionParser
from utls_v2 import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
set_session(sess)

class Train250(BaseComponent):
    def init(self):
        pass 

    async def run(self, param):
        print(param['logtime_s'])
        print(param['logtime_e'])

        start = datetime.datetime.now()
        week = datetime.datetime.today().weekday()

        version_day = datetime.date.today()
        self.save_version = str(version_day.year) + '_' + str(version_day.month).zfill(2) + '_' + str(version_day.day).zfill(2)
        # self.save_version = '00_sophia_test'zzzzzzzzzzzzzzzzzzzzzzzzzzzz
        print(self.save_version)

        att_dict = {'scanning':'HTTP_SCANNING',
                    'injection':'SQL_INJECTION',
                    'xss':'XSS',
                    'beaconing':'BEACONING',
                    'credential':'CREDENTIAL',
                    'normal':'NORMAL',
                    'anomaly':'ANOMALY'}
        try:
            if not self.load_data:
                print('NORMAL DATA LOADING')
                print(param['logtime_s'], param['logtime_e'])
                result, meta = execute_ch(normal_query(param['logtime_s'], param['logtime_e'],self.data_limit), with_column_types = True)
                feats = [m[0] for m in meta]
                normal_data = pd.DataFrame(result, columns = feats)
                print('DONE')
                print(normal_data['label'].value_counts())

                print('ATTACK DATA MAPPING')
                filtering_dict = {'beaconing':'', 
                                  'injection':'', 
                                  'scanning':'', 
                                  'credential':'', 
                                  'xss':''}
                attack_mapping = pd.read_csv('attack_files/notice_mapping.csv')
                for k in range(len(attack_mapping)):
                    if attack_mapping.loc[k, 'mapping'] in filtering_dict.keys():
                        filtering_dict[attack_mapping.loc[k, 'mapping']] += 'or note==\'' + attack_mapping.loc[k, 'note'] + '\' '
                for key in filtering_dict.keys():
                    if len(filtering_dict[key]) != 0:
                        filtering_dict[key] = filtering_dict[key][3:]
                    else:
                        filtering_dict[key] = '1!=1'
                filtering_dict2 = {'beaconing':'', 
                                  'injection':'', 
                                  'scanning':'', 
                                  'credential':'', 
                                  'xss':''}
                blockd_mapping = pd.read_csv('attack_files/blockd_mapping_2.csv')
                for k in range(len(blockd_mapping)):
                    if blockd_mapping.loc[k, 'mapping'] in filtering_dict2.keys():
                        filtering_dict2[blockd_mapping.loc[k, 'mapping']] += 'or attack_name==\''+blockd_mapping.loc[k, 'attack_name'].replace("'", "\\'")+'\' '
                for key in filtering_dict2.keys():
                    if len(filtering_dict2[key]) != 0:
                        filtering_dict2[key] = filtering_dict2[key][3:]
                    else:
                        filtering_dict2[key] = '1!=1'
                print('DONE')

                print('ATTACK_DATA_LOADING')
                manager = Manager()
                select_result = manager.list()

                scanning_rule = execute_ch("""
                                            select rule_body
                                            from dti.mysql_rule
                                            where rule_name == 'scanning' and id = (select max(id) from dti.mysql_rule where rule_name == 'scanning')
                                            """, with_column_types = False)

                credential_rule = execute_ch("""
                                            select rule_body
                                            from dti.mysql_rule
                                            where rule_name == 'credential' and id = (select max(id) from dti.mysql_rule where rule_name == 'credential')
                                            """, with_column_types = False)

                beaconing_rule = execute_ch("""
                                            select rule_body
                                            from dti.mysql_rule
                                            where rule_name == 'beaconing' and id = (select max(id) from dti.mysql_rule where rule_name == 'beaconing')
                                            """, with_column_types = False)

                scanning_rule = scanning_rule[0][0]
                credential_rule = credential_rule[0][0]
                beaconing_rule = beaconing_rule[0][0]

                sql_array = attack_sql_query(param, filtering_dict, filtering_dict2, self.data_limit, self.attack_days, 
                                            scanning_rule, credential_rule, beaconing_rule)
        #######################################################################################################################
                await asyncio.gather(*(self.loop.run_in_executor(None, execute_client, select_result, sql) for sql in sql_array))
                attack_data = pd.DataFrame(list(select_result), columns = feats)
                self.logger.info('[Train({})] len(select_result): {}'.format(self.model_name, len(select_result)))
                print('DONE')

                print('NORMAL DATA MAPPING WITH ATTACK DATA')
                total_data = pd.concat([normal_data, attack_data])
                total_data.sort_index(inplace = True)
                print('DONE')

                total_data.rename(columns = {'query_' : 'http_query'}, inplace = True)
                total_data.rename(columns = {'agent_' : 'http_agent'}, inplace = True)
                total_data.rename(columns = {'host_' : 'http_host'}, inplace = True)

                print(total_data['label'].value_counts())

                with open('data/total_data_'+self.save_version+'.pickle', 'wb') as f:
                    pickle.dump(total_data, f)
                print('----------------------------------- TOTAL DATA SAVE DONE ------------------------------')
            else:
                with open('data/total_data_'+self.save_version+'.pickle', 'rb') as f:
                    total_data = pickle.load(f)
                print('---------------------------------- TOTAL DATA LOADED DONE -----------------------------')
                
            labels = list(total_data['label'].unique())
            train_normal_data = total_data[total_data['label'] == 'normal']
            new_train = train_normal_data.sample(n=50000, replace=True)
            
            for key in labels :
                if key != 'normal':
                    train_attack_data = total_data[total_data['label'] == key]
                    train_attack_data = train_attack_data.sample(n=10000, replace = True)
                    new_train = pd.concat([new_train, train_attack_data])

            total_data = new_train.sample(frac = 1.0)

            train_len = int(len(total_data) * 0.7)
            train_data = total_data.iloc[:train_len]
            test_data = total_data.iloc[train_len:]

            # with open("validation_data/train_data"+self.save_version+".pickle", 'wb') as f:
            #     pickle.dump(train_data, f)
            # with open("validation_data/test_data"+self.save_version+".pickle", 'wb') as f:
            #     pickle.dump(test_data, f)

            train_x, train_y = train_data.iloc[:,:-1], train_data.iloc[:,-1]
            test_x, test_y = test_data.iloc[:,:-1], test_data.iloc[:,-1]
            self.logger.info(str(train_x.shape)+'_'+str(test_x.shape)+'_'+str(train_y.shape)+'_'+str(test_y.shape))
            
            del_list = ['lgtime','end','src_ip','dst_ip']
            test_key = test_x[del_list]
            train_key = train_x[del_list]
            train_x.drop(del_list, axis = 1, inplace = True)
            test_x.drop(del_list, axis = 1, inplace = True)

            train_x.reset_index(drop = True, inplace = True)
            train_y.reset_index(drop = True, inplace = True)
            test_x.reset_index(drop = True, inplace = True)
            test_y.reset_index(drop = True, inplace = True)
            test_key.reset_index(drop = True, inplace = True)
            train_key.reset_index(drop = True, inplace = True)
            char_data_list = ['http_host','http_agent','http_query']
            tfidf_model_fit_char(train_x, char_data_list, self.n_grams, 2000, self.save_version)
            train_x, feature_names, str_data, _ = tfidf_model_trans_char(train_x, char_data_list, 150000, self.save_version)
            test_x, _, _, _ = tfidf_model_trans_char(test_x, char_data_list, 150000, self.save_version)
            scl_model_fit(train_x, self.save_version)
            train_x = scl_model_trans(train_x, self.save_version)
            test_x = scl_model_trans(test_x, self.save_version)
        #########################################################################################################################
            dt_model = dt_model_fit(train_x, train_y, self.dt_depth, self.save_version)
            labels = list(dt_model.classes_)

            # with open('validation_data/labels_'+self.save_version+'.pickle', 'wb') as fw:
            #     pickle.dump(labels, fw)
            # with open('validation_data/features_'+self.save_version+'.pickle', 'wb') as fw:
            #     pickle.dump(feature_names, fw)

            attack_nm, features, value, node_num, node_to_left, node_to_right, level = get_dtree(self.save_version, feature_names, labels, train_x)
            temp = pd.DataFrame({'value':value, 'features':features, 'node_num':node_num, 'node_from':np.NaN, 
                                'node_to_left':node_to_left, 'node_to_right':node_to_right, 'level':level, 
                                'model_id':self.model_id, 'logtime':datetime.datetime.strptime(param['logtime_s'], '%Y-%m-%d %H:%M:%S'), 'attack_nm':attack_nm})

            
            # temp['node_from'][temp.index[temp['node_num'].isin(temp['node_to_left'])].tolist()] = temp.index[temp['node_to_left'].isin(temp['node_num'])].tolist()
            # temp['node_from'][temp.index[temp['node_num'].isin(temp['node_to_right'])].tolist()] = temp.index[temp['node_to_right'].isin(temp['node_num'])].tolist()
            temp['logtime'] = pd.to_datetime(temp['logtime'], format = '%Y-%m-%d %H:%M:%S')
            temp['node_from'] = temp['node_num'].apply(lambda x : get_parent(x, temp))

            dtree_df = temp[['model_id','logtime','attack_nm','value','features','node_num','node_from','node_to_left','node_to_right','level']].copy()
            dtree_df.fillna('-', inplace = True)
            temp = dtree_df.select_dtypes(include = 'int')
            dtree_df[list(temp)] = temp.astype('int')
            temp = dtree_df.select_dtypes(include = 'object')
            dtree_df[list(temp)] = temp.astype('str')

            dtree_df.loc[dtree_df['attack_nm'] == 'scanning','attack_nm'] = 'http scanning'
            dtree_df.loc[dtree_df['attack_nm'] == 'injection','attack_nm'] = 'sql injection'
            dtree_df['model_ver'] = self.save_version

            insert_result = await execute_async_ch('insert into {} values'.format(self.dt_table), dtree_df.values.tolist())
            self.logger.info('[TRAIN (SUPERVISED) - DT INSERT] result: {}, time: {}'.format(dtree_df.values.shape, datetime.datetime.now() - start))
        ##########################################################################################################################
            dt_importance_df = pd.DataFrame()
            dt_importance_df['model_id'] = [int(self.model_id) for i in range(len(feature_names))]
            dt_importance_df['logtime'] = [datetime.datetime.strptime(param['logtime_s'], '%Y-%m-%d %H:%M:%S') for i in range(len(feature_names))]
            dt_importance_df['attack_nm'] = ['total' for i in range(len(feature_names))]
            dt_importance_df['weekday'] = [str(week) for i in range(len(feature_names))]
            dt_importance_df['dayhour'] = [str(param['logtime_s'])[11:13] for i in range(len(feature_names))]
            dt_importance_df['feature'] = feature_names
            dt_importance_df['value'] = dt_model.feature_importances_
            insert_result = await execute_async_ch('insert into {} values'.format(self.dt_importance), dt_importance_df.values.tolist())
            self.logger.info('[TRAIN (SUPERVISED) - DT INSERT] result: {}, time: {}'.format(dt_importance_df.values.shape, datetime.datetime.now() - start))
            
            operator_dict = {'l':operator.le, 'r':operator.gt}
            print("*"*30+" DECISION TREE TRAIN "+"*"*30)
            dt_train_pred_y = dt_result_print(train_x, train_y, dt_model)
            print("*"*30+" DECISION TREE TEST "+"*"*30)
            dt_test_pred_y = dt_result_print(test_x, test_y, dt_model)

            train_x_normal = train_x[train_y == 'normal']
            
            EPOCH = self.eps
            BATCH_SIZE = self.bs
            history_df = pd.DataFrame()
            
            for attack in list(labels) :
                metrics_df = pd.DataFrame()
                metrics_df['accuracy'] = 0
                print(attack)
                if attack == 'normal':
                    metrics = ae_model_fit(train_x_normal, EPOCH, BATCH_SIZE, model_name = attack, save_version = self.save_version)
                else:
                    train_x_attack = train_x[train_y == attack]
                    temp_train_x = np.concatenate([train_x_attack, train_x_normal])
                    temp_train_y = np.concatenate([np.ones((len(train_x_attack))), np.zeros((len(train_x_normal)))])
                    perm = np.random.permutation(len(temp_train_x))
                    temp_train_x = temp_train_x[perm]
                    temp_train_y = temp_train_y[perm]
                    metrics = dnn_model_fit(temp_train_x, temp_train_y, EPOCH, BATCH_SIZE, model_name = attack, save_version=self.save_version)
                    metrics_df['accuracy'] = metrics.history['acc']
                metrics_df['loss'] = metrics.history['loss']
                metrics_df['attack_name'] = att_dict[attack]
                metrics_df['epochs'] = np.arange(len(metrics.history['loss'])) + 1
                metrics_df = metrics_df.fillna(0)
                metrics_df['model_id'] = self.model_id
                metrics_df['logtime'] = start

                metrics_df = metrics_df[['model_id','attack_name','logtime','epochs','loss','accuracy']]
                insert_result = await execute_async_ch('insert into {} values'.format(self.ai_metrics), metrics_df.values.tolist())
                self.logger.info('[TRAIN (SUPERVISED - AI METRICS  INSERT] result: {}, time: {}'.format(metrics_df.values.shape, datetime.datetime.now() - start)) 
                history_df = pd.concat([history_df, metrics_df[-1:]]) 
            self.logger.info('TRAINING FINISHED')
            print("*"*20+' DNN TRAIN DATA '+"*"*20)
        ############################################# box plot anomaly detection ##############################################
            for i in labels:
                test_key[i] = np.nan
            
            for i in labels:
                train_key[i] = np.nan

            def dnn_result(x, dt_result, key):
                for model_name in labels:
                    temp_x = pd.DataFrame(x, columns = feature_names)
                    temp_x['label'] = dt_result
                    temp_x = temp_x[temp_x['label'] == model_name]
                    temp_x.drop('label', axis = 1, inplace = True)
                    data_index = temp_x.index

                    if model_name == 'normal':
                        key.loc[data_index, model_name] = ae_model_predict(temp_x.values, model_name = model_name, save_version = self.save_version)
                    else:
                        key.loc[data_index, model_name] = dnn_model_predict(temp_x.values, model_name = model_name, save_version = self.save_version)

                final_result = key.fillna(0).copy()

                return final_result

            train_dnn_pred_y = dnn_result(train_x, dt_train_pred_y, train_key)
            test_dnn_pred_y = dnn_result(test_x, dt_test_pred_y, test_key)
            pred_y = train_dnn_pred_y.copy()

            def dnn_perfomance(pred_y, how = ['train', 'test']):
                print(how)
                def att_name(x):
                    key = model_label[x]
                    if key in att_dict.keys():
                        return att_dict[key]
                

                if how == 'train':
                    train_threshold = mad_score(pred_y['normal'], 0, 0, 'box')
                    print("*"*30)
                    print(train_threshold)
                    print("*"*30)

                    with open('data/train_threshold_'+self.save_version+'.pickle', 'wb') as fw:
                        pickle.dump(train_threshold, fw)
                
                elif how == 'test':
                    with open('data/train_threshold_'+self.save_version+'.pickle', 'rb') as f:
                        train_threshold = pickle.load(f)
                
                pred_y['anomaly'] = np.where(pred_y['normal'] > train_threshold, 1, 0)
                filter_data = pred_y.copy()

#                 for i in ['normal', 'anomaly', 'beaconing', 'injection', 'credential', 'xss', 'scanning']:
                for i in ['anomaly', 'beaconing', 'injection', 'credential', 'xss', 'scanning']:
                    if i != 'normal':
                        filter_data = filter_data[filter_data[i] <= 0.5]
                    else: pass

                pred_y.loc[filter_data.index, 'normal'] = 1
                model_label = list(labels)
                model_label.append('anomaly')
                matrix = pred_y[model_label].values > 0.5
                probs = np.max(matrix, 1)
                final_labels = np.argmax(matrix, 1)
                pred_y['result'] = final_labels
                pred_y['probs'] = probs
                pred_y.drop(model_label, 1, inplace = True)
                pred_y['result'] = pred_y['result'].astype('int')
                pred_y['model_id'] = self.model_id
                pred_y['att_name'] = pred_y['result'].apply(lambda x: att_name(x))
                pred_y['result'] = pred_y['att_name'].apply(lambda x: def_att(x)) + 250 * 1000

                return pred_y

        ################################################## DNN SCORE ########################################################
            final_result = dnn_perfomance(train_dnn_pred_y, how = 'train')

            # with open('validation_data/final_result_train'+self.save_version+'.pickle', 'wb') as f:
            #     pickle.dump(fianl_result, f)

            train_y = pd.get_dummies(train_y)
            train_y.rename(columns = att_dict, inplace = True)
            pred_y = pd.get_dummies(final_result['att_name'])

            history_df['train_cnt'] = 0
            history_df['test_cnt'] = 0
            history_df['accuracy'] = 0
            history_df['recall'] = 0
            history_df['precision'] = 0
            history_df['f1_score'] = 0
            history_df['train_time'] = datetime.datetime.now()

            print("*"*10+' TRAIN DNN SCORE '+"*"*10)
            for i in list(train_y):
                if i not in list(pred_y):
                    pass
                elif i == 'ANOMALY':
                    pass
                else:
                    print('*************************************')
                    print('<', i, '>')
                    acc = accuracy_score(train_y[i], pred_y[i])
                    recall = recall_score(train_y[i], pred_y[i])
                    precision = precision_score(train_y[i], pred_y[i])
                    f1_sc = f1_score(train_y[i], pred_y[i])
                    print('accuracy', acc)
                    print('recall', recall)
                    print('precision', precision)
                    print('f1_score', f1_sc)
                    history_df.loc[history_df['attack_name']==i, 'accuracy'] = acc
                    history_df.loc[history_df['attack_name']==i, 'recall'] = recall
                    history_df.loc[history_df['attack_name']==i, 'precision'] = precision
                    history_df.loc[history_df['attack_name']==i, 'f1_score'] = f1_sc
                    history_df.loc[history_df['attack_name']==i, 'train_cnt'] = len(pred_y[pred_y[i]==1])
                    history_df.loc[history_df['attack_name']==i, 'test_cnt'] = len(train_y[train_y[i]==1])
                    print(confusion_matrix(train_y[i], pred_y[i], labels = [0,1]))
        
        ############################################ TEST DNN SCORE #####################################################
            final_result_test = dnn_perfomance(test_dnn_pred_y, how = 'test')

            # with open('validation_data/final_result_test'+self.save_version+'.pickle', 'wb') as f:
            #     pickle.dump(fianl_result_test, f)

            test_y = pd.get_dummies(test_y)
            test_y.rename(columns = att_dict, inplace = True)
            pred_y = pd.get_dummies(final_result_test['att_name'])

            print("*"*10+' TEST DNN SCORE '+"*"*10)
            for i in list(test_y):
                if i not in list(pred_y):
                    pass
                elif i == 'ANOMALY':
                    pass
                else:
                    print('*************************************')
                    print('<', i, '>')
                    acc = accuracy_score(test_y[i], pred_y[i])
                    recall = recall_score(test_y[i], pred_y[i])
                    precision = precision_score(test_y[i], pred_y[i])
                    f1_sc = f1_score(test_y[i], pred_y[i])
                    print('accuracy', acc)
                    print('recall', recall)
                    print('precision', precision)
                    print('f1_score', f1_sc)
                    print(confusion_matrix(test_y[i], pred_y[i], labels = [0,1]))

        ################################################################################################################
            try:
                history_df = history_df[['model_id', 'attack_name', 'train_cnt', 'test_cnt', 'train_time', 'logtime', 'epochs', 'accuracy', 'precision', 'recall', 'f1_score']]
                history_insert = await execute_async_ch('insert into {} values'.format(self.ai_history), history_df.values.tolist())
                self.logger.info('[TRAIN (SUPERVISED) - AI HISTORY INSERT] result: {}, time : {}'.format(history_df.values.shape, datetime.datetime.now()- start))
            except Exception as err:
                self.logger.error(err)

            with open('data/labels_' + self.save_version + '.pickle', 'wb') as fw:
                pickle.dump(labels, fw)

            sess.close()
            return 'OK'

        except Exception as err:
            self.logger.error(err)
            self.logger.error(traceback.print_exc())
            sess.close()
            return None


class Prediction250(BaseComponent):
    def init(self):
        self.prob_threshold = self.model_config['prob_threshold']
        self.proc_count = self.model_config['proc_count']
        self.vector_size = self.model_config['vector_size']
        self.n_grams = (self.model_config['n_grams_lower'], self.model_config['n_grams_upper'])
        self.tfidf_http_model = self.model_name + '_tfidf_ast'
        self.query_bs = self.model_config['query_bs']
        self.cos_bs = self.model_config['cos_bs']
        self.sample_size = self.model_config['sample_size']
        self.version = self.model_name + '_ast'
        self.lr = self.model_config['lr']
        self.latent =self.model_config['latent']
        self.eps = self.model_config['eps']
        self.bs = self.model_config['bs']
        self.cpk = self.eps
        self.n_filters = self.model_config['n_filters']
        self.k_size = self.model_config['k_size']
        self.y_shape = np.zeros((self.model_config['y_shape_lower'], self.model_config['y_shape_upper'])).shape
        self.fb_threshold = self.model_config['fb_threshold']
        self.dt_prep = self.model_config['dt_prep']

    async def run(self, param):
        start = datetime.datetime.now()
        self.logger.info(start)
        week = datetime.datetime.today().weekday()
        
        version_day = datetime.datetime.strptime(param['logtime_s'], '%Y-%m-%d %H:%M:%S') - datetime.timedelta(days = 1)
        self.save_version = str(version_day.year) + '_' + str(version_day.month).zfill(2) + '_' + str(version_day.day).zfill(2)
        print(self.version)
        labels = None
        while labels == None:
            try:
                with open('data/labels_'+self.save_version + '.pickle', 'rb') as f:
                    labels = pickle.load(f)
            except:
                version_day = version_day - datetime.timedelta(days = 1)
                self.svae_version = str(version_day.year) + '_' + str(version_day.month).zfill(2) + '_' + str(version_day.day).zfill(2)

        try:
            print('*'*30)
            print(self.save_version)
            print('*'*30)
            print('*'*30 + self.save_version + ' VERSION PREDICTION START' + '*'*30)
            print('PREDICTION DATA LOADING')
            print(param['logtime_s'], param['logtime_e'])

            exception_result, meta = execute_ch(""" select * from dti.mysql_exception """, with_column_types = True)
            feats = [m[0] for m in meta]
            exception_df = pd.DataFrame(exception_result, columns = feats)

            http_query_ex = exception_df[exception_df['selectcolumn'] == 'http_query']
            http_query_ex = '|'.join(http_query_ex['keyword'].values)
            
            http_agent_ex = exception_df[exception_df['selectcolumn'] == 'http_agent']
            http_agent_ex = '|'.join(http_agent_ex['keyword'].values)
            
            http_host_ex = exception_df[exception_df['selectcolumn'] == 'http_host']
            http_host_ex = '|'.join(http_host_ex['keyword'].values)
            
            http_path_ex = exception_df[exception_df['selectcolumn'] == 'http_path']
            http_path_ex = '|'.join(http_path_ex['keyword'].values)
            
            result, meta = execute_ch(predict_query(param['logtime_s'], param['logtime_e'], http_agent_ex, http_path_ex, http_query_ex, http_host_ex)
                                    , with_column_types=True)

            feats = [m[0] for m in meta]
            predict_data = pd.DataFrame(result, columns = feats)
            print('DONE')

            predict_data.rename(columns = {'query_':'http_query'}, inplace = True)
            predict_data.rename(columns = {'agent_':'http_agent'}, inplace = True)
            predict_data.rename(columns = {'host_':'http_host'}, inplace = True)


            self.logger.info(str(predict_data.info()))
            del_list = ['lgtime','end','src_ip','dst_ip']
            predict_key = predict_data[del_list]
            predict_data.drop(del_list, axis = 1, inplace = True)
            char_data_list = ['http_host','http_agent','http_query']
            print('******************* START TFIDF ******************')
            predict_data, feature_names, str_data, values_dict = tfidf_model_trans_char(predict_data, char_data_list, 300000, self.save_version)
            print('******************* START DATA SCALING ******************')
            predict_data = scl_model_trans(predict_data, self.save_version)
            print('******************* START DECISION TREE ******************')
            with open('model/dt_model_' + self.save_version + '.pickle', 'rb') as f:
                dt_model = pickle.load(f)
            
            dt_pred_y = dt_model.predict(predict_data)
            
            for model_name in labels:
                temp_x = pd.DataFrame(predict_data, columns = feature_names)
                temp_x['label'] = dt_pred_y

                if model_name == labels[0]:
                    print(temp_x['label'].value_counts())

                temp_x = temp_x[temp_x['label'] == model_name]
                print(model_name)

                temp_x.drop('label', axis = 1, inplace = True)
                data_index = temp_x.index

                if model_name == 'normal' :
                    predict_key.loc[data_index, model_name] = ae_model_predict(temp_x.values, model_name = model_name, save_version = self.save_version)
                else:
                    predict_key.loc[data_index, model_name] = dnn_model_predict(temp_x.values, model_name = model_name, save_version = self.save_version)

            final_result = predict_key.fillna(0).copy()

            with open('data/train_threshold_' + self.save_version + '.pickle', 'rb') as f:
                train_threshold = pickle.load(f)

            print('*' * 30)
            print(train_threshold)
            print('*' * 30)
        #####################################################################################################################
            att_dict = {'scanning' : 'HTTP_SCANNING',
                        'injection' : 'SQL_INJECTION',
                        'xss': 'XSS',
                        'beaconing' : 'BEACONING',
                        'credential' : 'CREDENTIAL',
                        'normal':'NORMAL',
                        'anomaly':'ANOMALY'}
            
            def att_name(x):
                key = model_label[x]
                if key in att_dict.keys():
                    return att_dict[key]

            final_result['anomaly_score'] = final_result['normal']
            final_result['anomaly'] = 0
            filter_data = final_result.copy()

            for i in ['anomaly', 'beaconing', 'injection', 'credential', 'xss', 'scanning']:
                if i != 'normal':
                    filter_data = filter_data[filter_data[i] <= 0.5]
                else: pass
            
            final_result.loc[filter_data.index, 'normal'] = 1
            model_label = list(labels)
            model_label.append('anomaly')
            matrix = final_result[model_label].values > 0.5
            probs = np.max(matrix, 1)
            final_labels = np.argmax(matrix, 1)
            final_result['result'] = final_labels
            final_result['probs'] = probs
            final_result.drop(model_label, 1, inplace = True)
            final_result['result'] = final_result['result'].astype('int')
            final_result['model_id'] = self.model_id
            final_result['att_name'] = final_result['result'].apply(lambda x: att_name(x))
            final_result['result'] = final_result['att_name'].apply(lambda x: def_att(x)) + 250 * 1000
            final_result['rule'] = '-'
            final_result['feedback'] = '-'
            final_result['packet'] = 0
            final_result['if_label'] = 0
            final_result['src_ip_country_code'] = '-'
            final_result['dst_ip_country_code'] = '-'
            final_result['direction_inout'] = '-'
            final_result['direction_inout_bin'] = 0
            final_result['related_country'] = '-'
            final_result['ast_ip'] = 0
            final_result['related_ip'] = 0
            final_result['if_score'] = 0
            final_result['lof_label'] = 0
            final_result['lof_score'] = 0
            final_result['ai_label'] = 0
            final_result['ai_score'] = 0
            final_result['feedback'] = self.save_version
            final_result.loc[final_result['att_name'] == 'NORMAL', 'probs'] = train_threshold
            # fianl_result.loc[final_result['att_name'] == 'ANOMALY', 'probs'] = train_threshold
            final_result.loc[final_result['anomaly_score'] > train_threshold, 'att_name'] = 'ANOMALY'
        ########################################################################################################################
            insert_list = ['model_id','lgtime','end','rule','src_ip','dst_ip','src_ip_country_code','dst_ip_country_code','ast_ip','related_ip','direction_inout','direction_inout_bin','related_country','att_name','result','probs','anomaly_score','feedback','packet','if_label','if_score','lof_label','lof_score','ai_label','ai_score']

            final_result[insert_list].drop_duplicates(inplace = True)

            print("************************** INSERT RESULT TABLE START **********************")
            print(len(final_result))
            print(final_result['att_name'].value_counts())
            temp_normal = final_result[final_result['att_name'].isin(['NORMAL']) == True]
            final_result = final_result[final_result['att_name'].isin(['NORMAL']) == False]
            final_result = pd.concat([final_result, temp_normal.sample(frac = 0.01)])

            insert_result = await execute_async_ch('insert into {} values'.format(param['result_table']), final_result[insert_list].values.tolist())
            self.logger.info('[PREDICTION({}) - INSERT] result : {}, time : {}'.format(self.model_name, final_result.values.shape, datetime.datetime.now() - start))
            print("************************** INSERT RESULT TABLE FINISH ***********************")
        #######################################################################################################################
            predict_data = pd.DataFrame(predict_data, columns = feature_names)

            col_temp = pd.DataFrame()
            def tolist(x):
                return list(x)
            for i in values_dict.keys():
                col_temp[i+'_value'] = predict_data[values_dict[i]].apply(lambda x : tolist(x), axis = 1)
                col_temp[i] = [values_dict[i] for n in range(len(col_temp))]

            
            temp = pd.merge(final_result, col_temp, left_index = True, right_index = True)
            temp = pd.merge(temp, predict_data[['dst_port_cnt', 'info_st', 'succ_st', 'redir_st', 'cler_st', 'serer_st', 'oth_st']], left_index = True, right_index = True)

            temp['model_id'] = self.model_id
            temp['model_ver'] = self.model_version

            temp = temp[['model_id', 'lgtime', 'end', 'src_ip', 'dst_ip', 'dst_port_cnt', 
                        'info_st', 'succ_st', 'redir_st', 'cler_st','serer_st', 'oth_st',
                        'http_host', 'http_host_value', 'http_agent', 'http_agent_value', 'http_query', 'http_query_value',
                        'model_ver', 'att_name']]
            temp = temp[temp['att_name'].isin(['NORMAL']) == False]

            print(temp.info())
            print('insert_start')
            print(len(temp))
            print(temp['att_name'].value_counts())

            batch_size = 200000
            start_time = 0

            for i in range(int(len(temp)/batch_size)+1):
                if start_time == len(temp):
                    break
                else:
                    end_time = start_time + batch_size
                    if end_time > len(temp):
                        end_time = len(temp)

                print(start_time, end_time)
                insert_prep_result = await execute_async_ch('insert into {} values'.format(self.dt_prep), temp.iloc[start_time:end_time].values.tolist())
                start_time = end_time
        #####################################################################################################################
            return 'OK'
        except Exception as err:
            self.logger.error(err)
            self.logger.error(traceback.print_exc())
            return None



def main(model_id, train, prediction):
    model_name = isExistModel(model_id)
    if model_name == None:
        sys.exit(1)
    else:
        # nc=NATS()
        loop = asyncio.get_event_loop()

        if train:
            if model_id == 250:
                loop.run_until_complete(Train250(loop, model_id, model_name).start_train())
                # loop.run_until_complete(Train250(loop, model_id, model_name).test_train())
            else:
                print('[TRAIN({})] model_i is invalid'.format(model_id))
                sys.exit()

        if prediction:
            if model_id == 250:
                loop.run_until_complete(Prediction250(loop, model_id, model_name).start_pred())
                # loop.run_until_complete(Prediction250(loop, model_id, model_name).test_pred())
            else:
                print('[PREDICTION({})] model_id is invalid'.format(model_id))
                sys.exit()
        try:
            loop.run_forever()
        finally:
            loop.close()

if __name__ == '__main__':
    if lend(sys.argv) <= 1:
        print('python3 total_model.py -h or python3 total_model.py --help')
        sys.exit()

    parser = OptionParser(usage='Usage: python3 total_model.py -t -m [model_id] or python3 total_model.py -p -m [model_id]')
    parser.add_option('-t', action = 'store_true', dest = 'isTrain', default=False, help = 'train')
    parser.add_option('-p', action = 'store_true', dest = 'isPred', default=False, help = 'train')
    parser.add_option('-m', type = int, dest = 'MODEL_ID', help = 'model_id')
    options, args = parser.parse_args()

    if options.isTrain == options.isPred:
        print('you have to choose train or prediction')
        sys.exit()
    if options.MODEL_ID is None:
        print('you have to input model id')
        sys.exit()

    main(options.MODEL_ID, options.isTrain, options.isPred)