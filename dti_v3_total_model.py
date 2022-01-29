from dti_v3_data import *
from dti_v3_model import *
from dti_v3_prep import *
from dti_v3_utils import *
from dti_v3_xai import *
import datetime
import matplotlib.pyplot as plt
from optparse import OptionParser
import cv2


class Train1(BaseComponent):
    """
    학습 모듈
    """
    def init(self):
        pass
    
    async def run(self, param):
        try:
            
            model_config = self.model_config
            model_id = model_config['common']['model_id']
            config_name = model_config['common']['config_name']
            
            """
            모델 버전 세팅
            """
            start = datetime.datetime.now().replace(microsecond=0) + timedelta(hours=9)
            train_version = start.strftime("%Y_%m_%d")
            if not os.path.exists(pwd + '/' + model_config['common']['model_path'] + '/' + train_version):
                os.makedirs(pwd + '/' + model_config['common']['model_path'] + '/' + train_version)                   
            self.logger.info('MODEL VERSION : {}'.format(train_version))

            
            """
            데이터 로드
            """
            # INSERT mode / config_type / model_id / model_name
            raw_data = DataCreation('train', 'COMMON', model_config['common']['model_id'], model_config['common']['config_name'])
            
            over_index = []

            for i in np.unique(raw_data.label_data.values):
                over_index.append(raw_data.label_data[raw_data.label_data['label'] == i].sample(n = raw_data.label_data.value_counts().max(), replace = True).index.values.tolist())

            self.logger.info(raw_data.label_data.value_counts())
            raw_data.str_data = raw_data.str_data.iloc[sum(over_index, [])]
            raw_data.num_data = raw_data.num_data.iloc[sum(over_index, [])]
            raw_data.label_data = raw_data.label_data.iloc[sum(over_index, [])]
            self.logger.info(raw_data.label_data.value_counts())

            
            self.logger.info(raw_data.str_data.info())
            self.logger.info(raw_data.num_data.info())

            intPrep = IntProcessing()
            intPrep.save_scaling_model(raw_data.num_data, train_version, how='MinMaxScaler')

            catPrep = CatProcessing()
            catPrep.save_one_hot_enc_model(raw_data.label_data, train_version)

            strPrep = StrProcessing()
            strPrep.save_tfidf_model_fit(raw_data.str_data, list(raw_data.str_data), (1, 1), 256, train_version)

            train_x = strPrep.load_tfidf_model_trans(raw_data.str_data, list(raw_data.str_data), 10000, train_version)
            train_y = catPrep.trnsfm_one_hot_enc_data(raw_data.label_data, train_version)
            Object.save_obj(list(train_y), name ='train_label_{}'.format(train_version))

            """
            Model Fitting
            """
            ## Decision Tree
            model = DecisionTreeClassification(version=train_version, mode='train', config=model_config)
            model.fit_decision_tree(train_x, train_y)
            true, pred = model.validation(train_x, train_y)

            ## Cnn Model
            for i in list(train_y):
                if i == 'normal':
                    pass
                else:
                    self.logger.info("\n ******** {} MODEL FITTING START ********".format(i))
                    normal_y = train_y[train_y['normal'] == 1].copy()
                    attack_y = train_y[train_y[i] == 1].copy()
                    temp_y = pd.concat([normal_y, attack_y])
                    temp_y = temp_y[['normal',i]].copy()
                    temp_x = train_x.iloc[temp_y.index]
                    random_idx = np.random.permutation(len(temp_x))
                    temp_x = temp_x.iloc[random_idx]
                    temp_y = temp_y.iloc[random_idx]

                    cnn_train_x = np.array(temp_x).reshape(temp_x.shape[0], temp_x.shape[1], 1)
                    cnn_train_y = np.array(temp_y).reshape(temp_y.shape[0], -1)        
                    model_config["x_data_shape"] = cnn_train_x.shape
                    model_config["y_data_shape"] = cnn_train_y.shape
                    model_config["att_name"] = i
                    model = AttackClassification(version=train_version, mode='train', config=model_config)

                    _, globals()['ai_history_{}'.format(i)] = model.optimize_nn(cnn_train_x, cnn_train_y)
                    model.validation(cnn_train_x, cnn_train_y)
                    self.logger.info("{} MODEL FITTING FINISH".format(i))
            
            return "OK"

        except Exception as err:
            self.logger.error(err)
            self.logger.error(traceback.print_exc())
            return None
        
        
class Prediction1(BaseComponent):
    """
    예측 모듈
    """
    def init(self):
        pass
    
    async def run(self, param):
        try:
            model_config = self.model_config
            model_id = model_config['common']['model_id']
            config_name = model_config['common']['config_name']
            
            """
            모델 버전 세팅
            """
            start = datetime.datetime.now().replace(microsecond=0) + timedelta(hours=9)
            pred_version = start.strftime("%Y_%m_%d")
            for timerange in range(500):
                if not os.path.exists('{}/{}/{}'.format(pwd, model_config['common']['model_path'],pred_version)):
                    new_time = start - timedelta(days=timerange+1)
                    pred_version = new_time.strftime("%Y_%m_%d")
                else:
                    break        

            """
            데이터 로드
            """
            # INSERT mode / config_type / model_id / model_name
            raw_data = DataCreation('prediction', 'COMMON', model_id, config_name)

            self.logger.info(raw_data.str_data.info())
            self.logger.info(raw_data.num_data.info())

            save_test_x = raw_data.str_data.copy()
            save_test_x.reset_index(drop = True, inplace = True)
            data_idx = raw_data.index_data.copy()
            data_idx.reset_index(drop = True, inplace = True)

            intPrep = IntProcessing()
            catPrep = CatProcessing()
            strPrep = StrProcessing()
            
            test_x = strPrep.load_tfidf_model_trans(raw_data.str_data, list(raw_data.str_data), 10000, pred_version)
            test_y = catPrep.trnsfm_one_hot_enc_data(raw_data.label_data, pred_version)

            """
            Model Prediction
            """
            ## Decision Tree
            model = DecisionTreeClassification(version=pred_version, mode='test', config=model_config)
#             pred = model.predict(test_x)
            true, pred = model.validation(test_x, test_y) ## for validation
            train_label = Object.load_obj('train_label_{}'.format(pred_version))
            dt_pred = [train_label[i] for i in pred]
            from collections import Counter            
            print(Counter(dt_pred))
            
            ## Cnn Model
            save_test_x['ai_label_pred'] = np.NaN
            XAI_result = pd.DataFrame()

            for i in list(set(dt_pred)):
                y_index = [index for index, att_name in enumerate(dt_pred) if att_name == i]

                if i == 'normal':
                    save_test_x.at[y_index,'ai_label_pred'] = 'NORMAL'
                else:
                    print("\n ******** {} MODEL PREDICTION START ********".format(i))
                    temp_x = test_x.iloc[y_index]
                    temp_idx = data_idx.iloc[y_index]

                    cnn_test_x = np.array(temp_x).reshape(temp_x.shape[0], temp_x.shape[1], 1)

                    model_config["x_data_shape"] = cnn_test_x.shape
                    model_config["att_name"] = i

                    model = AttackClassification(version=pred_version, mode='predict', config=model_config)

                    temp_y = test_y.iloc[y_index][['normal',i]].copy() ## for validation
                    cnn_test_y = np.array(temp_y).reshape(temp_y.shape[0], -1) ## for validation                
                    model.validation(cnn_test_x, cnn_test_y) ## for validation

                    pred = model.predict(cnn_test_x).argmax(axis=1).tolist()
                    save_test_x.at[y_index,'ai_label_pred'] = pred
                    save_test_x['ai_label_pred'] = np.where(save_test_x['ai_label_pred'] == 1, i, save_test_x['ai_label_pred'])
                    print("{} MODEL PREDICTION FINISH".format(i))  

                    """ XAI """        
                    print("{} XAI MODEL START".format(i))
                    cls_xai = ClsXai()
                    gradcam_batch_size = len(temp_x)
                    last_conv_layer_name = "activation_3"
                    classifier_layer_names = [
                                            "max_pooling1d_3"
                                            , "flatten"
                                            , "dense"
                                            , "predictions"
                                            ]

                    gradcam_heatmap_list = cls_xai.make_gradcam_heatmap_loop(temp_x.values, model.neural_network, last_conv_layer_name, classifier_layer_names)
                    invrs_preds = save_test_x.iloc[y_index]['ai_label_pred']
                    guided_model = cls_xai.build_guided_model(model.neural_network)
                    gb = cls_xai.guided_backprop(guided_model, gradcam_heatmap_list[1][0].reshape(1, -1, 1), last_conv_layer_name)
                    im = cv2.resize(gradcam_heatmap_list[1][1], dsize=gb.shape).reshape(-1, 1)

                    model_xai_stats = pd.DataFrame()
                    guided_model = cls_xai.build_guided_model(model.neural_network)

                    for i in range(gradcam_batch_size):
                        img_array, gradcam_heatmap = gradcam_heatmap_list[i]
                        gb = cls_xai.guided_backprop(guided_model, img_array.reshape(1, -1, 1), last_conv_layer_name)
                        im = gb*cv2.resize(gradcam_heatmap, dsize=gb.shape).reshape(-1, 1)
                        array = np.flip(im, -1).reshape(-1)
                        diff = array - np.median(array)
                        sorted_idx = sorted(range(len(diff)), key=lambda k: diff[k], reverse=True)

                        xai_res = {
                                'model_id': model_id
                                , 'version': pred_version
                                , 'logtime': temp_idx[list(data_idx)[0]].tolist()[i]
                                , 'src_ip': temp_idx[list(data_idx)[1]].tolist()[i]
                                , 'dst_ip': temp_idx[list(data_idx)[2]].tolist()[i]
                                , 'feature': [np.array(list(test_x))[sorted_idx]]
                                , 'score': [diff[sorted_idx]]
                                , 'prediction': invrs_preds.tolist()[i]
                                }

                        temp_df = pd.DataFrame(xai_res)
                        model_xai_stats = pd.concat([model_xai_stats, temp_df])

                    XAI_result = pd.concat([XAI_result, model_xai_stats])
                    save_test_x['ai_label_pred'] = np.where(save_test_x['ai_label_pred'].isin(['0.0',0,'nan']), 'NORMAL', save_test_x['ai_label_pred'])
                    save_test_x['version'] = pred_version

            XAI_result = XAI_result[XAI_result.prediction.isin(train_label)]
            XAI_result.reset_index(drop = True, inplace = True)                    
                                
#             insert_result = await execute_async_ch("insert into dti.dti_ai_xai values", model_xai_stats.values.tolist())
            
            return "OK"

        except Exception as err:
            self.logger.error(err)
            self.logger.error(traceback.print_exc())
            return None
        
        
def main(model_id, train, prediction, now=False):
    model_name = isExistModel(model_id)
    if model_name == None:
        sys.exit(1)
    else:
        # nc=NATS()
        loop = asyncio.get_event_loop()

        if train:
            if model_id == 335:
                if now:
                    loop.run_until_complete(Train1(loop, model_id, model_name).test_train())
                    sys.exit()
                else:
                    loop.run_until_complete(Train1(loop, model_id, model_name).start_train())
#                     sys.exit()
            else:
                self.logger.info("[TRAIN({})] model_i is invalid".format(model_id))
                sys.exit()

        if prediction:
            if model_id == 335:
                if now:
                    loop.run_until_complete(Prediction1(loop, model_id, model_name).test_pred())
                    sys.exit()
                else:
                    loop.run_until_complete(Prediction1(loop, model_id, model_name).start_pred())
#                     sys.exit()
            else:
                self.logger.info("[PREDICTION({})] model_id is invalid".format(model_id))
                sys.exit()
        try:
            loop.run_forever()
        finally:
            loop.close()

            
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        self.logger.info("python3 total_model.py -h or python3 total_model.py --help")
        sys.exit()

    parser = OptionParser(usage="Usage: python3 dti_v3_total_model.py -t -m [model_id] -now or python3 total_model.py -p -m [model_id] -now")
    parser.add_option("-t", action = "store_true", dest = "isTrain", default=False, help = "train")
    parser.add_option("-p", action = "store_true", dest = "isPred", default=False, help = "pred")
    parser.add_option("-n", action = "store_true", dest = "now", default=False, help = "once_run")
    parser.add_option("-m", type = int, dest = "MODEL_ID", help = "model_id")
    options, args = parser.parse_args()

    if options.isTrain == options.isPred:
        self.logger.error("you have to choose train or prediction")
        sys.exit()
    if options.MODEL_ID is None:
        self.logger.error("you have to input model id")
        sys.exit()

    main(options.MODEL_ID, options.isTrain, options.isPred, now=options.now)
    
    
