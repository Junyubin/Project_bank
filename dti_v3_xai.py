import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

import scipy
import os

import tensorflow as tf
from tensorflow import keras

from dti_v3_utils import *
from dti_v3_prep import *
from dti_v3_data import *
from dti_v3_model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# print(tf.config.list_physical_devices('GPU'))
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


########################### XAI ################################
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class ClsXai:
    def __init__(self):
        pass

    def deprocess_image(self, x):
        """Same normalization as in:
        https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
        """
        # normalize tensor: center on 0., ensure std is 0.25
        x = x.copy()
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.25

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')

        return x

    def guided_backprop(self, model, img, layer_name):
        """
        Returns guided backpropagation image.

        Parameters
        ----------
        model : a keras model object

        img : an img to inspect with guided backprop.

        layer_name : a string
            a layer name for calculating gradients.


        """
        part_model = keras.Model(model.inputs, model.get_layer(layer_name).output)

        with tf.GradientTape() as tape:
            #         model_input = indentity_model(img)
            #         tape.watch(model_input)
            #         part_output = part_model(img)
            f32_img = tf.cast(img, tf.float32)
            tape.watch(f32_img)
            part_output = part_model(f32_img)

        grads = tape.gradient(part_output, f32_img)[0].numpy()

        # delete copied model
        del part_model

        return grads

    def build_guided_model(self, model):
        """
        Builds guided model

        """

        model_copied = keras.models.clone_model(model)
        model_copied.set_weights(model.get_weights())

        @tf.custom_gradient
        def guidedRelu(x):
            def grad(dy):
                return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy

            return tf.nn.relu(x), grad

        layer_dict = [layer for layer in model_copied.layers[1:] if hasattr(layer, 'activation')]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu

        return model_copied
    
    def make_gradcam_heatmap_loop(self,
            data, model, last_conv_layer_name, classifier_layer_names
    ):

        """
        Makes grad cam heatmap.

        Parameters
        ----------
        img_array : an input image

        model : a keras model object

        last_conv_layer_name : a string
            The gradients of y^c w.r.t the layer activation will be calculated.

        classifier_layer_names : a sequence
            layer names from next to the last conv layer to the output layer.

        """

        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
        # print("last_conv_layer_model.summary()\n", last_conv_layer_model.summary())

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            if layer_name == 'predictions':
                l = keras.layers.Dense(model.get_layer(layer_name).units, name="predictions_gradcam")
                x = l(x)
                l.set_weights(model.get_layer(layer_name).get_weights())
            else:
                x = model.get_layer(layer_name)(x)
        classifier_model = keras.Model(classifier_input, x)

        heatmap_list = []
        for img_array in data:
            # print("img from data", img_array.shape)
            img_array = img_array.reshape(1, -1)
            # Then, we compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            with tf.GradientTape() as tape:
                # Compute activations of the last conv layer and make the tape watch it
                # print("img_array.shape :", img_array.shape)
                last_conv_layer_output = last_conv_layer_model(img_array)
                tape.watch(last_conv_layer_output)
                # Compute class predictions
                preds = classifier_model(last_conv_layer_output)
                top_pred_index = tf.argmax(preds[0])
                top_class_channel = preds[:, top_pred_index]

            # This is the gradient of the top predicted class with regard to
            # the output feature map of the last conv layer
            grads = tape.gradient(top_class_channel, last_conv_layer_output)

            # This is a vector where each entry is the mean intensity of the gradient
            # over a specific feature map channel
            # print("grads.shape :", grads.shape)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the top predicted class
            last_conv_layer_output = last_conv_layer_output.numpy()[0]
            pooled_grads = pooled_grads.numpy()
            # print("pooled_grads.shape :", pooled_grads.shape)
            # print("last_conv_layer_output.shape :", last_conv_layer_output.shape)
            for i in range(pooled_grads.shape[-1]):
                last_conv_layer_output[:, i] *= pooled_grads[i]

            # The channel-wise mean of the resulting feature map
            # is our heatmap of class activation
            #     heatmap = np.mean(last_conv_layer_output, axis=-1)
            heatmap = last_conv_layer_output

            # For visualization purpose, we will also normalize the heatmap between 0 & 1
            heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
            heatmap_list.append((img_array, heatmap))

        return heatmap_list

    ## st grad_CAM : modified "guided_grad_cam.make_gradcam_heatmap()"
    def make_gradcam_heatmap_(self,
            img_array, model, last_conv_layer_name, classifier_layer_names
    ):
        """
        Makes grad cam heatmap.

        Parameters
        ----------
        img_array : an input image

        model : a keras model object

        last_conv_layer_name : a string
            The gradients of y^c w.r.t the layer activation will be calculated.

        classifier_layer_names : a sequence
            layer names from next to the last conv layer to the output layer.

        """

        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
        # print("last_conv_layer_model.summary()\n", last_conv_layer_model.summary())

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            if layer_name == 'predictions':
                l = keras.layers.Dense(model.get_layer(layer_name).units, name="predictions_gradcam")
                x = l(x)
                l.set_weights(model.get_layer(layer_name).get_weights())
            else:
                x = model.get_layer(layer_name)(x)
        classifier_model = keras.Model(classifier_input, x)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            # print("img_array.shape :", img_array.shape)
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        # print("grads.shape :", grads.shape)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        # print("pooled_grads.shape :", pooled_grads.shape)
        # print("last_conv_layer_output.shape :", last_conv_layer_output.shape)
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        #     heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = last_conv_layer_output

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        return heatmap
    ## en grad_CAM


def xai_train():
    cls_xai = ClsXai()

    dataClass = DataCreation()
    # print(dataClass)
    # print("str_data.info()", dataClass.str_data.info())
    # print("num_data.info()", dataClass.num_data.info())

    intPrep = IntProcessing()
    intPrep.save_scaling_model(dataClass.num_data, 'test_v1', how='MinMaxScaler')

    catPrep = CatProcessing()
    catPrep.save_one_hot_enc_model(dataClass.label_data, 'test_v1')

    strPrep = StrProcessing()
    strPrep.save_tfidf_model_fit(dataClass.str_data, list(dataClass.str_data), (1, 1), 256, 'test_v1')
    prep_data = strPrep.load_tfidf_model_trans(dataClass.str_data, list(dataClass.str_data), 10000, 'test_v1')
#     print("prep_data.info() :", prep_data.info())
#     print("prep_data.head() :", prep_data.head())

    label_df = catPrep.trnsfm_one_hot_enc_data(dataClass.label_data, 'test_v1')

    data_dict = train_test_split(dataClass.index_data, prep_data, label_df)
    # print(data_dict.keys())

#     model_config = {
#         "common": {
#             "model_name": "cnn_model",
#             "model_path": "cnn_model_path"
#         },
#         "train": {
#             "optimizer_help": ['Adam', 'SGD'],
#             "optimizer": 'Adam',
#             "learning_rate": 0.001,
#             "batch_size": 64,
#             "epochs": 1,
#             "result_table": "dti.dti_ai_log"
#         },
#         "predict": {
#             "batch_size": 8
#         }
#     }
    """ LOAD CONFIG"""
    model_config = Params.get_config('ALL', '335', 'dti_v3_model')
    
    model_config["x_data_shape"] = data_dict['X_train'].shape
    model_config["y_data_shape"] = data_dict['Y_train'].shape

    model = AttackClassification(version='test_v1', mode='train', config=model_config)
    history = model.optimize_nn(data_dict['X_train'], data_dict['Y_train'])
    accuracy, f1, precision, recall = model.validation(data_dict['X_test'], data_dict['Y_test'])

    # print(history[1].history['loss'][0])

    last_conv_layer_name = "activation_3"
    # TODO in dti_v3_model.py
    #  remove the last maxpooling at least
    #  rename the last layer ='predictions'
    #  Dense(self.y_data_shape[1], activation='softmax', name='predictions')
    classifier_layer_names = ["global_average_pooling1d", "predictions"]

    ### st grad_CAM
    test_idx = 3
    gradcam_heatmap = cls_xai.make_gradcam_heatmap_(data_dict['X_test'][test_idx].reshape(1, -1), model.neural_network,
                                                    last_conv_layer_name, classifier_layer_names)

    # print(gradcam_heatmap.shape)
    plt.figure(figsize=(20, 5))
    plt.imshow(gradcam_heatmap)
    plt.show()
    plt.close()

    guided_model = build_guided_model(model.neural_network)
    gb = guided_backprop(guided_model, data_dict['X_test'][test_idx].reshape(1, -1, 1), last_conv_layer_name)

    plt.figure(figsize=(20, 5))
    plt.imshow(np.flip(deprocess_image(gb), -1))
    plt.show()
    plt.close()

    # print(gradcam_heatmap.shape, gb.shape)
    # print(cv2.resize(gradcam_heatmap, dsize=gb.shape).reshape(1, -1).max())
    im = gb  # *cv2.resize(gradcam_heatmap, dsize=gb.shape).reshape(1, -1)
    # ggc = deprocess_image(guided_grad_cam(gb, gradcam_heatmap))

    plt.figure(figsize=(20, 10))
    plt.imshow(np.flip(im, -1))
    plt.show()
    plt.close()

    array = np.flip(im, -1).reshape(-1)
    diff = abs(array - np.median(array))
    sorted_idx = sorted(range(len(diff)), key=lambda k: diff[k], reverse=True)

    # print(array[sorted_idx][:20])
    # print(np.array(list(prep_data))[sorted_idx][:20])

    # print("median :", np.median(array), ", mean :", np.mean(array))

    
def xai_train_by_loop():
    cls_xai = ClsXai()

    dataClass = DataCreation()
    print(dataClass)
    print("str_data.info()", dataClass.str_data.info())
    print("num_data.info()", dataClass.num_data.info())

    intPrep = IntProcessing()
    intPrep.save_scaling_model(dataClass.num_data, 'test_v1', how='MinMaxScaler')

    catPrep = CatProcessing()
    catPrep.save_one_hot_enc_model(dataClass.label_data, 'test_v1')

    strPrep = StrProcessing()
    strPrep.save_tfidf_model_fit(dataClass.str_data, list(dataClass.str_data), (1, 1), 256, 'test_v1')
    prep_data = strPrep.load_tfidf_model_trans(dataClass.str_data, list(dataClass.str_data), 10000, 'test_v1')
    print("prep_data.info() :", prep_data.info())
    print("prep_data.head() :", prep_data.head())

    print("dataClass.label_data :", dataClass.label_data)
    label_df = catPrep.trnsfm_one_hot_enc_data(dataClass.label_data, 'test_v1')

    data_dict = train_test_split(dataClass.index_data, prep_data, label_df)
    print(data_dict.keys())
    print(data_dict['X_test'][3].shape)
    print(data_dict['X_test'][3].reshape(1, -1).shape)
    print(data_dict['X_test'].shape)
    print(data_dict['X_test'].reshape(data_dict['X_test'].shape[0], 1, -1).shape)
    
    """ LOAD CONFIG"""
    model_config = Params.get_config('ALL', '335', 'dti_v3_model')

    model_config["x_data_shape"] = data_dict['X_train'].shape
    model_config["y_data_shape"] = data_dict['Y_train'].shape
    print('model_config["x_data_shape"]', model_config["x_data_shape"], 'model_config["y_data_shape"]', model_config["y_data_shape"])

#     model = AttackClassification(version='test_v1', mode='predict', config=model_config)
    model = AttackClassification(version='test_v1', mode='train', config=model_config)
    history = model.optimize_nn(data_dict['X_train'], data_dict['Y_train'])

    accuracy, f1, precision, recall = model.validation(data_dict['X_test'], data_dict['Y_test'])

    # print(history[1].history['loss'][0])

    ### st grad_CAM
    st_idx = 0
    gradcam_batch_size = 100
    last_conv_layer_name = "activation_3"
    classifier_layer_names = [
                            "max_pooling1d_3"
                            , "flatten"
                            , "dense"
                            , "predictions"
                            ]

    gradcam_heatmap_list = cls_xai.make_gradcam_heatmap_loop(data_dict['X_test'][st_idx: st_idx + gradcam_batch_size], model.neural_network,
                                                    last_conv_layer_name, classifier_layer_names)
    preds = model.predict(data_dict['X_test'][st_idx: st_idx + gradcam_batch_size])
    print("preds :", preds)
    invrs_preds = catPrep.inverse_transform(preds, "test_v1")
    print("preds :", np.argmax(preds[0]))
    print("invrs_preds :", invrs_preds)
    print("invrs_preds :", invrs_preds[0])
    print("invrs_preds :", invrs_preds[1])

    guided_model = cls_xai.build_guided_model(model.neural_network)

    model_xai_stats = pd.DataFrame()
    for i in range(gradcam_batch_size):
        img_array, gradcam_heatmap = gradcam_heatmap_list[i]
        print("gradcam_heatmap.shape :", gradcam_heatmap.shape)
        print("gradcam_heatmap.shape :", gradcam_heatmap.shape)
        # plt.figure(figsize=(20, 5))
        # plt.imshow(gradcam_heatmap)
        # plt.show()
        # plt.close()

        print("img_array.shape in loop:", img_array.shape)
        print("img_array.reshape(1, -1, 1).shape :", img_array.reshape(1, -1, 1).shape)
        gb = cls_xai.guided_backprop(guided_model, img_array.reshape(1, -1, 1), last_conv_layer_name)

        # plt.figure(figsize=(20, 5))
        # plt.imshow(np.flip(cls_xai.deprocess_image(gb), -1))
        # plt.show()
        # plt.close()

        print("gradcam_heatmap.shape :", gradcam_heatmap.shape, "gb.shape :", gb.shape)
        print("cv2.resize(gradcam_heatmap, dsize=gb.shape).reshape(1, -1).max() :",
              cv2.resize(gradcam_heatmap, dsize=gb.shape).reshape(1, -1).max())
        im = gb  # *cv2.resize(gradcam_heatmap, dsize=gb.shape).reshape(1, -1)
        # ggc = deprocess_image(guided_grad_cam(gb, gradcam_heatmap))

        # plt.figure(figsize=(20, 10))
        # plt.imshow(np.flip(im, -1))
        # plt.show()
        # plt.close()

        array = np.flip(im, -1).reshape(-1)
        diff = array - np.median(array)
        # diff = abs(array - np.median(array))
        sorted_idx = sorted(range(len(diff)), key=lambda k: diff[k], reverse=True)

        # print(array[sorted_idx][:20])
        # print(diff[sorted_idx][:20])
        # print(np.array(list(prep_data))[sorted_idx][:20])

        print("\nmedian :", np.median(array), ", mean :", np.mean(array))
        xai_res = {
            'model_id': 1
            , 'version': 'test_v1'
            , 'logtime': data_dict['Y_indexes'][st_idx + i][0].to_pydatetime()
            , 'src_ip': data_dict['Y_indexes'][st_idx + i][1]
            , 'dst_ip': data_dict['Y_indexes'][st_idx + i][2]
            , 'feature': [np.array(list(prep_data))[sorted_idx]]
            , 'score': [diff[sorted_idx]]
            , 'prediction': invrs_preds[i][0]
            }
        temp_df = pd.DataFrame(xai_res)
        model_xai_stats = pd.concat([model_xai_stats, temp_df])

        print('logtime', data_dict['Y_indexes'][st_idx + i][0].to_pydatetime()
            , 'src_ip', data_dict['Y_indexes'][st_idx + i][1]
            , 'dst_ip', data_dict['Y_indexes'][st_idx + i][2]
            , '\nx_data', [[list(prep_data)[j], data_dict['X_test'][i][j][0]] for j in range(len(list(prep_data))) if data_dict['X_test'][i][j] != 0]
            , '\nfeature', [np.array(list(prep_data))[sorted_idx][:20]]
            , '\nscore', [diff[sorted_idx][:20]]
            , '\nprediction', invrs_preds[i][0])
        abs_diff = abs(diff)
        abs_sorted_idx = sorted(range(len(abs_diff)), key=lambda k: abs_diff[k], reverse=True)
        print('\nabs_feature', [np.array(list(prep_data))[abs_sorted_idx][:20]]
            , '\nabs_score', [abs_diff[abs_sorted_idx][:20]])
    # print(model_xai_stats.head())

    # insert dataFrame into DB
    # execute_ch("insert into dti.dti_ai_xai values", model_xai_stats.values.tolist())
    

# if __name__ == '__main__':
#     xai_train()
#     xai_train_by_loop()



