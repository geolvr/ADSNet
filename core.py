# -*- coding: utf-8 -*-
from keras.models import load_model
from keras import optimizers
from generator import DataGenerator, PredictDataGenerator,\
          getTimePeriod,ncFileDir_2016,ncFileDir_2017,M,npyWRFFileDir, getHoursGridFromNC, \
         getHoursGridFromNPY, getOneHourGridFromNPY, num_frames, wrf_fea_dim, \
         param_list, fea_dim, GuiTruthGridDir, num_frames_truth
import os
import numpy as np
import datetime
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import new_models
import scores

modelfileDir = 'models/'

def POD(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    true_positives = K.sum(ytrue * ypred)
    possible_positives = K.sum(ytrue)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def FAR(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    true_positives = K.sum(ytrue * ypred)
    predicted_positives = K.sum(ypred)
    precision = true_positives / (predicted_positives + K.epsilon())
    return 1 - precision

def TS(y_true, y_pred):
    ytrue = K.flatten(y_true)
    ypred = K.sigmoid(K.flatten(y_pred))
    ypred = K.round(ypred)
    N1 = K.sum(ytrue * ypred)
    N1pN2 = K.sum(ypred)
    N1pN3 = K.sum(ytrue)
    N2 = N1pN2 - N1
    N3 = N1pN3 - N1
    TS = N1 / (N1 + N2 + N3 + K.epsilon())
    return TS

def weight_loss(y_true,y_pred):
    pw = 25
    ytrue = K.flatten(y_true)
    ypred = K.flatten(y_pred)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=ypred,targets=ytrue,pos_weight=pw))

def binary_acc(y_true,y_pred):
    ypred = K.sigmoid(y_pred)
    return K.mean(K.equal(y_true, K.round(ypred)), axis=-1)

class RecordMetricsAfterEpoch(Callback):
    def on_epoch_end(self, epoch, logs={}):
        filename = modelrecordname
        with open('records/' + filename + '.txt','a') as f:
            f.write('epoch %d:\r\n' % (epoch+1))
            for key in ['loss', 'POD', 'FAR', 'TS', 'binary_acc', 'val_loss','val_POD', 'val_FAR', 'val_TS']:
                f.write('%s: %f   ' % (key, logs[key]))

def DoTrain(train_list, val_list):
    # parameters
    train_batchsize = 4
    val_batchsize = 1
    class_num = 2
    epochs_num = 30
    initial_epoch_num = 0

    # when train a new model -----------------------------------------

    model = new_models.WRF_decoder_model_with_x_att_v6_1()
    model = new_models.truthonly_model_v6_plain()
    model = new_models.StepDeep_model()
    model = new_models.WRF_decoder_model_wrfonly_v6_plain()
    model = new_models.WRF_decoder_model_with_x_att_v6_plain()

    # print(model.summary())

    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    print(dt_now)

    adam = optimizers.adam(lr=0.0001)
    model.compile(
                   loss=weight_loss,
                   optimizer=adam,
                   metrics=[POD,FAR,TS,binary_acc])
    modelfilename = "%s-%s-{epoch:02d}.hdf5" % (dt_now, model.name)

    global modelrecordname
    modelrecordname = dt_now + '_' + model.name

    checkpoint = ModelCheckpoint(modelfileDir + modelfilename, monitor='val_loss', verbose=1,
                                 save_best_only=False, mode='min')

    train_gen = DataGenerator(train_list, train_batchsize, class_num, generator_type='train')
    val_gen = DataGenerator(val_list, val_batchsize, class_num, generator_type='val')


    RMAE = RecordMetricsAfterEpoch()
    model.fit_generator(train_gen,
                        validation_data=val_gen,
                        epochs=epochs_num,
                        initial_epoch=initial_epoch_num,
                        # use_multiprocessing=True,
                        workers=3,
                        max_queue_size=50,
                        callbacks = [RMAE,checkpoint]
                       )

def DoTest_step_seq(test_list, model, modelfilepath, testset_disp):
    test_batchsize = 1
    M = 1
    test_gen = PredictDataGenerator(test_list, test_batchsize)
    print('generating test data and predicting...')
    ypred = model.predict_generator(test_gen, workers=3, verbose=1)  # [len(test_list),num_frames,159*159,1]
    ypred = 1.0 / (1.0 + np.exp(-ypred))  # if model doesn't include a sigmoid layer
    ## plot (the prediction for timesteps)  ------------------------------------
    with tf.device('/cpu:0'):
        for id, ddt_item in enumerate(test_list):
            ddt = datetime.datetime.strptime(ddt_item, '%Y%m%d%H%M')
            utc = ddt + datetime.timedelta(hours=-8)  # convert Beijing time into UTC time
            ft = utc + datetime.timedelta(hours=(-6) * M)
            nchour, delta_hour = getTimePeriod(ft)
            delta_hour += M * 6
            y_pred = ypred[id]     # [num_frames,159*159,1]
            for hour_plus in range(num_frames):
                y_pred_i = y_pred[hour_plus]
                dt = ddt + datetime.timedelta(hours=hour_plus)
                dt_item = dt.strftime('%Y%m%d%H%M')
                resDir = 'results/%s_set%s/' % (modelfilepath, testset_disp)
                if not os.path.isdir(resDir):
                    os.makedirs(resDir)
                with open(resDir + '%s_h%d' % (dt_item, hour_plus), 'w') as rfile:
                    for i in range(159*159):
                        rfile.write('%f\r\n' % y_pred_i[i])    # the probability value
                # print(dt_item)

def Test_att_weights(test_list, model, name):
    test_batchsize = 1
    test_gen = PredictDataGenerator(test_list, test_batchsize)
    alpha_lists = model.predict_generator(test_gen, workers=3, verbose=1)
    alpha_lists = np.array(alpha_lists)

    param_list = ['QICE_ave3_%d' % i for i in range(9)] + \
                 ['QSNOW_ave3_%d' % i for i in range(9)] + \
                 ['QGRAUP_ave3_%d' % i for i in range(9)] + ['W_max'] + ['RAINNC']

    import csv
    with open('att_weights_%s.csv' % name, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        for i in range(alpha_lists.shape[0]):
            csv_writer.writerow(['case %d:' % i,])
            tmp = ['hour-%d' % j for j in range(1, 13)]
            tmp.insert(0,'')
            tmp.append('ave')
            csv_writer.writerow(tmp)
            for k in range(alpha_lists.shape[2]):
                tmp = list(alpha_lists[i,:,k])
                ave = np.average(alpha_lists[i,:,k])
                tmp.insert(0,param_list[k])
                tmp.append(ave)
                csv_writer.writerow(tmp)
        csv_writer.writerow(['total average'])
        tmp = ['hour-%d' % j for j in range(1, 13)]
        tmp.insert(0, '')
        tmp.append('ave')
        for k in range(alpha_lists.shape[2]):      #  alpha_list   (sample_num, 12, fea_dim)
            cases_ave = np.average(alpha_lists, axis=0)   # (12,fea_dim)
            tmp = list(cases_ave[:,k])
            ave = np.average(cases_ave[:,k])
            tmp.insert(0, param_list[k])
            tmp.append(ave)
            csv_writer.writerow(tmp)
    return

if __name__ == "__main__":

    mode = 'TRAIN'
    # mode = 'TEST'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    KTF.set_session(sess)

    print('num_frames', num_frames)

    TrainSetFilePath = 'train_lite_new_12h.txt'

    ValSetFilePath = 'July.txt'
    TestSetFilePath = '20170809_6n.txt'
    testset_disp = '20170809_6n'

    if mode == 'TRAIN':
        train_list = []
        with open(TrainSetFilePath, 'r') as file:
            for line in file:
                train_list.append(line.rstrip('\n'))
        val_list = []
        with open(ValSetFilePath, 'r') as file:
            for line in file:
                val_list.append(line.rstrip('\n'))
        DoTrain(train_list, val_list)

    elif mode == 'TEST':
        test_list = []
        with open(TestSetFilePath, 'r') as file:
            for line in file:
                test_list.append(line.rstrip('\n'))

        for i in [14]:
            modelfilepath = '201908272358-WRF-ADSNet-%s.hdf5' % str(i).zfill(2)
            trained_model = load_model(modelfileDir + modelfilepath,
                                       {'weight_loss3': weight_loss, 'POD': POD, 'TS': TS,
                                        'FAR': FAR, 'binary_acc': binary_acc, 'num_frames': num_frames})
            DoTest_step_seq(test_list, trained_model, modelfilepath, testset_disp)
            resultfolderpath = modelfilepath + '_set%s' % testset_disp
            scores.eva(resultfolderpath, 0.5)

    sess.close()
