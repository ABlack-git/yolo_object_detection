from network.yolo_v0 import YoloV0
from network.yolo_v0_1 import YoloV01
from network.yolo_v0_small import YoloSmall
from network.yolo_v0_small import YoloSmallPretrain
from data.dataset_generator import DatasetGenerator
import numpy as np


# loss_scale, training_set_imgs, training_set_labels, batch_size, learning_rate

def first_run_v1():
    params = {'coord_scale': 0,
              'noobj_scale': 0.001,
              'isobj_scale': 0.1,
              'training_set_imgs': "E:\Andrew\Dataset\Training set\Images",
              'training_set_labels': "E:\Andrew\Dataset\Training set\Annotations",
              'batch_size': 32,
              'learning_rate': 0.0001,
              'optimizer': 'Nesterov',
              'opt_param': 0.9,
              'threshold': 0.25,
              'save_path': "E:\Andrew\Ann_models\pretrain_model_3_0\weights",
              'training': True}
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloSmallPretrain(grid_size, img_size, params)
    net.set_logger_verbosity()
    sum_path="E:\Andrew\Ann_models\pretrain_model_3_0\summaries"
    net.optimize(5, sum_path)
    net.save(params.get('save_path'), 'model')
    net.close_sess()

def restore_and_run_v1():
    params = {'coord_scale': 0,
              'noobj_scale': 0.1,
              'isobj_scale': 1,
              'training_set_imgs': "E:\Andrew\Dataset\Training set\Images",
              'training_set_labels': "E:\Andrew\Dataset\Training set\Annotations",
              'testing_set_imgs': "E:\Andrew\Dataset\Testing set\Images",
              'testing_set_labels': "E:\Andrew\Dataset\Testing set\Annotations",
              'batch_size': 32,
              'learning_rate': 0.0001,
              'optimizer': 'Adam',
              'opt_param': 0.9,
              'threshold': 0.25,
              'save_path': "E:\Andrew\Ann_models\pretrain_model_3_0\weights",
              'training': True}
    model_path ="E:\Andrew\Ann_models\pretrain_model_3_0\weights\model-11240"
    meta_path ="E:\Andrew\Ann_models\pretrain_model_3_0\weights\model-11240.meta"
    img_size = (720, 480)
    grid_size = (36, 24)
    net = YoloSmallPretrain(grid_size, img_size, params, restore=True)
    net.restore(model_path, meta_path)
    #net.print_trainable_variables()
    sum_path="E:\Andrew\Ann_models\pretrain_model_3_0\summaries"
    #net.test_model(20)
    net.optimize(10, sum_path)
    net.save(params.get('save_path'), 'model')
    net.close_sess()


def first_run():
    params = {'coord_scale': 1,
              'noobj_scale': 0.01,
              'isobj_scale': 1,
              'training_set_imgs': "E:\Andrew\Dataset\Training set\Images",
              'training_set_labels': "E:\Andrew\Dataset\Training set\Annotations",
              'testing_set_imgs': "E:\Andrew\Dataset\Testing set\Images",
              'testing_set_labels': "E:\Andrew\Dataset\Testing set\Annotations",
              'batch_size': 20,
              'learning_rate': 0.00001,
              'optimizer': 'Nesterov',
              'opt_param': 0.8,
              'threshold': 0.25,
              'save_path': "E:\Andrew\Ann_models\model_8_0\weights",
              'training': True}
    img_size = (720, 480)
    grid_size = (20, 20)
    net = YoloV0(grid_size, img_size, params)
    net.set_logger_verbosity()
    sum_path="E:\Andrew\Ann_models\model_8_0\summaries"
    net.optimize(5, sum_path)
    net.save(params.get('save_path'), 'model')
    net.close_sess()


def restore_and_run():
    params = {'coord_scale': 1,
            'noobj_scale': 0.01,
            'isobj_scale': 1,
            'prob_noobj' : 0.01,
            'prob_isobj' : 1,
            'training_set_imgs': "E:\Andrew\Dataset\Training set\Images",
            'training_set_labels': "E:\Andrew\Dataset\Training set\Annotations",
            'batch_size': 32,
            'learning_rate': 0.0000001,
            'optimizer': 'Nesterov',
            'opt_param': 0.8,
            'threshold': 0.3,
            'save_path': "E:\Andrew\Ann_models\model_13_0\weights_3",
            'training': True}
    model_path ="E:\Andrew\Ann_models\model_13_0\weights_3\model-562"
    img_size = (720, 480)
    grid_size = (30, 20)
    net = YoloSmall(grid_size, img_size, params, restore=True)
    net.restore(model_path)
    sum_path="E:\Andrew\Ann_models\model_13_0\summaries"
    net.test_model(20)
    net.optimize(30, sum_path)
    net.save(params.get('save_path'), 'model')
    #test_preds(net)
    net.close_sess()

def restore_pretrained():
  params = {'coord_scale': 1,
            'noobj_scale': 0.01,
            'isobj_scale': 1,
            'prob_noobj' : 0.01,
            'prob_isobj' : 1,
            'training_set_imgs': "E:\Andrew\Dataset\Training set\Images",
            'training_set_labels': "E:\Andrew\Dataset\Training set\Annotations",
            'batch_size': 32,
            'learning_rate': 0.0000001,
            'optimizer': 'Nesterov',
            'opt_param': 0.5,
            'threshold': 0.3,
            'save_path': "E:\Andrew\Ann_models\model_13_0\weights_3",
            'training': True}
  var_list=['Conv_1/weights', 'Conv_1/batch_norm_layer/gamma', 'Conv_1/batch_norm_layer/beta',
            'Conv_2/weights', 'Conv_2/batch_norm_layer/gamma', 'Conv_2/batch_norm_layer/beta',
            'Conv_3/weights', 'Conv_3/batch_norm_layer/gamma', 'Conv_3/batch_norm_layer/beta',
            'Conv_4/weights', 'Conv_4/batch_norm_layer/gamma', 'Conv_4/batch_norm_layer/beta',
            'Conv_5/weights', 'Conv_5/batch_norm_layer/gamma', 'Conv_5/batch_norm_layer/beta',
            'Conv_6/weights', 'Conv_6/batch_norm_layer/gamma', 'Conv_6/batch_norm_layer/beta']
            #'Conv_7/weights', 'Conv_7/batch_norm_layer/gamma', 'Conv_7/batch_norm_layer/beta',
            #'Conv_8/weights', 'Conv_8/batch_norm_layer/gamma', 'Conv_8/batch_norm_layer/beta']
  model_path ="E:\Andrew\Ann_models\pretrain_model_3_0\weights\model-14050"
  # meta_path ="E:\Andrew\Ann_models\model_7_0\weights\model-27000.meta"
  sum_path="E:\Andrew\Ann_models\model_13_0\summaries"
  img_size = (720, 480)
  grid_size = (30, 20)
  net = YoloSmall(grid_size, img_size, params, restore=True)
  net.restore(model_path, var_names=var_list)
  net.optimize(30,sum_path)
  net.save(params.get('save_path'), 'model')
  net.close_sess()

def test_preds(net):
  img_path="E:\Andrew\Dataset\Training set\Images"
  labels_path="E:\Andrew\Dataset\Training set\Annotations"
  img_size = (720, 480)
  grid_size = (30, 20)
  dataset=DatasetGenerator(img_path,labels_path,img_size,grid_size,1)
  batch=dataset.get_minibatch(32)
  imgs, _ =next(batch)
  preds=net.get_predictions(imgs)
  for img in preds:
    for p in img:
      if p>0.95:
          print(p,end=' ')
    print('')

if __name__ == '__main__':
  restore_and_run()