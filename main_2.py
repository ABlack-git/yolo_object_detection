from network.yolo_v0 import YoloV0

def run_training(cfg, epochs, summ_step):
	training_set=["E:\Andrew\Dataset\Training set\Images","E:\Andrew\Dataset\Training set\Annotations"]
	valid_set=None
	net=YoloV0(cfg)
	net.optimize(epochs, training_set, valid_set, summ_step)

def restore_and_train(cfg, epochs, summ_step):
	training_set=["E:\Andrew\Dataset\Training set\Images","E:\Andrew\Dataset\Training set\Annotations"]
	valid_set=None
	net=YoloV0(cfg)
	net.restore("E:\Andrew\Ann_models\weights\model_6l_test\model_6l_test-900")
	net.optimize(epochs, training_set, valid_set, summ_step)


if __name__ == '__main__':
	cfg="E:\Andrew\project\yolo_object_detection\cfg\model_6l_test.cfg"
	# restore_and_train(cfg,5)
	run_training(cfg, 5, 2)