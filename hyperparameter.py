import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('task', 'train', '''[train, test, see]''')
tf.app.flags.DEFINE_string('data', 'cifar10', '''[cifar10, cifar100, svhn]''')

#parameter for optimization
tf.app.flags.DEFINE_string('optimizer', None, '''[sgd, sgdg, adamg]''')
tf.app.flags.DEFINE_float('learnRate', 0.1, '''initial learning rate''')
tf.app.flags.DEFINE_float('learnRateG', 0.1, '''initial learning rate for parameters on G(1,n)''')
tf.app.flags.DEFINE_float('momentum', 0.9, '''momentum''')
tf.app.flags.DEFINE_float('learnRateDecay', 0.2, '''learnRateDecay''')
tf.app.flags.DEFINE_bool('nesterov', True,
                         '''if True, use Nesterov momentum''')
tf.app.flags.DEFINE_bool('grassmann', False,
                         '''if True, weights can be on G(1,n)''')
tf.app.flags.DEFINE_float('grad_clip', None, 
                          '''threshold of clipping gradient by norm''')
tf.app.flags.DEFINE_float('adam_beta2', 0.99, 
                          '''exponential decay rates for 2nd moment estimates in adamg''')
tf.app.flags.DEFINE_float('adam_eps', 1e-8, 
                          '''small constant to avoid dividing by 0 in adamg''')

# parmeters for model definition
tf.app.flags.DEFINE_string('model', None, '''[resnet, vgg11, vgg13, vgg16, vgg19]''')
tf.app.flags.DEFINE_float('keep_prob', None, 
                          '''the probability that each element is kept in dropout''')
tf.app.flags.DEFINE_integer('depth', 10, '''depth for resnet''')
tf.app.flags.DEFINE_integer('widen_factor', 1, '''width for resnet''')

# parameters for regularization
tf.app.flags.DEFINE_float('weightDecay', None, 
                          '''L2 regularization for weights''')
tf.app.flags.DEFINE_float('biasDecay', None, 
                          '''L2 regularization for biases''')
tf.app.flags.DEFINE_float('gammaDecay', None, 
                          '''L2 regularization for scaling parameters in BN''')
tf.app.flags.DEFINE_float('betaDecay', None,
                          '''L2 regularization for offset parameters in BN''')
tf.app.flags.DEFINE_float('omega', None, 
                          '''orthogonality regularization for weight matrices''')

# parameters for image augmentation 
tf.app.flags.DEFINE_integer('crop_size', 32,
                            '''size of cropped image''')
tf.app.flags.DEFINE_integer('padding_size', 4, 
                            '''size of padding image on each side''')

# other parameters for training
tf.app.flags.DEFINE_integer('num_gpus', 1, 
                            '''how many GPUs to use''')
tf.app.flags.DEFINE_integer('batch_size', 128,
                            '''number of images to process in a batch for training''')
tf.app.flags.DEFINE_integer('vali_batch_size', 250,
                            '''number of images to process in a batch for test''')

# etc
tf.app.flags.DEFINE_integer('report_freq', 10, 
                            '''how often to log results''')
tf.app.flags.DEFINE_integer('vali_freq', 1000, 
                            '''how often to run the test''')
tf.app.flags.DEFINE_integer('save_freq', 10000, 
                            '''how often to save checkpoint''')
tf.app.flags.DEFINE_string('data_dir', './datasets/',
                           '''path to the data directory''')
tf.app.flags.DEFINE_string('train_dir', './logs/',
                           '''directory where to write event logs '''
                           '''and checkpoint''')
tf.app.flags.DEFINE_string('train_path', None, 
                           '''specific path to write event logs '''
                           '''and checkpoint''')

tf.app.flags.DEFINE_string('load', None, '''checkpoint file to load''')
tf.app.flags.DEFINE_string('visible_devices', None, '''CUDA_VISIBLE_DEVICES''')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            '''whether to log device placement''')
tf.app.flags.DEFINE_integer('max_to_keep', 1,
                            '''maximum number of recent checkpoint files to keep''')
tf.app.flags.DEFINE_bool('verbose', True, '''verbose''')
tf.app.flags.DEFINE_integer('k', 5, '''top-k error rate''')
