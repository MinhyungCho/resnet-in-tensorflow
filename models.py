#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def model(images, bn_param, keep_prob, num_classes, device, FLAGS):
  if 'vgg' in FLAGS.model:
    import vgg
    if FLAGS.model=='vgg11': depth = [1,1,2,2,2]
    elif FLAGS.model=='vgg13': depth = [2,2,2,2,2]
    elif FLAGS.model=='vgg16': depth = [2,2,3,3,3]
    elif FLAGS.model=='vgg19': depth = [2,2,4,4,4]
    else:
      assert False, 'unknown model'

    logits = vgg.vgg(images, bn_param, keep_prob, depth, num_classes, device=device)
    return logits

  elif FLAGS.model=='resnet':
    assert (FLAGS.depth - 4) % 6 == 0, 'depth should be 6n+4'
    import resnet
    n = (FLAGS.depth-4)//6
    k = FLAGS.widen_factor

    logits = resnet.inference(images, bn_param, keep_prob, n=n, k=k, num_classes=num_classes, device=device)
    return logits

  else:
    assert False, 'unknown model'
