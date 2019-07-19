from network.ops import *
import tensorflow as tf
from utils.compute import *



def VGG19_slim(input, type, reuse,scope):
    # Define the feature to extract according to the type of perceptual
    if type == 'VGG54':
        # target_layer = scope + 'vgg_19/conv5/conv5_4'
        target_layer = 'vgg_19/conv5/conv5_4'
    elif type == 'VGG33':
        # target_layer = scope + 'vgg_19/conv2/conv2_2'
        target_layer = 'vgg_19/conv3/conv3_3'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, is_training=False, reuse=reuse)
    output = output[target_layer]

    return output
### vgg
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
  """Oxford Net VGG 19-Layers version E Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points

def res_block(input,feature_size=64,kernel_size=[3,3],strides=[1,1,1,1],name='resBlock'):
    with tf.variable_scope(name):
        conv1 = ReLU(instance_norm(conv_b(input,feature_size,kernel_size[0],kernel_size[1],strides,name='Resblock_conv1'),feature_size))
        conv2 = instance_norm(conv_b(conv1,feature_size,kernel_size[0],kernel_size[1],strides,name='Resblock_conv2'),feature_size)
        out = conv2 + input
        return out
def SE_resblock(input,feature_size=64,kernel_size=[3,3],strides=[1,1,1,1],name='resBlock'):
    with tf.variable_scope(name):
        conv1 = ReLU(
            instance_norm(conv_b(input, feature_size, kernel_size[0], kernel_size[1], strides, name='Resblock_conv1'),
                          feature_size))
        conv2 = instance_norm(
            conv_b(conv1, feature_size, kernel_size[0], kernel_size[1], strides, name='Resblock_conv2'), feature_size)
        conv2 = SE_block(conv2,ratio=16,name='SE_block')
        out = ReLU(conv2 + input)
        return out
def generator(input,reuse=False,args=None,name='Deblur'):
    with tf.variable_scope(name,reuse=reuse):
        x = ReLU(instance_norm(conv_b(input,args.n_feats,k_h=7,k_w=7,name='conv2d_L1'),args.n_feats),name='Relu_L1')
        for i in range(args.num_of_down_scale):
            x = ReLU(instance_norm(conv_b(x,args.n_feats*(2**(i+1)),strides=[1,2,2,1],name='Conv2d_down_'+str(i+1)+'_1'),args.n_feats*(2**(i+1))),name='Relu_down_'+str(i+1)+'_1')
            x = ReLU(instance_norm(conv_b(x,args.n_feats*(2**(i+1)),name='Conv2d_down_'+str(i+1)+'_2'),args.n_feats*(2**(i+1))),name='Relu_down_'+str(i+1)+'_2')
        for i in range(args.gen_resblocks):
            x = res_block(x,args.n_feats * (2 ** args.num_of_down_scale),name='res_block_'+str(i+1))
        for i in range(args.num_of_down_scale):
            size = x.get_shape().as_list()
            # x = ReLU(instance_norm(Deconv2d(x, [size[0], size[1] * 2, size[2] * 2, size[3]//2],name='Deconv2d_'+str(i+1)),size[3]//2),name='DeReLU_'+str(i+1))
            x = tf.image.resize_images(x,[size[1]*2, size[2] * 2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = ReLU(instance_norm(conv_b(x,size[3]//2,name='Conv2d_up_'+str(i+1)+'_1'),size[3]//2),name='Relu_up_'+str(i+1)+'_1')
            x = ReLU(instance_norm(conv_b(x,size[3]//2,name='Conv2d_up_'+str(i+1)+'_2'),size[3]//2),name='Relu_up_'+str(i+1)+'_2')

        x = conv_b(x,1,k_h=7,k_w=7,name='conv2d_L2')
        x = tf.nn.tanh(x)
        x = x + input
        x = tf.clip_by_value(x, -1.0, 1.0)

        return x
def generator2(input,reuse=False,args=None,name='Deblur'):
    with tf.variable_scope(name,reuse=reuse):
        x = ReLU(instance_norm(conv_b(input,args.n_feats,k_h=7,k_w=7,name='conv2d_L1'),args.n_feats),name='Relu_L1')
        x = SE_block(x, name='SE_block_L1')
        alist = []
        for i in range(args.num_of_down_scale):
            alist += [x]
            x = ReLU(instance_norm(conv_b(x,args.n_feats*(2**(i+1)),strides=[1,2,2,1],name='Conv2d_down_'+str(i+1)+'_1'),args.n_feats*(2**(i+1))),name='Relu_down_'+str(i+1)+'_1')
            x = ReLU(instance_norm(conv_b(x,args.n_feats*(2**(i+1)),name='Conv2d_down_'+str(i+1)+'_2'),args.n_feats*(2**(i+1))),name='Relu_down_'+str(i+1)+'_2')
            x = SE_block(x,name='SE_block_down_'+str(i))
        for i in range(args.gen_resblocks):
            x = SE_resblock(x,args.n_feats * (2 ** args.num_of_down_scale),name='res_block_'+str(i+1))
        for i in range(args.num_of_down_scale):
            size = x.get_shape().as_list()
            x_c = alist[-1-i]
            # x = ReLU(instance_norm(Deconv2d(x, [size[0], size[1] * 2, size[2] * 2, size[3]//2],name='Deconv2d_'+str(i+1)),size[3]//2),name='DeReLU_'+str(i+1))
            x = tf.image.resize_images(x,[size[1]*2, size[2] * 2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = ReLU(instance_norm(conv_b(x,size[3]//2,name='Conv2d_up_'+str(i+1)+'_1'),size[3]//2),name='Relu_up_'+str(i+1)+'_1')
            x = tf.concat((x,x_c),-1)
            x = ReLU(instance_norm(conv_b(x,size[3]//2,name='Conv2d_up_'+str(i+1)+'_2'),size[3]//2),name='Relu_up_'+str(i+1)+'_2')
            x = SE_block(x,name='SE_block_up_'+str(i))
        x = ReLU(instance_norm(conv_b(x, 32, name='Conv2d_L2_1'),32),name='Relu_L2_1')
        x = ReLU(instance_norm(conv_b(x, 16, name='Conv2d_L2_2'),16),name='Relu_L2_2')
        x = conv_b(x,1,k_h=7,k_w=7,name='conv2d_L2')
        x = tf.nn.tanh(x)
        x = x + input
        x = tf.clip_by_value(x, -1.0, 1.0)

        return x
def discriminator(x, reuse=False,args=None,name='discriminator'):


    with tf.variable_scope(name_or_scope=name, reuse=reuse):
        x = conv_b(x, args.n_feats,k_h=4,k_w=4,strides=[1,2,2,1],name='conv1')
        x = LReLU(instance_norm(x, dim=args.n_feats,name='inst_norm1'),name='LReLU1')

        for i in range(args.discrim_blocks):
            n = min(2 ** (i + 1), 8)
            x = conv_b(x, args.n_feats * n, k_h = 4, k_w = 4,strides=[1,2,2,1],name='conv%02d' % i)
            x = LReLU(instance_norm(x=x, dim=args.n_feats * n,name='instance_norm%02d' % i),name='LReLU%02d' % i)

        n = min(2 ** args.discrim_blocks, 8)
        x = conv_b(x, args.n_feats * n,k_h = 4, k_w = 4,name='conv_d1')
        x = LReLU(instance_norm(x=x, dim=args.n_feats * n,name='instance_norm_d1'),name='LReLU_dl')

        x = conv_b(x, 1,k_h=4,k_w=4,name='conv_d2')
        x = tf.nn.sigmoid(x)

        return x
def gen_loss(output,label,fake_prob,EPS,perceptual_mode):
    # loss_mse = tf.reduce_mean(tf.abs(output-label))
    with tf.name_scope('vgg19_1') as scope:
        extracted_feature_gen = VGG19_slim(output, perceptual_mode, reuse=False, scope=scope)
    with tf.name_scope('vgg19_2') as scope:
        extracted_feature_target = VGG19_slim(label, perceptual_mode, reuse=True, scope=scope)

    loss_vgg = tf.reduce_mean(tf.square(extracted_feature_gen-extracted_feature_target))
    adversarial_loss = -tf.reduce_mean(fake_prob)
    gen_loss = adversarial_loss + 100*loss_vgg
    return gen_loss
def discr_loss(gen_prob,real_prob,EPS=1e-8):
    # d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_prob+EPS, labels=tf.zeros_like(gen_prob)))
    # d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_prob+EPS, labels=tf.ones_like(real_prob)))
    # d_loss = d_loss_real + d_loss_fake

    d_loss_real = - tf.reduce_mean(real_prob)
    d_loss_fake = tf.reduce_mean(gen_prob)

    d_loss = d_loss_fake+d_loss_real
    return d_loss

def GP_loss(gene_img,label,args=None):
    epsilon = tf.random_uniform(shape=[args.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated_input = epsilon * label + (1 - epsilon) * gene_img
    gradient = tf.gradients(discriminator(interpolated_input, reuse=True,args=args,name='discriminator'), [interpolated_input])[0]
    gp_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gradient), axis=[1, 2, 3])) - 1))
    return gp_loss


def gen_loss2(output,label,fake_prob1,fake_prob2,fake_prob3,EPS,perceptual_mode):
    # loss_mse = tf.reduce_mean(tf.abs(output-label))
    with tf.name_scope('vgg19_1') as scope:
        extracted_feature_gen = VGG19_slim(output, perceptual_mode, reuse=False, scope=scope)
    with tf.name_scope('vgg19_2') as scope:
        extracted_feature_target = VGG19_slim(label, perceptual_mode, reuse=True, scope=scope)
    loss_vgg = tf.reduce_mean(tf.square(extracted_feature_gen-extracted_feature_target))

    # adversarial_loss1 = tf.reduce_mean(tf.square(fake_prob1-tf.random_uniform(shape=[fake_prob1.get_shape().as_list()[0],1],minval=0.9,maxval=1.1,dtype=tf.float32)))
    adversarial_loss1 = tf.reduce_mean(tf.nn.l2_loss(fake_prob1 - tf.ones_like(fake_prob1)))
    adversarial_loss2 = tf.reduce_mean(tf.nn.l2_loss(fake_prob2 - tf.ones_like(fake_prob2)))
    adversarial_loss3 = tf.reduce_mean(tf.nn.l2_loss(fake_prob3 - tf.ones_like(fake_prob3)))
    adversarial_loss = (adversarial_loss1+adversarial_loss2+adversarial_loss3)/3
    gen_loss = adversarial_loss + 100*loss_vgg
    return gen_loss

def discr_loss2(gen_prob,real_prob,EPS=1e-8):

    d_loss_fake = tf.reduce_mean(tf.nn.l2_loss(gen_prob - tf.zeros_like(gen_prob)))
    d_loss_real = tf.reduce_mean(tf.nn.l2_loss(real_prob - tf.ones_like(real_prob)))

    d_loss = d_loss_fake+d_loss_real
    return d_loss

def discriminator2(x, reuse=False,args=None,name='discriminator'):


    with tf.variable_scope(name_or_scope=name, reuse=reuse):
        x = conv_b(x, args.n_feats,k_h=4,k_w=4,strides=[1,2,2,1],name='conv1')
        x = LReLU(instance_norm(x, dim=args.n_feats,name='inst_norm1'),name='LReLU1')

        for i in range(args.discrim_blocks):
            n = min(2 ** (i + 1), 8)
            x = conv_b(x, args.n_feats * n, k_h = 4, k_w = 4,strides=[1,2,2,1],name='conv%02d' % i)
            x = LReLU(instance_norm(x=x, dim=args.n_feats * n,name='instance_norm%02d' % i),name='LReLU%02d' % i)
        out1 = conv_b(x,args.n_feats,k_h=1,k_w=1,name='conv2d_out1_1')
        out1 = LReLU(instance_norm(out1, dim=args.n_feats,name='inst_norm_out1_1'),name='LReLU_out1_1')
        out1 = tf.nn.sigmoid(conv_b(out1,1,name='conv2d_out1_2'))
        n = min(2 ** args.discrim_blocks, 8)
        x = conv_b(x, args.n_feats * n,k_h = 4, k_w = 4,name='conv_d1')
        x = LReLU(instance_norm(x=x, dim=args.n_feats * n,name='instance_norm_d1'),name='LReLU_dl')

        out2 = conv_b(x, 1,k_h=4,k_w=4,name='conv_out2')
        out2 = tf.nn.sigmoid(out2)

        x = conv_b(x, args.n_feats * n,k_h = 4, k_w = 4,strides=[1,2,2,1],name='conv_d2')
        x = LReLU(instance_norm(x, dim=args.n_feats,name='inst_norm_d2'),name='LReLU_d2')
        out3 = conv_b(x, 1,k_h=4,k_w=4,name='conv_out3')
        out3 = tf.nn.sigmoid(out3)
        return out1,out2,out3