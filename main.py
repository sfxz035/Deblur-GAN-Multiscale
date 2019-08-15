import tensorflow as tf
import dataset
import network.model as model
import numpy as np
import os
import argparse
import cv2 as cv
import scipy.misc

from utils.compute import *
os.environ["CUDA_VISIBLE_DEVICES"] = '1'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.InteractiveSession(config = config)


parser = argparse.ArgumentParser()
parser.add_argument("--train_file",default="./data_face/train")
parser.add_argument("--test_file",default="./data_face/test")
parser.add_argument("--batch_size",default=1,type=int)
parser.add_argument("--savenet_path",default='./libSaveNet/savenet3_loss/')
parser.add_argument("--vgg_ckpt",default='./libSaveNet/vgg_ckpt/vgg_19.ckpt')
parser.add_argument("--epoch",default=50,type=int)
parser.add_argument("--learning_rate",default=0.0001,type=float)
parser.add_argument("--crop_size",default=148,type=int) #72  148
parser.add_argument("--num_train",default=15000,type=int)
parser.add_argument("--num_test",default=1500,type=int)
parser.add_argument("--EPS",default=1e-12,type=float)
parser.add_argument("--perceptual_mode",default='VGG33')

parser.add_argument("--num_of_down_scale", type = int, default = 2)
parser.add_argument("--gen_resblocks", type = int, default = 9)
parser.add_argument("--n_feats", type = int, default = 64)
parser.add_argument("--discrim_blocks", type = int, default = 2)

args = parser.parse_args()

def GAN_train(args):

    x_train, y_train = dataset.load_imgs_label(args.train_file, crop_size=args.crop_size)
    x_test, y_test = dataset.load_imgs_label(args.test_file, crop_size=args.crop_size)

    genInput = tf.placeholder(tf.float32,shape = [args.batch_size,args.crop_size,args.crop_size,3])
    genLabel = tf.placeholder(tf.float32,shape = [args.batch_size,args.crop_size,args.crop_size,3])
    genOutput = model.generator2(genInput,args=args,name='generator')

    ### 判别器单尺度输出和wgan loss
    # discr_outlabel = model.discriminator(genLabel,args=args,name='discriminator')
    # discr_outGenout = model.discriminator(genOutput,args=args,reuse=True,name='discriminator')
    # gen_loss = model.gen_loss(genOutput,genLabel,discr_outGenout,args.EPS,args.perceptual_mode)
    # dis_loss = model.discr_loss(discr_outGenout,discr_outlabel)+10*model.GP_loss(genInput,genLabel,args=args)

    ### 判别器多尺度输出和loss计算，改为lsganlosss
    discr_outlabel1,discr_outlabel2,discr_outlabel3 = model.discriminator2(genLabel,args=args,name='discriminator')
    discr_outGenout1,discr_outGenout2,discr_outGenout3 = model.discriminator2(genOutput,args=args,reuse=True,name='discriminator')
    gen_loss = model.gen_loss2(genOutput,genLabel,discr_outGenout1,discr_outGenout2,discr_outGenout3,args.EPS,args.perceptual_mode)
    dis_loss1 = model.discr_loss2(discr_outGenout1,discr_outlabel1)
    dis_loss2 = model.discr_loss2(discr_outGenout2,discr_outlabel2)
    dis_loss3 = model.discr_loss2(discr_outGenout3,discr_outlabel3)
    dis_loss = (dis_loss1+dis_loss2+dis_loss3)/3



    PSNR = compute_psnr(genOutput,genLabel,convert=True)

    tf.summary.scalar('genloss', gen_loss)
    tf.summary.scalar('disloss', dis_loss)
    tf.summary.scalar('PSNR', PSNR)
    # tf.summary.image('out', genOutput)
    summary_op = tf.summary.merge_all()
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list,max_to_keep=40)
    # var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
    #             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    genvar_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disvar_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    gensave = tf.train.Saver(genvar_list,max_to_keep=10)
    dissave = tf.train.Saver(disvar_list,max_to_keep=10)

    gen_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='generator'))
    with tf.control_dependencies([gen_updates_op]):
        gentrain_step = tf.train.AdamOptimizer(args.learning_rate).minimize(gen_loss,var_list=genvar_list)
    dis_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='discriminator'))
    with tf.control_dependencies([dis_updates_op]):
        distrain_step = tf.train.AdamOptimizer(args.learning_rate).minimize(dis_loss,var_list=disvar_list)

    vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    vgg_restore = tf.train.Saver(vgg_var_list)

    train_writer = tf.summary.FileWriter('./my_graph/train3_loss', sess.graph)
    test_writer = tf.summary.FileWriter('./my_graph/test3_loss')
    valid_writer = tf.summary.FileWriter('./my_graph/valid3_loss')
    tf.global_variables_initializer().run()

    vgg_restore.restore(sess, args.vgg_ckpt)

    # savepath = './libSaveNet/savenet/GAN_net450000.ckpt-done'
    # saver.restore(sess, savepath)

    # last_file = tf.train.latest_checkpoint(args.savenet_path)
    # if last_file:
    #     saver.restore(sess, last_file)
    count, m = 0, 0
    for ep in range(args.epoch):
        batch_idxs = len(x_train) // args.batch_size
        for idx in range(batch_idxs):
            # batch_input = x_train[idx * args.batch_size: (idx + 1) * args.batch_size]
            # batch_labels = y_train[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_input, batch_labels = dataset.random_batch(x_train,y_train,args.batch_size)
            # for i in range(2):
            sess.run(distrain_step, feed_dict={genInput: batch_input, genLabel: batch_labels})
            sess.run(gentrain_step, feed_dict={genInput: batch_input, genLabel: batch_labels})

            count += 1
            if count % 100 == 0:
                m += 1
                batch_input_test, batch_labels_test = dataset.random_batch(x_test, y_test, args.batch_size)

                PSNR_train = sess.run(PSNR, feed_dict={genInput: batch_input,genLabel: batch_labels})
                PSNR_test = sess.run(PSNR, feed_dict={genInput: batch_input_test, genLabel: batch_labels_test})

                genloss_train = sess.run(gen_loss,feed_dict={genInput:batch_input,genLabel:batch_labels})
                disloss_train = sess.run(dis_loss,feed_dict={genInput:batch_input,genLabel:batch_labels})
                genloss_test = sess.run(gen_loss,feed_dict={genInput:batch_input_test,genLabel:batch_labels_test})
                disloss_test = sess.run(dis_loss,feed_dict={genInput:batch_input_test,genLabel:batch_labels_test})
                print("Epoch: %-5.2d step: %2d" % ((ep), count),
                      "\n",'train/test_PSNR: %-12.8f' % PSNR_train,PSNR_test,
                      "\t", 'train/test_genloss: %-12.8f' % genloss_train,genloss_test,
                      "\t", 'train/test_disloss: %-12.8f' % disloss_train,disloss_test)
                train_writer.add_summary(sess.run(summary_op, feed_dict={genInput: batch_input, genLabel: batch_labels}), m)
                test_writer.add_summary(sess.run(summary_op, feed_dict={genInput: batch_input_test,
                                                                     genLabel: batch_labels_test}), m)
        test_batch_idxs = len(x_test) // args.batch_size
        sum_PSNR = 0
        for test_idx in range(test_batch_idxs):
            batch_input_valid = x_test[test_idx * args.batch_size: (test_idx + 1) * args.batch_size]
            batch_labels_valid = y_test[test_idx * args.batch_size: (test_idx + 1) * args.batch_size]
            PSNR_test = sess.run(PSNR, feed_dict={genInput: batch_input_valid, genLabel: batch_labels_valid})
            sum_PSNR += PSNR_test
        sum_PSNR /= test_batch_idxs
        summary = tf.Summary()
        summary.value.add(tag='valid_PSNR', simple_value=sum_PSNR )
        valid_writer.add_summary(summary,ep)
        print("Epoch: %-5.2d PSNR_valid: %-5.8f" % ((ep), sum_PSNR ))
        saver.save(sess, os.path.join(args.savenet_path, 'GAN_net%d.ckpt-done' % (count)))


def adtest(args):
    savepath = './libSaveNet/savenet3_148loss/GAN_net139788.ckpt-done'
    path_blur = './data_face/valid2/face_blur'
    path_sharp = './data_face/valid2/face_sharp'
    blur_file = os.listdir(path_blur)
    sharp_file = os.listdir(path_sharp)
    blur_file.sort()
    sharp_file.sort()
    dir_blur,dir_sharp = [],[]
    for each in blur_file:
        dir_blur += [os.path.join(path_blur,each)]
    for each in sharp_file:
        dir_sharp += [os.path.join(path_sharp,each)]
    x = tf.placeholder(tf.float32, shape=[1, 148, 148, 3])
    y_ = tf.placeholder(tf.float32, shape=[1, 148, 148, 3])
    y = model.generator2(x, args=args,name='generator')
    loss = tf.reduce_mean(tf.square(y - y_))
    PSNR = compute_psnr(y, y_)
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=None)
    tf.global_variables_initializer().run()
    saver.restore(sess, savepath)
    for i in range(len(blur_file)):
        img_blur = cv.imread(dir_blur[i])
        img = cv.imread(dir_sharp[i])

        img_blur = img_blur[0:148,0:148]
        img = img[0:148,0:148]

        ## 归一化
        img_norm = img/ (255. / 2.) - 1
        img_blur_norm = img_blur / (255. / 2.) - 1

        img_blur_input = np.expand_dims(img_blur_norm,0)
        img_label = np.expand_dims(img_norm,0)

        output = sess.run(y,feed_dict={x:img_blur_input})
        loss_test = sess.run(loss,feed_dict={y:output,y_:img_label})
        PSNR_test = sess.run(PSNR,feed_dict={y:output,y_:img_label})

        output = (output+1)*(255/2)
        output = np.squeeze(output).astype(np.uint8)
        cv.imwrite('./output/output_face/deblur_0'+str(i+1)+'.png',output,[int(cv.IMWRITE_PNG_COMPRESSION), 0])
        # np.save('./output/deblur_img.npy',output)
        print('loss_test:[%.8f],PSNR_test:[%.8f]' % (loss_test,PSNR_test))

def predict(args):
    savepath = './libSaveNet/savenet/GAN_net300000.ckpt-done'
    path_blur = './data_blur/valid/face'
    blur_file = os.listdir(path_blur)
    blur_file.sort()
    dir_blur = []
    for each in blur_file:
        dir_blur += [os.path.join(path_blur,each)]
    x = tf.placeholder(tf.float32, shape=[1, 1920, 1200, 3])
    y = model.generator2(x,args=args,name='generator')
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=None)
    tf.global_variables_initializer().run()
    saver.restore(sess, savepath)
    for i in range(len(dir_blur)):
        path_load = dir_blur[i]
        img = cv.imread(path_load)
        img = img / (255. / 2.) - 1
        img_shape = np.shape(img)
        if img_shape[0]%4 != 0:
            img = img[:img_shape[0]-img_shape[0]%4]
        if img_shape[1]%4 != 0:
            img = img[:,:img_shape[1]-img_shape[1]%4]
        img_input = np.expand_dims(img,0)

        output = sess.run(y,feed_dict={x:img_input})
        output = (output+1)*(255/2)
        output = np.squeeze(output).astype(np.uint8)
        cv.imwrite('./output/deblur_face_0'+str(i)+'.png',output,[int(cv.IMWRITE_PNG_COMPRESSION), 0])
        # np.save('./output/deblur_face.npy',output)

if __name__ == '__main__':
    GAN_train(args)
    # adtest(args)
    # predict(args)