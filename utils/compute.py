import tensorflow as tf

def compute_psnr(ref, target,convert=None):
    if convert:
        ref = tf.image.convert_image_dtype(ref, dtype=tf.uint8, saturate=True)
        target = tf.image.convert_image_dtype(target, dtype=tf.uint8, saturate=True)
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    # diff = target - ref
    # sqr = tf.multiply(diff, diff)
    # err = tf.reduce_sum(sqr)
    # v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    # mse = err / tf.cast(v, tf.float32)
    mse = tf.reduce_mean(tf.square(ref - target))
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))
    return psnr


