import tensorflow as tf
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np

def gaussian_kernel(
        sigma: float,
):
    """Makes 2D gaussian Kernel for convolution. ref = https://www.tensorflow.org/api_docs/python/tf/case """
    w = 2 * int(4.0 * sigma + 0.5) + 1

    d = tf.distributions.Normal(0., sigma)

    size = int(w / 2)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                             vals,
                             vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def get_median(v):
    b = tf.shape(v)
    v = tf.reshape(v, [b[0], -1])
    m = tf.shape(v)[1] // 2
    return tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values, axis=1)

def auto_canny_tf(img, sigma):
    # img is 3channel [ b, h, w, 1 ] and gray scale.
    #

    # step 0 get - parameter
    v = get_median(img)

    lower = tf.clip_by_value(tf.to_float((1.0 - sigma) * v), 0, 255)
    upper = tf.clip_by_value(tf.to_float((1.0 + sigma) * v), 0, 255)
    lower = lower[:, tf.newaxis, tf.newaxis, tf.newaxis]
    upper = upper[:, tf.newaxis, tf.newaxis, tf.newaxis]
    # step 1 Gaussian Filtering
    gauss_kernel = gaussian_kernel(sigma=1.0)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

    [h, w, _, __] = gauss_kernel.shape
    padding_hw = int(int(w) / 2)

    img = tf.pad(img, [[0, 0], [padding_hw, padding_hw], [padding_hw, padding_hw], [0, 0]], mode='SYMMETRIC')
    img = tf.nn.conv2d(img, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")

    # step 2 get Gradient Magnitude
    gradient_kernel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    gradient_kernel_y = tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], tf.float32)

    gradient_kernel_x = gradient_kernel_x[:, :, tf.newaxis, tf.newaxis]
    gradient_kernel_y = gradient_kernel_y[:, :, tf.newaxis, tf.newaxis]

    img = tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')

    gradient_x = tf.nn.conv2d(img, gradient_kernel_x, strides=[1, 1, 1, 1], padding="VALID")
    gradient_y = tf.nn.conv2d(img, gradient_kernel_y, strides=[1, 1, 1, 1], padding="VALID")

    magnitude = tf.sqrt(tf.square(gradient_x) + tf.square(gradient_y))
    theta = tf.atan2(gradient_y, gradient_x)
    thetaQ = (tf.round(theta * (5.0 / np.pi)) + 5) % 5  # Quantize direction
    thetaQ = thetaQ % 4

    gradSup = tf.identity(magnitude)

    E_MATRIX = tf.constant([[0, 0, 0], [0, 0, 1], [0, 0, 0]], tf.float32)
    E_MATRIX = E_MATRIX[:, :, tf.newaxis, tf.newaxis]

    W_MATRIX = tf.constant([[0, 0, 0], [1, 0, 0], [0, 0, 0]], tf.float32)
    W_MATRIX = W_MATRIX[:, :, tf.newaxis, tf.newaxis]

    N_MATRIX = tf.constant([[0, 1, 0], [0, 0, 0], [0, 0, 0]], tf.float32)
    N_MATRIX = N_MATRIX[:, :, tf.newaxis, tf.newaxis]

    S_MATRIX = tf.constant([[0, 0, 0], [0, 0, 0], [0, 1, 0]], tf.float32)
    S_MATRIX = S_MATRIX[:, :, tf.newaxis, tf.newaxis]

    NE_MATRIX = tf.constant([[0, 0, 1], [0, 0, 0], [0, 0, 0]], tf.float32)
    NE_MATRIX = NE_MATRIX[:, :, tf.newaxis, tf.newaxis]

    NW_MATRIX = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, 0]], tf.float32)
    NW_MATRIX = NW_MATRIX[:, :, tf.newaxis, tf.newaxis]

    SE_MATRIX = tf.constant([[0, 0, 0], [0, 0, 0], [0, 0, 1]], tf.float32)
    SE_MATRIX = SE_MATRIX[:, :, tf.newaxis, tf.newaxis]

    SW_MATRIX = tf.constant([[0, 0, 0], [0, 0, 0], [1, 0, 0]], tf.float32)
    SW_MATRIX = SW_MATRIX[:, :, tf.newaxis, tf.newaxis]

    E_VAL = tf.nn.conv2d(gradSup, E_MATRIX, strides=[1, 1, 1, 1], padding="SAME")
    W_VAL = tf.nn.conv2d(gradSup, W_MATRIX, strides=[1, 1, 1, 1], padding="SAME")

    N_VAL = tf.nn.conv2d(gradSup, N_MATRIX, strides=[1, 1, 1, 1], padding="SAME")
    S_VAL = tf.nn.conv2d(gradSup, S_MATRIX, strides=[1, 1, 1, 1], padding="SAME")

    NE_VAL = tf.nn.conv2d(gradSup, NE_MATRIX, strides=[1, 1, 1, 1], padding="SAME")
    SW_VAL = tf.nn.conv2d(gradSup, SW_MATRIX, strides=[1, 1, 1, 1], padding="SAME")

    NW_VAL = tf.nn.conv2d(gradSup, NW_MATRIX, strides=[1, 1, 1, 1], padding="SAME")
    SE_VAL = tf.nn.conv2d(gradSup, SE_MATRIX, strides=[1, 1, 1, 1], padding="SAME")

    NE_SW_LOGIC = tf.logical_or(tf.less_equal(gradSup, NE_VAL), tf.less_equal(gradSup, SW_VAL))
    NW_SE_LOGIC = tf.logical_or(tf.less_equal(gradSup, NW_VAL), tf.less_equal(gradSup, SE_VAL))

    EW_LOGIC = tf.logical_or(tf.less_equal(gradSup, E_VAL), tf.less_equal(gradSup, W_VAL))
    NS_LOGIC = tf.logical_or(tf.less_equal(gradSup, N_VAL), tf.less_equal(gradSup, S_VAL))

    EW_POS = tf.equal(thetaQ, 0)
    EW_ZERO_POSITION = tf.logical_and(EW_POS, EW_LOGIC)
    gradSup = tf.where(EW_ZERO_POSITION, tf.zeros_like(gradSup), gradSup)

    NE_SW_POS = tf.equal(thetaQ, 1)
    NE_SW_ZERO_POSITION = tf.logical_and(NE_SW_POS, NE_SW_LOGIC)
    gradSup = tf.where(NE_SW_ZERO_POSITION, tf.zeros_like(gradSup), gradSup)

    NS_POS = tf.equal(thetaQ, 2)
    NS_ZERO_POSITION = tf.logical_and(NS_POS, NS_LOGIC)
    gradSup = tf.where(NS_ZERO_POSITION, tf.zeros_like(gradSup), gradSup)

    NW_SE_POS = tf.equal(thetaQ, 3)
    NW_SE_ZERO_POSITION = tf.logical_and(NW_SE_POS, NW_SE_LOGIC)
    gradSup = tf.where(NW_SE_ZERO_POSITION, tf.zeros_like(gradSup), gradSup)

    CENTER_MATRIX = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]], tf.float32)
    CENTER_MATRIX = CENTER_MATRIX[:, :, tf.newaxis, tf.newaxis]
    gradSup = tf.nn.conv2d(gradSup, CENTER_MATRIX, strides=[1, 1, 1, 1], padding="VALID")
    gradSup = tf.pad(gradSup, [[0, 0], [1, 1], [1, 1], [0, 0]])

    # step 4 Thresh holding
    strongEdges = gradSup > upper  # highThreshold

    thresholdedEdges = tf.to_float(strongEdges) + tf.to_float(gradSup > lower)

    finalEdges = tf.cast(tf.identity(strongEdges), tf.float32)

    patchMax = tf.nn.max_pool(thresholdedEdges, [1, 3, 3, 1], [1, 1, 1, 1], padding="VALID")
    patchMax = tf.pad(patchMax, [[0, 0], [1, 1], [1, 1], [0, 0]])
    weak_strong_bind = tf.logical_and(tf.equal(patchMax, 2.0), tf.equal(thresholdedEdges, 1.0))
    finalEdges = tf.where(weak_strong_bind, tf.ones_like(finalEdges), finalEdges)

    cond = lambda wk_bj, te, fe: tf.reduce_any(wk_bj) is True

    def body(wk_bj, te, fe):
        currentPixels = tf.to_float(wk_bj)
        targetPixels = tf.nn.max_pool(currentPixels, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        targetPixels = targetPixels > 0
        new_weak_strong_bind = tf.logical_and(tf.logical_and(tf.equal(fe, 0), tf.equal(te, 1.0)), targetPixels)
        fe = tf.where(new_weak_strong_bind, tf.ones_like(fe), fe)
        return [new_weak_strong_bind, te, fe]

    weak_strong_bind, thresholdedEdges, finalEdges = tf.while_loop(cond, body,
                                                                   [weak_strong_bind, thresholdedEdges, finalEdges])


    # Socred Edge
    all_grad_index = tf.cast(gradSup > 0, tf.float32)
    very_weak = tf.abs(all_grad_index - finalEdges)
    gradSup = gradSup * very_weak
    gradSup = gradSup / upper
    finalEdges += gradSup

    return tf.to_float(finalEdges)

if __name__ == '__main__':
    print('made by Jang Hae Woong')
    print('Canny Edge Code Reference: https://github.com/ISI-RCG/spicy/blob/master/apps/canny/original/reference.py')
    print('if you want to original canny edge, remove final 5 line')
    img = misc.imread('test_input.bmp', 'L')
    img = img[np.newaxis,:,:,np.newaxis]
    print(img.shape)
    with tf.Session() as sess:
        plt.imshow(auto_canny_tf(img,0.33).eval()[0,:,:,0])
        plt.show()
