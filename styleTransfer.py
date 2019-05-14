import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf  # Import TensorFlow after Scipy or Scipy will break
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from vgg_config import load_vgg_model
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

def load_img(path_to_img, max_dim = 512):
    """
    purpose: 
        Reshape an input image into dimensions compatible with VGG19 model
    
    details:
        1) Load image
        2) Resize to maximum dimension size
        3) Store image in numpy array
        4) Convert 3D image into 4D via np.expand()
        
    args:
        path_to_img (string)    file location of image
        max_dim     (integer)   maximum dimensions for rescaling large images
        
    rets:
        img         (np array)  rescaled 4D image array
    """
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim/long

    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    return img


def imshow(img, ax, title=None):
    """
    purpose:
        -Remove 4th dimension from image array, plot via
        matplotlib.pyplot.imshow()
        
    args:
        img    (np array)    4D representation of image
        ax     (plt Axes)    Location of plotted image
        title  (string)      Title for plotted image
    """
    out = np.squeeze(img, axis=0).astype('uint8')
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.imshow(out)
    if title:
        ax.set_title(title)    
        
        
def reshape_images(content_image, style_image):
    _, w, h, _ = content_image.shape
    style_image = style_image[:,:w,:h,:] #reshape style
    
    _, w, h, _ = style_image.shape
    content_image = content_image[:,:w,:h,:] #reshape content
    return content_image, style_image        
def load_and_process_img(path_to_img):
    """
    purpose:
        apply vgg19 preprocessing to image
    args:
        path_to_img   (string)   file location of image
    rets:
        img           (np array) 4D representation of image
    """
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    """
    purpose:
        reverse vgg19 preprocessing
    details:
        -add [103.939, 116.779, 123.68] to respective BGR vectors
        -remove 4th dimension from image
    args:
        processed_img (4D np array)
    rets:
        img (3D np array)
    """
    img = processed_img.copy()
    if len(img.shape) == 4:
        img = np.squeeze(img, 0)
    else:
        raise ValueError("Invalid input to deprocessing image")

    img[:, :, 0] += 103.939 # Blue
    img[:, :, 1] += 116.779 # Green
    img[:, :, 2] += 123.68  # Red
    img = img[:, :, ::-1]   # 4D to 3D

    img = np.clip(img, 0, 255).astype('uint8')
    return img

def compute_content_cost(content_activations, generated_activations):
    """
    purpose:
        For a **single layer's** activations, compute the cost between a generated
        image and the original content image
    
    details:
        - As described in arXiv:1508.06576, compute the cost as:
        
                    (1 / 4 * height * width * channels) * 
            sum( (content_activation - generated_activation) ^2 )
 
    args:
        content_activations   (4D array)
            - numpy representation of content image's activations
            
        generated_activations (4D array)
            - numpy representation of generated image's activations
    
    rets:
        Sum of squared errors, multiplied by (1/4 * height * width * channels)
    """
    
    m, height, width, channels = generated_activations.get_shape().as_list()
    
    return (1 / 4 * height * width * channels) *  tf.reduce_sum(tf.square(content_activations - generated_activations))


def gram_matrix(A):
    """
    args:
        A: - matrix of shape (n_c, n_H*n_W)
    rets:
        Gram Matrix of A, shape (n_C, n_C)
    """
    return tf.matmul(A, tf.transpose(A))

# GRADED FUNCTION: compute_layer_style_cost

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S,(n_H*n_W,n_C)))
    a_G = tf.transpose(tf.reshape(a_G,(n_H*n_W,n_C)))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = 1 /(4*n_H*n_W*n_C*n_H*n_W*n_C)*(tf.reduce_sum(tf.square(tf.subtract(GS,GG))))
    
    ### END CODE HERE ###
    
    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS, sess):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0
    coeff = 1 / len(STYLE_LAYERS)

    for layer_name in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def compute_total_cost(content_loss, style_loss, alpha=10, beta=40):
    """
    purpose:
        Compute the average style loss for a given model, across given layers

    details:
        - For each layer in style_layers(), gather the activations from 
        
    args:
    
    rets:
       
    """
    total_loss = alpha*content_loss + beta*style_loss
    return total_loss

def add_noise_to_image(image, ratio):
    """
    purpose:
        add ratio % noise to an image
    details:
        - create a random numpy array with same dimensions as input image
        - add (noise * ratio) + (image * 1-ratio)
    args:
        image (np array) original image
        ratio (float)    decimal representation of % noise to add
    rets:
        (np array) image with noise added
    
    """
    noise = np.random.uniform(-20, 20, (image.shape)).astype('float32')
    
    return (noise * ratio) + (image * (1 - ratio))

def model_nn(sess,
             content_path,
             style_path,
             model_path,
             name,
             contentLayer='conv4_2',
             styleLayers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
             noise_ratio=0.6,       
             num_iterations = 200):
    #sess = tf.InteractiveSession()
    
    # Load images, resize, generate noisy image
    content_image = load_img(content_path).astype('uint8')
    style_image = load_img(style_path).astype('uint8')
    content_image, style_image = reshape_images(content_image, style_image)
    noisy_image = add_noise_to_image(content_image, ratio=noise_ratio)
    
    # load vgg model with specified dimensions, compute initial cost function
    vgg = load_vgg_model(model_path, content_image.shape)
    sess.run(vgg['input'].assign(content_image))

    content_activation = sess.run(vgg[contentLayer])
    generated_activation = vgg[contentLayer] # placeholder for now. will be updated later
    content_cost = compute_content_cost(content_activation, generated_activation)

    sess.run(vgg['input'].assign(style_image))
    style_cost = compute_style_cost(vgg, styleLayers, sess)

    total_cost = compute_total_cost(content_cost, style_cost,
                                    alpha=10, beta=40)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(total_cost)
    

    # initialize global variables, run the noisy image, loop through epochs
    sess.run(tf.global_variables_initializer())
    sess.run(vgg['input'].assign(noisy_image))

    for i in range(num_iterations):

        sess.run(train_step)
        generated_image = sess.run(vgg['input'])
       # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([total_cost, content_cost, style_cost])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            path = 'output/{0}_{1}.png'.format(name, i)
            scipy.misc.imsave(path, deprocess_img(generated_image))

    # save last generated image
    path = 'output/{0}_final.png'.format(name, i)
    scipy.misc.imsave(path, deprocess_img(generated_image))

    return generated_image