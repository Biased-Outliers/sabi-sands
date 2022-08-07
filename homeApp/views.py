# Django Imports
from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageForm,  ImageFormStyleTransfer

# Numpy and TF imports
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Image imports
import PIL

# Style Transfer Model
hub_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Create your views here.
def hello(request):
    if request.method=='POST':
        
        # Database
        ## Do WE NEED TO SAVE IT TO DATABASE?
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            print("Stylizing form...")
            style_transfer(img_obj)
            img_obj.delete()
            return render(request, 'homeApp/index.html', context={'image_url':'./media/styled_image.jpg'})
        
    form = ImageForm()
    return render(request, 'homeApp/index.html', {
        'form':form
    })

def image_norm(img, max_dim=512):

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape*scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    
    return(img)

def read_image_from_db(path):
    img = tf.io.read_file(path)
    # print(img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def save_image(pil_image, img_path):
    path = "." + img_path
    print(path)
    img = PIL.Image.open(path)
    img.save("styled_image.jpg")

def style_transfer(img_obj):
    
    # Reading in the images from Path
    print(f'Path of images: {"."+img_obj.first_image.url}')
    content_image = read_image_from_db("." + img_obj.first_image.url)
    style_image = read_image_from_db("." + img_obj.second_image.url)

    # Normalizing the images
    content_image = image_norm(content_image)
    style_image = image_norm(style_image)

    # Creating the image with style transfer
    tensor_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    # Converting tensor to image
    pil_image = tensor_to_image(tensor_image)

    # Saving image
    pil_image.save("./media/styled_image.jpg")