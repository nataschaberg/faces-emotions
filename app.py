import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image
import io
import time

import tensorflow as tf

# @st.cache_resource
# def load_custom_resnet_model():
#     r = tf.keras.models.load_model(f'custom_model_resnet34_7_classes_ver1', compile=False)
#     return r


@st.cache_data
def load_data():
    data = pd.concat([pd.read_csv('dataset/train_set_a.csv'), pd.read_csv('dataset/train_set_b.csv'), 
        pd.read_csv('dataset/train_set_c.csv'), pd.read_csv('dataset/val_set.csv'), 
        pd.read_csv('dataset/test_set.csv')], ignore_index=True, axis=0)
    return data 


def toast_message(pl, message):
    pl.success(message)
    time.sleep(2)
    pl.empty()


def get_img_channels(img_array):
    c = ([], [], [], [])
    for i in range(img_array.shape[2]):
        t = img_array.copy()
        for j in range(img_array.shape[2]):
            if j != i:
                t[:,:,j] = 0
            if j == 3 and j != i:
                t[:,:,j] = 255
        c[i].append(t)
    return c


EMOTIONS = {
        0: 'anger', 
        1: 'disgust', 
        2: 'fear', 
        3: 'happiness', 
        4: 'sadness',
        5: 'surprise',
        6: 'neutral'
    }

EMOTIONS_NUMS = {key: value for (value, key) in EMOTIONS.items()}

EMOTIONS_EMOJIE = {
    0: '😠',
    1: '🤢',
    2: '😱',
    3: '😃',
    4: '😥',
    5: '😲',
    6: '😐'
}

def get_class_category(num):
    return EMOTIONS[num]


def get_class_num(cat):
    return EMOTIONS_NUMS[cat]


def render_home_section():
    st.header('Image Classification - Facial Expressions')
    st.subheader('Classification of emotions based on facial expressions - underlying data set FER-2013')
    
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    data = load_data()

    st.write(' ')
    st.write(' ')

    col_a, col_b, col_c, col_d, col_e, col_f, col_g = st.columns(7)
    for i, col in enumerate((col_a, col_b, col_c, col_d, col_e, col_f, col_g)):
        filtered = data[data['emotion'] == i]
        im = np.array(filtered.iloc[2]['pixels'].split(' ')).reshape(48, 48).astype('float32')/255. 
        col.image(im, width=100)
        col.markdown(f'**{get_class_category(i).title()}**')

    st.write(' ')
    st.write(' ')
    st.markdown('----')
    st.write(' ')
    st.subheader('Dataset Insights')

    st.markdown(f'- Dataset comprising {data.shape[0]} images')
    st.markdown(f'- Images are labeled with one of seven emotions based on facial expression') 
    st.markdown(f'- Images are grayscale with dimensions 48 x 48') 
    st.write(' ')
    st.write(' ')
    st.write(' ')

    st.markdown('**Labels breakdown in dataset**')
    fig, ax = plt.subplots()
    sns.countplot(y="emotion", data=data, ax=ax, palette='flare')
    ind = np.arange(7)
    width = 0.2 
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(EMOTIONS.values())

    for c in ax.containers:
        labels = [f'{ round(v.get_width()/data.shape[0]*100,2)}%' for v in c]
    ax.bar_label(c, labels=labels, label_type='edge')
    ax.set_xlim([0, 11000])
    st.pyplot(fig)
    

    st.write(' ')
    st.write(' ')
    st.markdown('----')
    st.write(' ')
    st.subheader('Explore Dataset')


    st.sidebar.write(' ')
    st.sidebar.write(' ')
    st.sidebar.write(' ')
    examples = st.sidebar.slider('Show Selection of Images', 3, 36, step=3)
    options = st.sidebar.multiselect(
            'Image Classification',
            ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Surprise', 'Sadness'],
            ['Happiness'],)

    options_nums = [get_class_num(o.lower()) for o in options]

    if options_nums:
        for i in range(0, examples, 3):
            col_1, col_2, col_3 = st.columns(3)
            filtered = data[data['emotion'].isin(options_nums)]
            col_1.image(np.array(filtered.iloc[i]['pixels'].split(' ')).reshape(48, 48).astype('float32')/255., 
                width=150, caption=get_class_category(filtered.iloc[i]['emotion'])) 
            col_2.image(np.array(filtered.iloc[i+1]['pixels'].split(' ')).reshape(48, 48).astype('float32')/255., 
                width=150, caption=get_class_category(filtered.iloc[i+1]['emotion']))
            col_3.image(np.array(filtered.iloc[i+2]['pixels'].split(' ')).reshape(48, 48).astype('float32')/255.,
                width=150, caption=get_class_category(filtered.iloc[i+2]['emotion']))
    else:
        st.write('no class selected to display')

    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    if st.checkbox('Show raw data', False):
        st.subheader('Raw Data')
        
        st.write(data.loc[0:100, :])
    

def render_model_section():
    st.title('Model')
    st.subheader('Model Architecture')
    st.markdown('- ResNet (short for "Residual Network") is a deep neural network architecture developed by Microsoft Research in 2015. \n - It introduced the concept of "skip connections" to solve the problem of vanishing gradients in very deep neural networks. \n - Skip connections allow the network to bypass one or more layers, allowing the gradient to flow more easily and making it easier for the network to learn. \n - ResNet is typically built using residual blocks, which consist of two or more convolutional layers followed by a shortcut connection. The shortcut connection adds the input of the residual block to its output, allowing the network to learn a residual mapping instead of trying to learn the entire mapping from inputs to outputs. \n - ResNet comes in various depths, from 18 layers (ResNet-18) to over 1000 layers (ResNet-1000).', unsafe_allow_html=True)

    st.write(' ')
    st.write(' ') 
    st.markdown('**Example of Stacked Layers**')
    st.image('https://www.researchgate.net/profile/Sajid-Iqbal-13/publication/336642248/figure/fig1/AS:839151377203201@1577080687133/Original-ResNet-18-Architecture.png')
    st.markdown('<sup>[Source](https://www.researchgate.net/figure/Original-ResNet-18-Architecture_fig1_336642248)</sup><br /><br />', unsafe_allow_html=True)

    st.markdown('**Residual Block - Zoom In**')
    st.image('https://media.geeksforgeeks.org/wp-content/uploads/20200424011510/Residual-Block.PNG')
    st.markdown('<sup>[Source: Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)</sup><br /><br />', unsafe_allow_html=True)

    st.markdown('**Different Types of ResNet - Overview**')
    st.image('https://pytorch.org/assets/images/resnet.png')
    st.markdown('<sup>[Source: Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)</sup><br /><br />', unsafe_allow_html=True)


    st.subheader('Model Performance')

    st.markdown('**Benchmark for FER-2013 dataset - Shown for ResNet Models**')
    st.image('https://github.com/nataschaberg/faces-emotions/blob/master/figures/fer-2012-resnet-benchmark.png?raw=true')
 
    st.write(' ')
    st.write(' ') 

    st.markdown('#### My Model Performance so far ...')
    st.markdown('**Loss Progression**')
    col1, col2 = st.columns(2)


    col1.image('https://raw.githubusercontent.com/nataschaberg/faces-emotions/master/figures/7_classes_ver1_10epochs.png')
    col2.image('https://raw.githubusercontent.com/nataschaberg/faces-emotions/master/figures/7_classes_ver1_20epochs.png')


    st.markdown('**Accuracy Progression**')
    col3, col4 = st.columns(2)
    
    col3.image('https://raw.githubusercontent.com/nataschaberg/faces-emotions/master/figures/7_classes_ver1_10epochs_acc.png')
    col4.image('https://raw.githubusercontent.com/nataschaberg/faces-emotions/master/figures/7_classes_ver1_20epochs_acc.png')

    st.markdown('##### loss: 0.9897 - accuracy: 0.6291  - val_loss: 1.0465 - val_accuracy: 0.6076 <br /><br />', unsafe_allow_html=True)

    st.image('https://github.com/nataschaberg/faces-emotions/blob/master/figures/resnet_confusion_matrix.png?raw=true')

    st.subheader('Model Implementation Code - ResNet 34-Layer')
    st.write(' ')
    st.write(' ')
    st.markdown('**Custom Conv2D Layer**  - inherits from Layer <br /> **Main responsibilities**: <br /> 1) takes care of combining convolution layer and <br /> 2) batch normalization', unsafe_allow_html=True)

    code_custom_conv2d_layer = '''
    class CustomConv2D(Layer):
        def __init__(self, n_filters, kernel_size, n_strides, padding='valid'):
            super(CustomConv2D, self).__init__(name='custom_conv2d')
            
            self.conv = Conv2D(filters=n_filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=n_strides,
                            padding=padding)
            
            
            self.batch_norm = BatchNormalization()
            
        def call(self, x):
            x = self.conv(x)
            x = self.batch_norm(x)
            
            return x
    '''
    st.code(code_custom_conv2d_layer, language='python')
    st.write(' ')
    st.markdown('**Residual Block**  - inherits from Layer <br /> **Main responsibilities**: <br /> 1) encompass two or three custom conv2D layers <br /> 2) takes care of skip connection <br /> 3) makes sure that if we have a change in dimensions that this is compatible on ADD step (see `self.inputs.deviate`)', unsafe_allow_html=True)


    code_custom_resblock = '''
    class ResidualBlock(Layer):
        def __init__(self, n_channels, n_strides=1):
            super(ResidualBlock, self).__init__(name='res_block')
                    
            self.inputs_deviate = (n_strides != 1)
            
            self.custom_conv_1 = CustomConv2D(n_channels, 3,
                                              n_strides, padding='same')
            self.custom_conv_2 = CustomConv2D(n_channels, 3, 1, 
                                              padding='same')
            self.activation = Activation('relu')
                    
            if self.inputs_deviate:
                self.custom_conv_3 = CustomConv2D(n_channels, 1,
                                     n_strides) # filter 1x1
                
            
        def call(self, input):
            x = self.custom_conv_1(input)
            x = self.custom_conv_2(x)
            
            if self.inputs_deviate:
                x_add = self.custom_conv_3(input)
                x_add = Add()([x, x_add])
            else:
                x_add = Add()([x, input])
            
            
            return self.activation(x_add)
    '''
    st.code(code_custom_resblock, language='python')


    st.write(' ')
    st.markdown('**Custom ResNet 34**  - inherits from Model <br /> **Main responsibilities**: <br /> 1) wraps all elements in one structure  <br /> 2) outlines the explicit layer sequence for resnet 34', unsafe_allow_html=True)
    code_resnet = '''
    class CustomResNet34(Model):
        def __init__(self, num_classes):
            super(CustomResNet34, self).__init__(name='resnet_34')
            
            # first section from paper
            self.conv_1 = CustomConv2D(64, 7, 2, padding='same')
            self.max_pool = MaxPool2D(3, 2)
            
            # ======= # paper: conv2_x section
            self.conv_2_1 = ResidualBlock(64)
            self.conv_2_2 = ResidualBlock(64)
            self.conv_2_3 = ResidualBlock(64)
            
            # ======= # paper: conv3_x section
            self.conv_3_1 = ResidualBlock(128, 2)
            self.conv_3_2 = ResidualBlock(128)
            self.conv_3_3 = ResidualBlock(128)
            self.conv_3_4 = ResidualBlock(128)
            
            # ======= # paper: conv4_x section
            self.conv_4_1 = ResidualBlock(256, 2)
            self.conv_4_2 = ResidualBlock(256)
            self.conv_4_3 = ResidualBlock(256)
            self.conv_4_4 = ResidualBlock(256)
            self.conv_4_5 = ResidualBlock(256)
            self.conv_4_6 = ResidualBlock(256)
            
            # ======= # paper: conv5_x section
            self.conv_5_1 = ResidualBlock(512, 2)
            self.conv_5_2 = ResidualBlock(512)
            self.conv_5_3 = ResidualBlock(512)
            
            self.global_pool = GlobalAveragePooling2D()
            self.fc_dense = Dense(num_classes, activation='softmax')
            
            
        
        def call(self, x):
            x = self.conv_1(x)
            x = self.max_pool(x)
            
            x = self.conv_2_1(x)
            x = self.conv_2_2(x)
            x = self.conv_2_3(x)
            
            x = self.conv_3_1(x)
            x = self.conv_3_2(x)
            x = self.conv_3_3(x)
            x = self.conv_3_4(x)
            
            x = self.conv_4_1(x)
            x = self.conv_4_2(x)
            x = self.conv_4_3(x) 
            x = self.conv_4_4(x)
            x = self.conv_4_5(x)
            x = self.conv_4_6(x)
            
            x = self.conv_5_1(x)
            x = self.conv_5_2(x)
            x = self.conv_5_3(x)
            
            x = self.global_pool(x)
            x = self.fc_dense(x)
            
            return x
    '''
    st.code(code_resnet, language='python')


def render_try_it_yourself_section():
    uploaded_file = None
    #resnet_model = load_custom_resnet_model()

    st.sidebar.write('ℹ️ Please note:  <br /> uploaded pictures are not saved on our servers!', unsafe_allow_html=True)
    st.header('Now you can try it yourself')

    img_mode = st.radio('Upload Image from disk or sue your webcam?', ('disk', 'webcam'), horizontal=True)
    st.write(' ')
    st.write(' ')

    if img_mode == 'disk':
        img_buffer = st.file_uploader('Upload a photo', type=['jpeg', 'jpg', 'png', 'webp'])
        if img_buffer is not None:
            uploaded_file = Image.open(io.BytesIO(img_buffer.getvalue())) 
        
    if img_mode == 'webcam':
        img_buffer = st.camera_input('Upload a photo')
        if img_buffer is not None:
            original = Image.open(io.BytesIO(img_buffer.getvalue())) 

            # crop webcam picture
            width, height = original.size   # Get dimensions
            left = width/5
            right = 4 * width/5
            uploaded_file = original.crop((left, 0, right, height))

    if uploaded_file is not None: 
        upload_success = st.empty()
        toast_message(upload_success, 'File uploaded successfully!')

        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        
        cola, colb, colc = st.columns([1,1,1])
        cola.write("")
        colb.image(uploaded_file, caption='Original Image', width=200)
        colc.write("")

        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ') 

        arr = np.array(uploaded_file) 

        st.subheader(f'Found Channels: {len(uploaded_file.getbands())}')
        r, g, b, a = get_img_channels(arr)
        col1, col2, col3, col4 = st.columns(4)
        col1.image(r)
        col2.image(g)
        col3.image(b) 
        col4.image(a) 


        st.subheader('Preprocess for Cutom ResNet Model')
        coly, colz = st.columns(2)
        coly.image(uploaded_file.convert('L'), caption=f'Original dimesions {arr.shape[0]} x {arr.shape[1]} pixel')
        colz.image(uploaded_file.convert('L').resize((48, 48)), width=200, caption='Adjusted dimensions 48 x 48 pixel')

        
        # prepped = np.array(uploaded_file.convert('L').resize((48, 48))).reshape(48, 48, 1).astype('float32')
        # resnet_pred = resnet_model.predict(np.array([prepped]))
        
        # st.subheader(f'ResNet Prediction: ')
        # st.write(resnet_pred)
        # st.title(f'{EMOTIONS_EMOJIE[np.argmax(resnet_pred)]} {EMOTIONS[np.argmax(resnet_pred)]} {EMOTIONS_EMOJIE[np.argmax(resnet_pred)]}')
    
        
camera_predict_option = 'Try it yourself!'

with st.sidebar:
    selected = option_menu("", ["Home", 'Model', camera_predict_option], 
        icons=['house', 'graph-up', 'camera'], menu_icon="", default_index=0)


if selected == 'Home':
    render_home_section()

if selected == 'Model':
    render_model_section()

if selected == camera_predict_option:
    render_try_it_yourself_section()

