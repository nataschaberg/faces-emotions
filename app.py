import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image
import io
import time


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
    st.subheader('Model')


def render_try_it_yourself_section():
    uploaded_file = None
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


        st.subheader('Preprocess for model')
        coly, colz = st.columns(2)
        coly.image(uploaded_file.convert('L'), caption=f'Original dimesions {arr.shape[0]} x {arr.shape[1]} pixel')
        colz.image(uploaded_file.convert('L').resize((48, 48)), width=200, caption='Adjusted dimensions 48 x 48 pixel')
    

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
