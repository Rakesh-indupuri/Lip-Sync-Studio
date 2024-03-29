# Import all of the dependencies
import streamlit as st
import os 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

 
st.set_page_config(layout='wide')
with st.sidebar: 
    st.image("https://cdn.pixabay.com/photo/2013/07/12/18/17/equalizer-153212_1280.png")
    st.title('LipSync Studio')
    st.info('This is made by using Deep Learning with help of CNN and RNN algorithms.')
    st.info('©️ Rakesh Indupuri and Pavan kumar Paidi.')

st.title('LipSync Studio') 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Upload a video', options)
print("selected video is :"+selected_video)
col1, col2 = st.columns(2)

if options: 
    with col1: 
        st.info('Uploaded video:')
        file_path = os.path.join('..','data','s1', selected_video)
        conversion_command = f'ffmpeg -i {file_path} -vcodec libx264 sample.mp4 -y'
        conversion_result = os.system(conversion_command)

        if conversion_result == 0:
            st.video('sample.mp4')
        else:
            st.error('Video conversion failed. Check the FFmpeg command and try again.')
            video = open('test_video.mp4', 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)
    with col2: 
        model = load_model()
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.info('Text spoken by person is:')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.markdown("Text generated from the video is : "+f'<span style="color: red;">{converted_prediction}</span>', unsafe_allow_html=True)