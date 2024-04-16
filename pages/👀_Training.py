import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title='Training',
                   page_icon='👀',
                   layout='wide')

st.title('Training Monitoring Dashboard')
st.markdown('---')

option = st.selectbox(
    'select model',
    ('Resnet-56', 'Resnet-110')
)
if option == 'Resnet-110':
    s_path = 'image_classification/save/cifar100-resnet-110/scores.tsv'
elif option == 'Resnet-56':
    s_path = 'image_classification/save/cifar100-resnet-56/scores.tsv'

# 读取TSV文件
df = pd.read_csv(s_path, sep='\t', encoding='utf-8')
df_epoch = df['epoch']
df_loss = df[['train_loss', 'val_loss']]
df_lr = df['lr']
df_err = df[['train_err1', 'val_err1', 'train_err5', 'val_err5']]


st.markdown('#### epoch')
st.line_chart(df_epoch)

st.markdown('#### learning rate')
st.line_chart(df_lr)

st.markdown('#### loss')
st.line_chart(df_loss)

st.markdown('#### error rate')
st.line_chart(df_err)