import streamlit as st
from image_classification.test import Predict, load_labels_name
from PIL import Image

st.set_page_config(page_title='Predict',
                   page_icon='👻',
                   layout='wide')

label_names = load_labels_name('image_classification/data/cifar-100-python/meta')['fine_label_names']
# 在Streamlit页面中嵌入TensorBoard的IFrame
st.title('Image Classification Prediction')
st.markdown('---')

uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg"])

file_contents = None
if uploaded_file is not None:        
    # 读取文件内容
    file_contents = Image.open(uploaded_file)
    # 显示文件内容
    st.image(file_contents, width=200)

option = st.selectbox(
    'select model',
    ('Resnet-56', 'Resnet-110')
)
if option == 'Resnet-110':
    depth = 110
    checkpoint_path = 'image_classification/save/cifar100-resnet-110/model_best.pth.tar'
elif option == 'Resnet-56':
    depth = 56
    checkpoint_path = 'image_classification/save/cifar100-resnet-56/model_best.pth.tar'

device = st.radio("device", ['cpu', 'cuda'], index=None)
value = st.slider('Top k', 0, 10, 5)

if device is not None and file_contents is not None:
    if st.button('Execute'):
        with st.spinner('Running...'):
            confidence, res = Predict(checkpoint_path=checkpoint_path, depth=depth, test_file=file_contents, device=device, topk=value)
            predict_names = []
            confidence = confidence.tolist()
            for r in res.tolist():
                name = label_names[r]
                predict_names.append(name)
            st.markdown(f'###### Top {value} Results:')
            c1, c2 = st.columns(spec=2)
            with c1:
                for row in predict_names:
                    st.write(row)
            with c2:
                for row in confidence:
                    st.write(round(row, 2))
