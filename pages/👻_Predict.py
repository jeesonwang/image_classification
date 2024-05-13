import streamlit as st
from PIL import Image
import sys
import os
import tempfile
from pathlib import Path
import pathlib
import time
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "yolov5//"))

from image_classification.test import Predict, load_labels_name
from yolov5.detect import run

st.set_page_config(page_title='Predict',
                   page_icon='👻',
                   layout='wide')

label_names = load_labels_name('image_classification/data/cifar-100-python/meta')['fine_label_names']
# 在Streamlit页面中嵌入TensorBoard的IFrame
st.title('Image Classification Prediction')
st.markdown('---')

# 文件载入
uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg"])
file_contents = None
folder_path = None
file_path = None
if uploaded_file:
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    # 读取文件内容
    file_contents = Image.open(uploaded_file)
    st.image(file_contents, width=500)

if file_contents is None:
    folder_path = st.text_input('Enter the folder path', 'yolov5/data/images')

device = st.radio("device", ['cpu', 'cuda'], index=None)

option = st.selectbox(
    'select model',
    ('yolov5', 'Resnet-56', 'Resnet-110')
)
if option.startswith('Resnet'):
    if option == 'Resnet-110':
        depth = 110
        checkpoint_path = 'image_classification/save/cifar100-resnet-110/model_best.pth.tar'
    elif option == 'Resnet-56':
        depth = 56
        checkpoint_path = 'image_classification/save/cifar100-resnet-56/model_best.pth.tar'

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
else:
    conf_thres = st.slider('Confidence threshold', 0.0, 1.0, 0.25)  # confidence threshold
    if device is not None:
        if file_path or folder_path:
            if st.button('Execute'):
                with st.spinner('Running...'):
                    checkpoint_path = Path('yolov5\\runs\\train\\exp\\weights\\best.pt')
                    if device == 'cuda':
                        device = "0"
                    
                    if file_path is not None:
                        path = file_path
                    elif folder_path is not None:
                        path = folder_path
                    else:
                        st.warning("🚨")
                    
                    imgs, speed_info, save_info = run(weights=checkpoint_path, source=path, conf_thres=conf_thres, device=device, exist_ok=True)
                    st.info(speed_info)
                    if save_info:
                        st.info(save_info)
                # st.image(img, width=500, channels="BGR")
                if file_path:
                    col1, col2 = st.columns(2)
                    # 在每列中显示一张图片
                    with col1:
                        st.image(file_contents, channels="BGR",width=600, caption="Original image")
                    with col2:
                        placeholder_image = st.empty()
                        with placeholder_image:
                            for img in imgs:
                                st.image(img, channels="BGR", width=600, caption="Detected Images")
                                time.sleep(1.5)
                elif folder_path:
                    st.info(f"There are a total of {len(imgs)} images in the folder.")
                    placeholder_image = st.empty()
                    while True:
                        for img in imgs:
                            with placeholder_image:
                                st.image(img, channels="BGR", width=500, caption="Detected Images")
                                time.sleep(1.75)