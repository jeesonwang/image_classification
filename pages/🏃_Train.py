import os
import sys
import ndraw

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "image_classification"))
from argparse import Namespace

from image_classification.main import main as ic_run
from image_classification.utils import TrainCallback
# from yolov5.train import run as od_run

st.set_page_config(page_title='Train',
                   page_icon='🏃',
                   layout='wide')

st.title('Train Your Model')
st.markdown('---')



def ic_train():
    st.markdown('#### Image Classification Train Configuration')
    
    epochs = st.number_input('epochs', value=164, step=1, key="od_epoch")
    arch = st.selectbox('select model', ('resnet', 'densnet'))
    batch_size = st.number_input('batch_size', value=8, step=1, key="od_batch_size")
    learning_rate = float(st.text_input('learning rate', '0.1'))
    train_data = st.selectbox('train data', ('cifar10', 'cifar100'))
    depth = st.radio("depth", [56, 110], index=None)
    save_path = st.text_input('result save path', 'save/cifar100-resnet-56')
    use_validset = st.radio("use_validset", [True, False], index=0)
    data_aug = st.radio("data_aug", [True, False], index=0)
    optimizer = st.radio("optimizer", ['sgd', 'rmsprop', 'adam'], index=0)

    # Run experiment when user clicks the button
    if st.button('Run Experiment', key="ic_run"):
        if st.sidebar.button('stop'):
            st.stop()
        # python3 main.py --arch resnet --depth 56 --save save/cifar100-resnet-56 --data cifar100 --data_aug
        args = Namespace(save=save_path, resume='', evaluate='', force=False, print_freq=10, tensorboard=True, seed=0, data=train_data, 
                  use_validset=use_validset, data_root='data', num_workers=4, normalized=False, cutout=False, n_holes=1, length=16, data_aug=data_aug, arch=arch, depth=depth, 
                  drop_rate=0.0, death_mode='none', death_rate=0.5, growth_rate=12, bn_size=4, compression=0.5, trainer='train', epochs=epochs, start_epoch=1, patience=0, 
                  batch_size=batch_size, optimizer=optimizer, lr=learning_rate, decay_rate=0.1, momentum=0.9, nesterov=True, alpha=0.99, beta1=0.9, beta2=0.999, weight_decay=0.0001, 
                  config_of_data={'num_classes': 100, 'augmentation': False}, num_classes=100)
        
        ic_run(args, callbacks=TrainCallback())
        st.success('训练结束')

    

def od_train():
    # st.markdown('#### Object Detection Train Configuration')

    if st.button('Run Experiment', key="od_run"):
        # Call your main function with the provided arguments
        # od_run()
        pass


def main():
    # tab1, tab2 = st.tabs(["🏃 Image classification", "🏃 Object detection"])
    # with tab1:
    #     ic_train()
    # with tab2:
    #     od_train()
    tab1 = st.tabs(["🏃 Image classification"])
    ic_train()

if __name__ == "__main__":
    main()