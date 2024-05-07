import streamlit as st
import pandas as pd

st.set_page_config(page_title='Training Monitoring Dashboard',
                   page_icon='👀',
                   layout='wide')

st.title('Training Monitoring Dashboard')
st.markdown('---')

def ic_page():
    st.header("Image classification training process")
    # st.markdown("![confusion_matrix](image_exas/confusion_matrix.png)")
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

def od_page():
    st.header("Object detection traing process")
    # 读取YAML文件内容
    with open("yolov5\\runs\\train\\exp\\opt.yaml", "r") as f:
        yaml_content = f.read()
    with st.expander("训练参数"):
        # 在Streamlit页面中显示YAML内容
        st.write(yaml_content)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### confusion matrix")
        st.image("yolov5\\runs\\train\\exp\\confusion_matrix.png", width=600)
    with col2:
        st.markdown("#### Precision Recall curve")
        st.image("yolov5\\runs\\train\\exp\\PR_curve.png", width=600)
    st.markdown("#### Train results")
    st.image("yolov5\\runs\\train\\exp\\results.png", width=1500)

    with st.expander("Training set predictions"):
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            st.markdown("##### Train Batch 0")
            st.image("yolov5\\runs\\train\\exp\\train_batch0.jpg", width=600)
        with col_2:
            st.markdown("##### Train Batch 1")
            st.image("yolov5\\runs\\train\\exp\\train_batch1.jpg", width=600)
        with col_3:
            st.markdown("##### Train Batch 2")
            st.image("yolov5\\runs\\train\\exp\\train_batch2.jpg", width=600)

    with st.expander("Validation set predictions"):
        col_4, col_5, col_6 = st.columns(3)
        with col_4:
            st.markdown("##### Validation Batch 0")
            st.image("yolov5\\runs\\train\\exp\\val_batch0_pred.jpg", width=600)
        with col_5:
            st.markdown("##### Validation Batch 1")
            st.image("yolov5\\runs\\train\\exp\\val_batch1_pred.jpg", width=600)
        with col_6:
            st.markdown("##### Validation Batch 2")
            st.image("yolov5\\runs\\train\\exp\\val_batch2_pred.jpg", width=500)

def main():
    tab1, tab2 = st.tabs(["📈 Image classification", "📊 Object detection"])
    with tab1:
        ic_page()
    with tab2:
        od_page()


    # # 添加一个滑块来选择子页面
    # page = st.sidebar.select_slider(
    #     "选择页面",
    #     options=["页面1", "页面2", "页面3"]
    # )
    
    # # 根据滑块值展示不同的子页面
    # if page == "页面1":
    #     page1()
    # elif page == "页面2":
    #     page2()
    # elif page == "页面3":
    #     page3()

if __name__ == "__main__":
    main()