import streamlit as st
from functions import predict_step
from itertools import cycle


def image_uploader():
    with st.form("uploader"):
        images = st.file_uploader("Upload Images",accept_multiple_files=True,type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            predicted_captions = predict_step(images,False)
            for i,caption in enumerate(predicted_captions):
                st.write(str(i+1)+'. '+caption)

def images_url():
    with st.form("url"):
        urls = st.text_input('Enter URL of Images followed by comma for multiple URLs')
        images = urls.split(',')
        submitted = st.form_submit_button("Submit")
        if submitted:
            predicted_captions = predict_step(images,True)
            for i,caption in enumerate(predicted_captions):
                st.write(str(i+1)+'. '+caption)

def main():
    st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
    st.title("Image Caption")

    st.subheader("Upload your own Images")
    image_uploader()

    st.subheader("Enter Image URLs")
    images_url()

if __name__ == '__main__':
    main()
