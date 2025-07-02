import cv2 
from model import RCNN
from util import preprocess_img, postprocess_tens
import streamlit as st 
import torch

def load_model_from_github():
    model_url = 'https://github.com/Sai-Santhosh/DataScienceProjects/raw/main/black_white_colorization_using_faster_rcnn/colorization_release_v2-9b330a0b.pth'
    model = RCNN(pretrained=False)  # Load the model without pretrained weights
    model.load_state_dict(torch.hub.load_state_dict_from_url(model_url, map_location='cpu', progress=False))
    return model

# Load the model
colorizer_rcnn = load_model_from_github().eval()
use_gpu = False
if use_gpu:
    colorizer_rcnn.cuda()

st.title('Video Colorization')

def main():
    
    up_vdo = st.file_uploader('Upload a Video', type=['mp4'])
    if up_vdo is not None:
        with open('./Videos/input.mp4', 'wb') as f:
            f.write(up_vdo.getvalue())
            
    if st.button('Colorize'):
        cap = cv2.VideoCapture('./Videos/input.mp4')
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 800, 600)

        while True:
            ret, img = cap.read()
            if not ret:
                st.warning('Completed Successfully')
                break
            
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
            
            output_img = postprocess_tens(tens_l_orig, colorizer_rcnn(tens_l_rs).cpu())

            cv2.imshow("Resized_Window", output_img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            
if __name__ == '__main__':
    main()
