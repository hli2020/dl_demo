import cv2
import numpy as np
import torch
from network.rtpose_vgg import get_model
from network.post import decode_pose
from evaluate.coco_eval import get_multiplier, get_outputs

weight_name = '../network/weight/pose_model.pth'
model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
# model.cuda()
# model.float()
model.eval()

if __name__ == "__main__":
    
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, oriImg = video_capture.read()
        shape_dst = np.min(oriImg.shape[0:2])

        # Get multiple scales of original image
        multiplier = get_multiplier(oriImg)

        with torch.no_grad():
            paf, heatmap = get_outputs(
                multiplier, oriImg, model,  'rtpose')

        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        canvas, to_plot, candidate, subset = decode_pose(
            oriImg, param, heatmap, paf)

        # Display the resulting frame
        cv2.imshow('Video', to_plot)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
