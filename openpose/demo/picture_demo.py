import cv2
import numpy as np
import torch
from network.rtpose_vgg import get_model
from network.post import decode_pose
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat


weight_name = '../network/weight/pose_model.pth'

model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
# model = torch.nn.DataParallel(model).cuda()
# model.float()
model.eval()

test_image = '../readme/ski.jpg'
oriImg = cv2.imread(test_image)     # B,G,R order
shape_dst = np.min(oriImg.shape[0:2])

# Get results of original image
multiplier = get_multiplier(oriImg)

with torch.no_grad():
    orig_paf, orig_heat = get_outputs(
        multiplier, oriImg, model,  'rtpose')
          
    # Get results of flipped image
    swapped_img = oriImg[:, ::-1, :]
    flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                            model, 'rtpose')

    # compute averaged heatmap and paf
    paf, heatmap = handle_paf_and_heat(
        orig_heat, flipped_heat, orig_paf, flipped_paf)
            
param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
canvas, to_plot, candidate, subset = decode_pose(
    oriImg, param, heatmap, paf)
 
cv2.imwrite('result_25_Aug.png',   to_plot)

