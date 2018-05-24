from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import os
import time
import math

# this is the prediction definition
def predict(model, input):
    response = model.predict([input])

    input_im_str = response['outputs'][0]['input']['data']['image']['url']

    pred = response['outputs'][0]['data']['concepts'][0]
    input_label = pred['name']

    true_name = os.path.basename(input.file_obj.name)
    pred_conf = pred['value']
    pred_res = pred_conf > threshold
    show = 'Correct' if pred_res else 'Wrong'
    print('\tinput image {}'.format(true_name))
    print('\tlabel: {}, confidence: {:.5f}, prediction: {}\n'.format(input_label, pred_conf, show))
    return pred_res


# STUDENTS CHANGE HERE
train_folder = 'summer_high_school/train'
test_folder = 'summer_high_school/test'
api_key = 'ad04791d94504bc082ac425d37463efe'
concepts_list = ['house', 'lake']
model_id = 'my_model'
threshold = .3
train_data_cnt = -1  # set to -1 if use all training data; otherwise indicate a positive number here

# API_key
app = ClarifaiApp(api_key=api_key)


# DATA LOADING
# app.inputs.delete_all()
# app.models.delete_all()
avg_num_per_cls = math.floor(train_data_cnt/len(concepts_list)) if train_data_cnt > 0 else 10e10
train_im_list, test_im_list = [], []
train_list, test_list, train_cnt = [], [], 0

# loop for training images
for cls in concepts_list:
    for (_, _, filenames) in os.walk(os.path.join(train_folder, cls)):
        for file in filenames:
            train_cnt += 1
            if train_cnt > avg_num_per_cls:
                break
            curr_im = os.path.join(train_folder, cls, file)
            train_im_list.append(ClImage(file_obj=open(curr_im, 'rb'), concepts=[cls]))

# loop for test images
for cls in concepts_list:
    for (_, _, filenames) in os.walk(os.path.join(test_folder, cls)):
        for file in filenames:
            curr_im = os.path.join(test_folder, cls, file)
            test_im_list.append(ClImage(file_obj=open(curr_im, 'rb'), concepts=[cls]))

print('uploading images to Cloud')
app.inputs.bulk_create_images(train_im_list)


# MODEL
# create the model via API
if not app.models.search(model_id):
    print('create the model ...')
    model = app.models.create(model_id, concepts=concepts_list)

print('fetch the model ...')
model = app.models.get(model_id)
print('train the model ...')
t = time.time()

# this is the core part where the model is trained
model.train()
train_t = time.time() -t
# model.get_info(verbose=True)
# model.get_inputs()
print('Done training! takes {:.4f} seconds ...'.format(train_t))

# PREDICT using the prediction function defined above
cnt, correct_cnt = 0., 0.
print('prediction result:')
for im in test_im_list:
    cnt += 1
    res = predict(model, im)
    if res:
        correct_cnt += 1

print('Done! prediction accuracy is {:.4f}'.format(correct_cnt/cnt))


