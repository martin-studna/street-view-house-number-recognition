import tensorflow as tf
import tensorflow_addons as tfa
from svhn_dataset import SVHN
import cv2
import numpy as np
import bboxes_utils

def draw_bboxes(img, bboxes):
    import cv2
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    im = np.array(img)
    for i in range(len(bboxes)):
        cv2.rectangle(im, (int(bboxes[i][1]), int(bboxes[i][0])), (int(
            bboxes[i][3]), int(bboxes[i][2])), (255, 0, 0), 3)
    cv2.imshow("image", im)
    cv2.waitKey(0)

anchors = []

a1 = [0.15, 0.2]
a2 = [0.2, 0.3]
a3 = [0.3, 0.6]
a4 = [0.4, 0.75]
a5 = [0.5, 0.85]

anchors_count = 5
for y in np.linspace(0, 1, 14):
    for x in np.linspace(0, 1, 14):
        anchors.append((y-a1[1]/2, x-a1[0]/2, y+a1[1]/2, x+a1[0]/2))
        anchors.append((y-a2[1]/2, x-a2[0]/2, y+a2[1]/2, x+a2[0]/2))
        anchors.append((y-a3[1]/2, x-a3[0]/2, y+a3[1]/2, x+a3[0]/2))
        anchors.append((y-a4[1]/2, x-a4[0]/2, y+a4[1]/2, x+a4[0]/2))
        anchors.append((y-a5[1]/2, x-a5[0]/2, y+a5[1]/2, x+a5[0]/2))

anchors = np.array(anchors)



model = tf.keras.models.load_model("model.h5", custom_objects={
                                   "SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy()})


image = cv2.imread("test.png")
image = cv2.resize(image, (224, 224))
# image = cv2.GaussianBlur(image, (3,3), 0)
# image = cv2.Canny(image, 30, 150)
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

classes, bboxes = model.predict(np.array([image]))
scores = classes[0, :, 1:].max(axis=1)
classes = classes[0, :, 1:].argmax(axis=1)

bboxes = bboxes_utils.bboxes_from_fast_rcnn(
    anchors, bboxes[0]) * 224

selected_indices = tf.image.non_max_suppression(
    bboxes, scores, 5, iou_threshold=0.2, score_threshold=0.2).numpy()
selected_boxes = bboxes[selected_indices]
selected_predicted_classes = classes[selected_indices]


print(selected_predicted_classes)

draw_bboxes(image, selected_boxes)



# svhn = SVHN()

# test = svhn.test.map(lambda data: (
#     data['image'], data['bboxes'], data['classes'])).take(-1)
# test = list(test.as_numpy_iterator())[:20]

# X_test_multiples = []
# X_test = []

# for image, _, _ in test:
#     height, width, _ = image.shape
#     multiple_h = 224 / height
#     multiple_w = 224 / width
#     X_test.append(cv2.resize(image, (224, 224)))
#     X_test_multiples.append([multiple_h, multiple_w])
# X_test = np.array(X_test)

# anchors = []

# a1 = [0.15, 0.2]
# a2 = [0.2, 0.3]
# a3 = [0.3, 0.6]
# a4 = [0.4, 0.75]
# a5 = [0.5, 0.85]

# anchors_count = 5
# for y in np.linspace(0, 1, 14):
#     for x in np.linspace(0, 1, 14):
#         anchors.append((y-a1[1]/2, x-a1[0]/2, y+a1[1]/2, x+a1[0]/2))
#         anchors.append((y-a2[1]/2, x-a2[0]/2, y+a2[1]/2, x+a2[0]/2))
#         anchors.append((y-a3[1]/2, x-a3[0]/2, y+a3[1]/2, x+a3[0]/2))
#         anchors.append((y-a4[1]/2, x-a4[0]/2, y+a4[1]/2, x+a4[0]/2))
#         anchors.append((y-a5[1]/2, x-a5[0]/2, y+a5[1]/2, x+a5[0]/2))

# anchors = np.array(anchors)




# with open(f"svhn_competition.txt", "w", encoding="utf-8") as predictions_file:
#     MAX_ROI = 5
#     IOU_THRESHOLD = 0.2
#     SCORE_THRESHOLD = 0.2

#     predicted_classes, predicted_bboxes = model.predict(
#         X_test, batch_size=32)

#     for i in range(len(predicted_classes)):
#         scores = predicted_classes[i, :, 1:].max(axis=1)
#         classes = predicted_classes[i, :, 1:].argmax(axis=1)
#         bboxes = predicted_bboxes[i]

#         bboxes = bboxes_utils.bboxes_from_fast_rcnn(
#             anchors, bboxes) * 224

#         selected_indices = tf.image.non_max_suppression(
#             bboxes, scores, MAX_ROI, iou_threshold=IOU_THRESHOLD, score_threshold=SCORE_THRESHOLD).numpy()
#         selected_boxes = bboxes[selected_indices]
#         selected_predicted_classes = classes[selected_indices]

#         orig_selected_boxes = np.array(selected_boxes)
#         # height
#         orig_selected_boxes[:, 0] /= X_test_multiples[i][0]
#         # height
#         orig_selected_boxes[:, 2] /= X_test_multiples[i][0]
#         # width
#         orig_selected_boxes[:, 1] /= X_test_multiples[i][1]
#         # width
#         orig_selected_boxes[:, 3] /= X_test_multiples[i][1]

#         print(selected_predicted_classes + 1)
#         draw_bboxes(X_test[i], selected_boxes)


#         output = ""
#         for label, bbox in zip(selected_predicted_classes, orig_selected_boxes):
#             output += str(label+1) + " " + str(int(bbox[0])) + " " + str(
#                 int(bbox[1])) + " " + str(int(bbox[2])) + " " + str(int(bbox[3])) + " "
#         print(*output, file=predictions_file, sep='')


# print()
