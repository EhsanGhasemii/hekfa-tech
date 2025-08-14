# import cv2
# import insightface
# from insightface.app import FaceAnalysis
# 
# # Initialize the face analysis model
# face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# face_app.prepare(ctx_id=0)  # Use GPU (CUDA)
# 
# # Load image
# img = cv2.imread('persons.jpg')  # Replace with your image filename
# if img is None:
#     raise FileNotFoundError("Image not found!")
# 
# # Detect faces
# faces = face_app.get(img)
# 
# # Draw results on image
# for face in faces:
#     x1, y1, x2, y2 = map(int, face.bbox)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# 
#     # Draw facial landmarks
#     for (x, y) in face.kps:
#         cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)
# 
# # Save result image
# output_path = 'output.jpg'
# cv2.imwrite(output_path, img)
# print(f"Output image saved as: {output_path}")
# 
# 
import cv2
import onnxruntime
import onnx
import numpy as np
from skimage import transform as trans

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio

    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    print('lmk: ', lmk)
    print('dst: ', dst)
#     print('dst: ', dst)
#     print('tform: ', tform)
    M = tform.params[0:2, :]
    print('M: ', M)
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped



def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def nms(dets, nms_thresh):
    thresh = nms_thresh
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep





# print
print('=' * 50)
print('start coding without api')

# Load image
img = cv2.imread('persons.jpg')  # Replace with your image filename


# define paths of the models
recognition_model_path = "/root/.insightface/models/buffalo_l/w600k_r50.onnx"
genderage_model_path = "/root/.insightface/models/buffalo_l/genderage.onnx"
detection_model_path = "/root/.insightface/models/buffalo_l/det_10g.onnx"
landmark_2d_106_model_path = "/root/.insightface/models/buffalo_l/2d106det.onnx"
landmark_3d_68_model_path = "/root/.insightface/models/buffalo_l/1k3d68.onnx"

# start session
session = onnxruntime.InferenceSession(detection_model_path, None)

inputs = session.get_inputs()
input_cfg = inputs[0]
print('input_cfg: ', input_cfg)
input_shape = input_cfg.shape
print('input_shape: ', input_shape)
outputs = session.get_outputs()
print('outputs: ', outputs)


# initialize the model
taskname = 'detection'
center_cache = {}
nms_thresh = 0.4
dket_thresh = 0.5

input_cfg = session.get_inputs()[0]
input_shape = input_cfg.shape
if isinstance(input_shape[2], str):
    input_size = None
else:
    input_size = tuple(input_shape[2:4][::-1])
#print('image_size:', self.image_size)
input_name = input_cfg.name
output_names = []
for o in outputs:
    output_names.append(o.name)
input_mean = 127.5
input_std = 128.0
use_kps = False
_anchor_ratio = 1.0
_num_anchors = 1
if len(outputs)==6:
    fmc = 3
    _feat_stride_fpn = [8, 16, 32]
    _num_anchors = 2
elif len(outputs)==9:
    fmc = 3
    _feat_stride_fpn = [8, 16, 32]
    _num_anchors = 2
    use_kps = True
elif len(outputs)==10:
    fmc = 5
    _feat_stride_fpn = [8, 16, 32, 64, 128]
    _num_anchors = 1
elif len(outputs)==15:
    fmc = 5
    _feat_stride_fpn = [8, 16, 32, 64, 128]
    _num_anchors = 1
    use_kps = True


# prepare the detection model
ctx_id = 0
det_thresh = 0.5
det_size = (640, 640)
input_size = det_size
max_num = 0





# detection_model.detect
im_ratio = float(img.shape[0]) / img.shape[1]
model_ratio = float(input_size[1]) / input_size[0]
if im_ratio>model_ratio:
    new_height = input_size[1]
    new_width = int(new_height / im_ratio)
else:
    new_width = input_size[0]
    new_height = int(new_width * im_ratio)
det_scale = float(new_height) / img.shape[0]
resized_img = cv2.resize(img, (new_width, new_height))
det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
det_img[:new_height, :new_width, :] = resized_img


print('@@ img shape: ', img.shape)
print('@@ input_size shape: ', input_size)
print('@@ im_ratio: ', im_ratio)
print('@@ model_ratio: ', model_ratio)
print('@@ new_height: ', new_height)
print('@@ new_width: ', new_width)


# detection_model.forward
scores_list = []
bboxes_list = []
kpss_list = []
input_size = tuple(det_img.shape[0:2][::-1])
blob = cv2.dnn.blobFromImage(det_img, 1.0/input_std, input_size, (input_mean, input_mean, input_mean), swapRB=True)
net_outs = session.run(output_names, {input_name : blob})


input_height = blob.shape[2]
input_width = blob.shape[3]


print('_feat_stride_fpn: ', _feat_stride_fpn)
print('fmc: ', fmc)
for idx, stride in enumerate(_feat_stride_fpn):
    print('idx: ', idx)
    print('stride: ', stride)
    scores = net_outs[idx]
    bbox_preds = net_outs[idx+fmc]
    bbox_preds = bbox_preds * stride
    if use_kps:
        kps_preds = net_outs[idx+fmc*2] * stride


    height = input_height // stride
    width = input_width // stride
    K = height * width
    key = (height, width, stride)

    print('height: ', height)
    print('input_height: ', input_height)
    print('width: ', width)
    print('input_width: ', input_width)
    print('K: ', K)
    print('key: ', key)
    print('det_thresh: ', det_thresh)
    print('_num_anchors: ', _num_anchors)
    if key in center_cache:
        anchor_centers = center_cache[key]
    else:
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

        anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
        if _num_anchors>1:
            anchor_centers = np.stack([anchor_centers]*_num_anchors, axis=1).reshape( (-1,2) )
        if len(center_cache)<100:
            center_cache[key] = anchor_centers

    pos_inds = np.where(scores>=det_thresh)[0]
    print('pos_inds: ', pos_inds)
    print('anchor_centers: ', anchor_centers.shape)
    print('bbox_preds: ', bbox_preds.shape)
    bboxes = distance2bbox(anchor_centers, bbox_preds)
    pos_scores = scores[pos_inds]
    pos_bboxes = bboxes[pos_inds]
    scores_list.append(pos_scores)
    bboxes_list.append(pos_bboxes)

    if use_kps:
        kpss = distance2kps(anchor_centers, kps_preds)
        kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
        pos_kpss = kpss[pos_inds]
        kpss_list.append(pos_kpss)


    print('=' * 40)

# return scores_list, bboxes_list, kpss_list


# # scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

scores = np.vstack(scores_list)
scores_ravel = scores.ravel()
order = scores_ravel.argsort()[::-1]
# print('bboxes_list: ', bboxes_list)
bboxes = np.vstack(bboxes_list) / det_scale
if use_kps:
    kpss = np.vstack(kpss_list) / det_scale
pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
pre_det = pre_det[order, :]
keep = nms(pre_det, nms_thresh)
det = pre_det[keep, :]




if use_kps:
    kpss = kpss[order,:,:]
    kpss = kpss[keep,:,:]
else:
    kpss = None
if max_num > 0 and det.shape[0] > max_num:
    area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                            det[:, 1])
    img_center = img.shape[0] // 2, img.shape[1] // 2
    offsets = np.vstack([
        (det[:, 0] + det[:, 2]) / 2 - img_center[1],
        (det[:, 1] + det[:, 3]) / 2 - img_center[0]
    ])
    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
    if metric=='max':
        values = area
    else:
        values = area - offset_dist_squared * 2.0  # some extra weight on the centering
    bindex = np.argsort(
        values)[::-1]  # some extra weight on the centering
    bindex = bindex[0:max_num]
    det = det[bindex, :]
    if kpss is not None:
        kpss = kpss[bindex, :]


# print('keep: ', keep)
# print('det: ', det)
# print('kpss: ', kpss)
# return det, kpss




# Arc-face Init

recognition_model_path = "/root/.insightface/models/buffalo_l/w600k_r50.onnx"
model_file = recognition_model_path
taskname = 'recognition'
find_sub = False
find_mul = False
model = onnx.load(model_file)
graph = model.graph
for nid, node in enumerate(graph.node[:8]):
    #print(nid, node.name)
    if node.name.startswith('Sub') or node.name.startswith('_minus'):
        find_sub = True
    if node.name.startswith('Mul') or node.name.startswith('_mul'):
        find_mul = True
if find_sub and find_mul:
    #mxnet arcface model
    input_mean = 0.0
    input_std = 1.0
else:
    input_mean = 127.5
    input_std = 127.5

find_sub = False
find_mul = False
input_mean = 127.5
input_mean = 127.5
session = onnxruntime.InferenceSession(model_file, None)
input_cfg = session.get_inputs()[0]
input_shape = input_cfg.shape
input_name = input_cfg.name
input_size = tuple(input_shape[2:4][::-1])
outputs = session.get_outputs()
output_names = []
for out in outputs:
    output_names.append(out.name)
output_shape = outputs[0].shape


# Arc-face prepare
if ctx_id<0:
    session.set_providers(['CPUExecutionProvider'])

ret = []
print('det: ', det)
for i in range(det.shape[0]):
    bbox = det[i, 0:4]
    det_score = det[i, 4]
    kps = None
    if kpss is not None:
        kps = kpss[i]

    face = {'bbox':bbox, 'kps':kps, 'det_score':det_score}

    # Arc-face get
    aimg = norm_crop(img, landmark=face['kps'], image_size=input_size[0])
#     face.embedding = self.get_feat(aimg).flatten()
#     return face.embedding

    # Arc-face get-feat
    if not isinstance(aimg, list):
        aimg = [aimg]
    
    blob = cv2.dnn.blobFromImages(aimg, 1.0 / input_std, input_size,
                                  (input_mean, input_mean, input_mean), swapRB=True)
    face['embedding'] = session.run(output_names, {input_name: blob})[0]

    print('embedding: ', face['embedding'])
    print('embedding: ', face['embedding'].shape)



    ret.append(face)


    print('-' * 50)



