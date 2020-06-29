# DLSR_FinalProject OpenVINO Deployment
## Requirement
- openvino_2019.3.376
- tensorflow==1.12

## Download Yolov3 model
- git clone https://github.com/mystic123/tensorflow-yolo-v3.git

## Frozen and convert darknet weight to tensorflow format
- python3 convert_weights_pb.py

## Convert Yolov3 model to OpenVINO IR format using OpenVINO MO(model optimizer)
### FP32
- python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /usr/src/tensorflow-yolo-v3/frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json --input_shape=[1,416,416,3] --data_type=FP32 -o /usr/src/tensorflow-yolo-v3/FP32/
### FP16
- python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /usr/src/tensorflow-yolo-v3/frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json --input_shape=[1,416,416,3] --data_type=FP16 -o /usr/src/tensorflow-yolo-v3/FP16/

## Inference
### Using CPU
- python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_yolov3_async/object_detection_demo_yolov3_async.py -i /dev/video0 -m ./tensorflow-yolo-v3/FP32/frozen_darknet_yolov3_model.xml --labels ./tensorflow-yolo-v3/unrel.names
### Using GPU
- python3 /opt/intel/openvino/deployment_tools/open_model_zoo/demos/python_demos/object_detection_demo_yolov3_async/object_detection_demo_yolov3_async.py -i /dev/video0 -m ./tensorflow-yolo-v3/FP32/frozen_darknet_yolov3_model.xml -d GPU --labels ./tensorflow-yolo-v3/unrel.names
