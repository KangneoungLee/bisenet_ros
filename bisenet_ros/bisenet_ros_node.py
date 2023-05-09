# ROS 2 for bisenet  
# Author:
# - Kangneoung Lee
  

import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import CameraInfo, Image, CompressedImage, PointCloud2, PointField # 
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import message_filters

import os
import math
import time

import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F

from bisenet_ros.lib import transform

from bisenet_ros.nets.bisenetv1 import BiSeNetV1
from bisenet_ros.nets.bisenetv2 import BiSeNetV2

model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,
}

def colorize(gray_in, palette):
    # gray: numpy array of the label and 1*3N size list palette
    #print("type check", type(gray_in))
    color = PILImage.fromarray(gray_in.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    # convert P mode to RGB mode 010622
    color = color.convert('RGB')
    return color


class BiseNetROS(Node):
  """
  Create an BiseNetROS class
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """

    super().__init__('bisenet_ros')
    
    self.declare_parameter('depth_caminfo_topic_name', 'camera/depth/camera_info')
    self.declare_parameter('rgb_image_topic_name', 'camera/color/image_raw')
    self.declare_parameter('depth_image_topic_name', 'camera/depth/image_rect_raw')
    self.declare_parameter('aligned_depth_topic_name', 'camera/aligned_depth_to_color/image_raw')
    self.declare_parameter('result_topic_name', 'net/result')
    self.declare_parameter('cost_image_topic_name', 'net/cost_image')
    self.declare_parameter('sync_depth_topic_name', 'net/sync_depth/image_rect_raw')
    self.declare_parameter('cost_pcl_topic_name', 'net/pcl/cost')
    self.declare_parameter('tf_prefix', '')
    self.declare_parameter('pcl_frame', 'camera_depth_optical_frame')
    
      
    depth_caminfo_topic_name = self.get_parameter('depth_caminfo_topic_name').get_parameter_value().string_value
    rgb_image_topic_name = self.get_parameter('rgb_image_topic_name').get_parameter_value().string_value
    depth_image_topic_name = self.get_parameter('depth_image_topic_name').get_parameter_value().string_value
    aligned_depth_topic_name = self.get_parameter('aligned_depth_topic_name').get_parameter_value().string_value
    result_topic_name = self.get_parameter('result_topic_name').get_parameter_value().string_value
    cost_image_topic_name = self.get_parameter('cost_image_topic_name').get_parameter_value().string_value
    sync_depth_topic_name = self.get_parameter('sync_depth_topic_name').get_parameter_value().string_value 
    cost_pcl_topic_name = self.get_parameter('cost_pcl_topic_name').get_parameter_value().string_value
    tf_prefix = self.get_parameter('tf_prefix').get_parameter_value().string_value
    pcl_frame = self.get_parameter('pcl_frame').get_parameter_value().string_value
    self.pcl_frame = pcl_frame
    
    if not tf_prefix == "":
      self.pcl_frame = tf_prefix + "/" + pcl_frame
                      
    # Create the publisher. This publisher will publish an Image
    # to the video_frames topic. The queue size is 10 messages.
    self.result_publisher_ = self.create_publisher(Image, result_topic_name, 10)
    self.cost_img_publisher_ = self.create_publisher(Image, cost_image_topic_name, 10)
    self.sync_depth_publisher_ = self.create_publisher(Image, sync_depth_topic_name, 10)
    self.cost_pcl_publisher_ = self.create_publisher(PointCloud2, cost_pcl_topic_name, 10)
    
    #self.subscription = self.create_subscription(Image, 'camera/color/image_raw', self.listener_callback, 1)
    self.depthcam_info_sub = self.create_subscription(CameraInfo, depth_caminfo_topic_name, self.camera_info_callback, 1)
    subscription_1 = message_filters.Subscriber(self, Image, rgb_image_topic_name)
    subscription_2 = message_filters.Subscriber(self, Image, depth_image_topic_name)
    self.ats = message_filters.ApproximateTimeSynchronizer([subscription_1, subscription_2],queue_size = 2,slop = 0.1)
    self.ats.registerCallback(self.color_depth_callback)
    
    self.time_stamp = self.get_clock().now().to_msg()
    
    self.declare_parameter('manual_camera_info', False)
    self.declare_parameter('depth_scale', 0.001)
    self.declare_parameter('focal_len_x', 385)
    self.declare_parameter('focal_len_y', 385)
    self.declare_parameter('principle_x', 320)
    self.declare_parameter('principle_y', 240)
    
    manual_camera_info = self.get_parameter('manual_camera_info').value
    depth_scale = self.get_parameter('depth_scale').value
    focal_len_x = self.get_parameter('focal_len_x').value
    focal_len_y = self.get_parameter('focal_len_y').value
    principle_x = self.get_parameter('principle_x').value
    principle_y = self.get_parameter('principle_y').value
        
    self.manual_camera_info = manual_camera_info 
    
    self.depth_scale = depth_scale
    self.fx = focal_len_x
    self.fy = focal_len_y
    self.px = principle_x
    self.py = principle_y
    
    self.declare_parameter('unit_test_enable', False)
    self.declare_parameter('class_num', 25)
    self.declare_parameter('img_height_net_input', 192)
    self.declare_parameter('img_width_net_input', 256)
    self.declare_parameter('gpu_available', False)
    self.declare_parameter('is_jit_model_cpu', True)
    self.declare_parameter('num_of_gpu', 2)
    self.declare_parameter('jit_script_model_path', '/home/artlab/ros2_ws/src/bisenet_ros/bisenet_ros/trained_model/bisenetv2_jit_calimg_1500.pt')
    self.declare_parameter('model_path', '/home/artlab/ros2_ws/src/bisenet_ros/bisenet_ros/trained_model/bisenetv2_train_epoch_200.pth')
    self.declare_parameter('colors_path', '/home/artlab/ros2_ws/src/bisenet_ros/bisenet_ros/label/rugd/rugd_colors.txt')
    self.declare_parameter('names_path', '/home/artlab/ros2_ws/src/bisenet_ros/bisenet_ros/label/rugd/rugd_names.txt')
    self.declare_parameter('cost_low', 5)
    self.declare_parameter('cost_med', 128)
    self.declare_parameter('cost_high', 255)
    
    
    unit_test_enable = self.get_parameter('unit_test_enable').value           
    class_num = self.get_parameter('class_num').value
    test_h = self.get_parameter('img_height_net_input').value
    test_w = self.get_parameter('img_width_net_input').value
    gpu_available = self.get_parameter('gpu_available').value
    is_jit_model_cpu = self.get_parameter('is_jit_model_cpu').value
    num_of_gpu = self.get_parameter('num_of_gpu').value
    jit_script_model_path = self.get_parameter('jit_script_model_path').get_parameter_value().string_value
    model_path = self.get_parameter('model_path').get_parameter_value().string_value
    colors_path = self.get_parameter('colors_path').get_parameter_value().string_value
    names_path = self.get_parameter('names_path').get_parameter_value().string_value
    cost_low = self.get_parameter('cost_low').value
    cost_med = self.get_parameter('cost_med').value
    cost_high = self.get_parameter('cost_high').value
    
    self.unit_test_enable = unit_test_enable
    self.class_num = class_num
    self.test_h = test_h
    self.test_w = test_w
    self.gpu_available = gpu_available
    self.is_jit_model_cpu = is_jit_model_cpu
    self.num_of_gpu = num_of_gpu
    self.jit_script_model_path = jit_script_model_path
    self.model_path = model_path
    self.colors_path = colors_path
    self.names_path = names_path
    self.cost_low = cost_low
    self.cost_med = cost_med
    self.cost_high = cost_high
    
    self.pcl_gen_stride1 = 6
    self.pcl_gen_stride2 = 15
    self.pcl_gen_stride3 = 25

    self.pcl_gen_th1 = 240
    self.pcl_gen_th2 = 320
    self.pcl_gen_th3 = 360
    
    self.pcl_gen_depth_th = 6 
    
    if self.gpu_available == True:
      self.is_jit_model_cpu = False
    
    self.colors = np.loadtxt(self.colors_path).astype('uint8')
    self.label_name = np.loadtxt(self.names_path, dtype=str)
    
    terrain_type_low =['void', 'dirt','sand','grass','asphalt','gravel','mulch','rock_bed','log']
    terrain_type_med =['water','rock']
    self.cost_map = np.zeros((self.class_num),dtype=int)
    
    for k in range(0,len(self.label_name)):
      if self.label_name[k] in terrain_type_low:
        self.cost_map[k] = self.cost_low
      elif self.label_name[k] in terrain_type_med:
        self.cost_map[k] = self.cost_med
      else:
        self.cost_map[k] = self.cost_high
    
    self.get_logger().info('net input image size, height : %d width : %d'%(self.test_h, self.test_w))
    self.get_logger().info('****Terrain type low****')
    print(terrain_type_low)
    self.get_logger().info('****Terrain type med****')
    print(terrain_type_med)
    
    value_scale = 255
    self.mean = [0.3257, 0.3690, 0.3223]
    self.mean = [item * value_scale for item in self.mean]
    self.std = [0.2112, 0.2148, 0.2115]
    self.std = [item * value_scale for item in self.std]
    
    self.prediction = np.zeros((1,1,self.class_num),dtype=float)
    
    
    self.declare_parameter('model_type', 'bisenetv2')
    model_type = self.get_parameter('model_type').get_parameter_value().string_value
    self.model_type = model_type
    
    self.model = None
    
    if self.gpu_available == True:
      self.model = model_factory[self.model_type](self.class_num)
      self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
      self.model.cuda()
      cudnn.benchmark = True
      
    elif self.is_jit_model_cpu == True:
      torch.backends.quantized.engine = 'qnnpack'
      self.model = torch.jit.load(self.jit_script_model_path, map_location='cpu')
      self.model.qconfig = torch.quantization.get_default_qconfig('qnnpack') 
    
    else: # self.is_jit_model_cpu = False and self.gpu_available = False
      self.model = model_factory[self.model_type](self.class_num)
      self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
    
    self.model.aux_mode = 'eval'
    self.model.eval()
    
    timer_period = 0.5  # seconds
      
    # Create the timer
    self.timer = self.create_timer(timer_period, self.netloop_callback)

    # Used to convert between ROS and OpenCV images
    self.cv_bridge = CvBridge()
    
    self.rgb_image = None
    self.depth_image = None
    
    self.img_receive_flag = False
    
    self.get_logger().info('Initialization done')
    
    self.checkparam()
    
  def checkparam(self):
    if (self.test_h % 64 != 0):
      self.get_logger().error("remainder of test_h divided by 64 should be 0")

    if (self.test_w % 64 != 0):
      self.get_logger().error("remainder of test_w divided by 64 should be 0")

  def unit_test(self):
    image_path ='/home/artlab/ros2_ws/src/bisenet_ros/bisenet_ros/creek_00156.png'
    self.rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
    
    self.depth_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    self.img_receive_flag = True
      
  def color_depth_callback(self, rgb_image_msg, depth_image_msg):
    """
    Callback function.
    """
    if self.unit_test_enable == True:
      self.get_logger().info("Unit test is enabled, if you want to get image topic, disable unit test")
      return
    if self.img_receive_flag == True:
      return
 
    # Convert ROS Image message to OpenCV image
    self.rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_image_msg)
    self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
    
    self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth_image_msg)
    # Display image
    #cv2.imshow("camera", self.rgb_image)
    
    #cv2.waitKey(1000)   
    
    #cv2.imwrite("/home/artlab/ros2_ws/src/bisenet_ros/rgb.png",self.rgb_image)
    #cv2.imwrite("/home/artlab/ros2_ws/src/bisenet_ros/depth.png",self.depth_image)
    
    h, w, c = self.rgb_image.shape
    self.time_stamp = self.get_clock().now().to_msg()
    
    self.img_receive_flag = True
    self.get_logger().info("iamge shape height : %d, width : %d, channel : %d"% (h,w,c))
    
  def camera_info_callback(self,msg):
    
    if self.manual_camera_info == False:
      self.depth_scale = 0.001;
      self.fx = msg.k[0];
      self.fy = msg.k[4];
      self.px = msg.k[2];
      self.py = msg.k[5];  
        
    #self.get_logger().info("depth_scale : %f, fx : %f, fy : %f, px : %f, py : %f"% (self.depth_scale, self.fx, self.fy, self.px, self.py))  
  def netloop_callback(self):
    """
    Callback function.
    """
    if self.unit_test_enable == True:
      self.unit_test()
    
    if self.img_receive_flag == False:
      return
    
    end = time.time()
    
    image = self.rgb_image.copy()
    org_h, org_w, _ = image.shape
    
    input = torch.from_numpy(image.transpose((2, 0, 1))).float() #change the order or dim H*W*C --> C*H*W
    
    for t, m, s in zip(input, self.mean, self.std): #normalize process. subtract by mean and divided by std
      t.sub_(m).div_(s)
      
    input = input.unsqueeze(0)  # add batch dimension C*H*W --> N*C*H*W
    im_resized = F.interpolate(input, size=(self.test_h, self.test_w), mode='bilinear', align_corners=True)
    if self.gpu_available == True:
      im_resized = im_resized.cuda()
    
    logits = None
    if self.gpu_available == False and self.is_jit_model_cpu == True:
      logits = self.model(im_resized)  # only jit script has a little different output argument. This is because, a little modification is required to generate jit script model
    else:
      logits = self.model(im_resized)[0]
    logits = F.interpolate(logits, size=(org_h, org_w), mode='bilinear', align_corners=True)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    preds = preds.squeeze(0)
    
    gray = np.uint8(preds.data.cpu().numpy())
    color = colorize(gray, self.colors)
    color = np.array(color, dtype = np.uint8)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    #h, w, c = color.shape
    #self.get_logger().info('debug shape color h : %d  w : %d  c :  %d'%(h, w, c))
    #cv2.imwrite('/home/artlab/ros2_ws/src/bisenet_ros/bisenet_ros/result_creek_00156.png', color)
    
    cost_map_function = lambda x: self.cost_map[x]
    cost = cost_map_function(gray)
    cost = np.uint8(cost)
    
    reslt_msg = self.cv_bridge.cv2_to_imgmsg(color)
    reslt_msg.header.stamp = self.time_stamp
    reslt_msg.height = org_h
    reslt_msg.width = org_w
    self.result_publisher_.publish(reslt_msg)
    
    cost_msg = self.cv_bridge.cv2_to_imgmsg(cost)
    cost_msg.header.stamp = self.time_stamp
    cost_msg.height = org_h
    cost_msg.width = org_w
    self.cost_img_publisher_.publish(cost_msg)
    
    depth_msg = self.cv_bridge.cv2_to_imgmsg(self.depth_image)
    depth_msg.header.stamp = self.time_stamp
    depth_msg.height = org_h
    depth_msg.width = org_w
    self.sync_depth_publisher_.publish(cost_msg)
    
    self.pcl_pub_(cost, self.depth_image)
    
    self.get_logger().info('Inference time : %f'%(time.time() - end))
    self.img_receive_flag = False
    # Display the message on the console
    self.get_logger().info('Processing done')

  def pcl_pub_(self, cost_img, depth_img):
    depth_img_res = None
    if depth_img.dtype == 'uint16':
      depth_img_res = np.float32(depth_img)
      depth_img_res = depth_img_res*self.depth_scale
    elif depth_img.dtype == 'float32':
      depth_img_res = depth_img.copy
    else:
      self.get_logger().error("type of depth image should be uint16 or float32")
      return
    
    print("depth image shape", depth_img_res.shape)
    h, w = depth_img_res.shape
    
    pcl_data = [] #declare empty list for pcl data
    
    h_stride = self.pcl_gen_stride1
    
    for h_step in range(self.pcl_gen_th1, h, h_stride):  
    
      if h_step >= self.pcl_gen_th3:
        h_stride = self.pcl_gen_stride3
      elif h_step >= self.pcl_gen_th2:
        h_stride = self.pcl_gen_stride2
      
      for w_step in range(0, w):
        d_val = depth_img_res[h_step, w_step]
        cost_val = cost_img[h_step, w_step]
        cost_val = float(cost_val)
        
        if d_val > self.pcl_gen_depth_th or d_val < 0.01 or cost_val == 0:
          continue
        
        pcl_x = (w_step - self.px)*d_val/self.fx
        pcl_y = (h_step - self.py)*d_val/self.fy
        pcl_z = d_val
        
        pcl_data.append([pcl_x, pcl_y, pcl_z, cost_val])
        #pcl_data.append(pcl_x)
        #pcl_data.append(pcl_y)
        #pcl_data.append(pcl_z)
        #pcl_data.append(cost_val)
    
    #print("pcl_data list length : ", len(pcl_data))    
    pcl_data = np.array(pcl_data, dtype='float32')
    #print("pcl_data np shape : ", pcl_data.shape, "pcl_data np type : ", pcl_data.dtype) 
    flatten_pcl_data = pcl_data.astype(np.float32).tobytes()
    #print("flatten_pcl_data length : ", len(flatten_pcl_data))
    pcl_dtype = PointField.FLOAT32
    elemsize = np.dtype(np.float32).itemsize # return the number of bytes for a single element
    
    fields = [PointField(
            name =n, offset = i * elemsize, datatype = pcl_dtype, count=1)
            for i,n in enumerate('xyzi')]
    
    pcl_msg = PointCloud2()
    pcl_msg.header.frame_id = self.pcl_frame
    pcl_msg.header.stamp = self.time_stamp
    pcl_msg.height = 1
    pcl_msg.width = pcl_data.shape[0]
    pcl_msg.is_dense = False
    pcl_msg.is_bigendian = False
    pcl_msg.fields = fields
    pcl_msg.point_step = (elemsize*4)
    pcl_msg.row_step = (elemsize*4*pcl_data.shape[0])
    pcl_msg.data = flatten_pcl_data
    self.cost_pcl_publisher_.publish(pcl_msg)
    
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  bisenet_ros = BiseNetROS()
  
  # Spin the node so the callback function is called.
  rclpy.spin(bisenet_ros)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  bisenet_ros.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
