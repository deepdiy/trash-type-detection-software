import os,rootpath
rootpath.append(pattern='requirements.txt')
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty,DictProperty,NumericProperty,StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.audio import SoundLoader
import cv2,time
import tensorflow as tf
import numpy as np
from ssd_mobilenet_utils import generate_colors,draw_boxes,non_max_suppression

class MainWindow(BoxLayout):
	"""docstring for MainWindow."""

	data=ObjectProperty()
	bundle_dir = rootpath.detect(pattern='.git')
	Builder.load_file(bundle_dir +os.sep+'ui'+os.sep+'main.kv')
	timer=NumericProperty()
	state=NumericProperty()

	def __init__(self):
		super(MainWindow, self).__init__()
		self.PATH_TO_CKPT = 'model/frozen_inference_graph.pb'
		self.SCORE_THRESHOLD = .76
		self.class_names=['background','bottle','trash']
		self.pool=[]

		self.capture = cv2.VideoCapture(1)
		self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
		self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
		self.graph=self.get_graph()
		self.graph.as_default()
		self.sess=tf.Session(graph=self.graph)
		self.bind(timer=lambda x,y:Clock.schedule_once(self.update,0.0001))
		self.bind(state=self.on_state_changed)
		self.update()

	def get_graph(self):
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.FastGFile(self.PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		return detection_graph

	def draw_box(self,image,boxes,scores,classes):
		boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)
		out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)
		self.add_result_to_pool(out_classes)
		# Generate colors for drawing bounding boxes.
		colors = generate_colors(self.class_names)
		# Draw bounding boxes on the image file
		image = draw_boxes(image, out_scores, out_boxes, out_classes, self.class_names, colors)
		return image

	def on_state_changed(self,*arg):
		if self.state==0 or len(self.pool)<20:
			return
		self.play_sound()
		self.save_detection_result()

	# @concurrent.thread
	def play_sound(self,*arg):
		sound = SoundLoader.load(os.sep.join([self.bundle_dir,'audio',str(self.state)+'.wav']))
		if sound:
			sound.play()

	def add_result_to_pool(self,result):
		if len(self.pool)>50:
			self.pool=self.pool[-50:]
		if list(result)==[]:
			result=[0]
		self.pool+=list(result)
		if len(self.pool)>0:
			counts=[self.pool.count(i) for i in range(3)]
			self.state=counts.index(max(counts))
		print(self.state)

	def update(self, *args):
		# display image from cam in opencv window
		ret, frame = self.capture.read()
		self.frame=frame
		buf1=self.detect_trash(frame)
		buf2 = cv2.flip(buf1, 0)
		buf = buf2.tostring()
		texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
		texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
		# display image from the texture
		self.ids.preview.texture = texture1
		self.timer=time.time()

	def detect_trash(self,frame):
		frame_expanded = np.expand_dims(frame, axis=0)
		image_tensor = self.graph.get_tensor_by_name(
			'image_tensor:0')
		boxes = self.graph.get_tensor_by_name('detection_boxes:0')
		scores = self.graph.get_tensor_by_name(
			'detection_scores:0')
		classes = self.graph.get_tensor_by_name(
			'detection_classes:0')
		num_detections = self.graph.get_tensor_by_name(
			'num_detections:0')
		(boxes, scores, classes, num_detections) = self.sess.run(
			[boxes, scores, classes, num_detections],
			feed_dict={image_tensor: frame_expanded})

		self.draw_box(frame,boxes,scores,classes)
		self.boxes=boxes
		self.scores=scores
		self.classes=classes
		return frame

	def save_detection_result(self,*args):
		boxes, scores, classes = np.squeeze(self.boxes), np.squeeze(self.scores), np.squeeze(self.classes).astype(np.int32)
		out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)
		colors = generate_colors(self.class_names)
		image = draw_boxes(self.frame, out_scores, out_boxes, out_classes, self.class_names, colors)
		cv2.imwrite('../dataset/predicted/'+str(time.time())+'.jpg',image)

	def save_img(self,*args):
		cv2.imwrite('../dataset/capture/'+str(time.time())+'.jpg',self.frame)


class Test(App):
	"""docstring for Test."""

	data=ObjectProperty()
	plugins=DictProperty()

	def __init__(self):
		super(Test, self).__init__()

	def build(self):
		demo=MainWindow()
		return demo

if __name__ == '__main__':
	Test().run()
