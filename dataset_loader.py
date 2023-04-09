import os
import collections
import random
import tensorflow.compat.v1 as tf
import util


def data_loader(
		data_path,
		epochs,
		batch_size,
		training=True,
		load_estimated_rot=False):
  """Load stereo image datasets.
	
  Args:
    data_path: (string)
    epochs: (int) the number of training epochs.
    batch_size: (int) batch size.
    training: (bool) set it True when training to enable illumination randomization
    	for input images.
    load_estimated_rot: (bool) set it True when training DirectionNet-T to load
    	estimated rotation from DirectionNet-R saved as 'rotation_pred' on disk.

  Returns:
    Tensorflow Dataset
  """

  def load_data(path):
    "Load files saved as pickle."
    img_id, rotation = tf.py_func(util.read_pickle, 
      [path + '/rotation_gt.pickle'], [tf.string, tf.float32])
    _, translation = tf.py_func(util.read_pickle, 
      [path + '/epipoles_gt.pickle'], [tf.string, tf.float32])
    _, fov = tf.py_func(util.read_pickle, 
      [path + '/fov.pickle'], [tf.string, tf.float32])

    if load_estimated_rot:
      _, rotation_pred = tf.py_func(util.read_pickle, 
        [path + '/rotation_pred.pickle'], [tf.string, tf.float32])
    else:
      rotation_pred = tf.zeros_like(rotation)

    img_path = path + '/' + img_id
    return tf.data.Dataset.from_tensor_slices(
      (img_id, img_path, rotation, translation, fov, rotation_pred))

  def load_images(img_id, img_path, rotation, translation, fov, rotation_pred):
    """Load images and decode text lines."""

  	def load_single_image(img_path):
  	  image = tf.image.decode_png(tf.read_file(img_path))
	  image = tf.image.convert_image_dtype(image, tf.float32)
	  image.set_shape([512, 512, 3])
	  image = tf.squeeze(
	  	tf.image.resize_area(tf.expand_dims(image, 0), [256, 256]))
	  return image

  	input_pair = collections.namedtuple(
  		'data_input',
  		[
		  'id',
		  'src_image',
		  'trt_image',
		  'rotation',
		  'translation',
		  'fov',
		  'rotation_pred'
		 ])
  	src_image = load_single_image(img_path+'.src.perspective.png')
        trt_image = load_single_image(img_path+'.trt.perspective.png')
	random_gamma = random.uniform(0.7, 1.2)
	if training:
	  src_image = tf.image.adjust_gamma(src_image, random_gamma)
          trt_image = tf.image.adjust_gamma(trt_image, random_gamma)

  	rotation = tf.reshape(
  		tf.stack([tf.decode_csv(rotation, [0.0] * 9)], 0), [3, 3])
  	rotation.set_shape([3, 3])

  	translation = tf.reshape(
  		tf.stack([tf.decode_csv(translation, [0.0] * 3)], 0), [3])
  	translation.set_shape([3])

  	fov = tf.reshape(tf.stack([tf.decode_csv(fov, [0.0])], 0), [1])
  	fov.set_shape([1])

  	if load_estimated_rot:
  	  rotation_pred = tf.reshape(
  	  	tf.stack([tf.decode_csv(rotation_pred, [0.0] * 9)], 0), [3, 3])
	  rotation_pred.set_shape([3, 3])

	return input_pair(img_id, src_image, trt_image, rotation, translation, fov, rotation_pred)

  ds = tf.data.Dataset.list_files(os.path.join(data_path, '*'))
  ds = ds.flat_map(load_data)
  ds = ds.map(load_images, num_parallel_calls=50).apply(
  	tf.data.experimental.ignore_errors()).repeat(epochs)
  ds = ds.batch(batch_size, drop_remainder=True).prefetch(10)
  return ds

  
