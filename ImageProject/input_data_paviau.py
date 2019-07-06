from __future__ import print_function
from __future__ import division
from numpy import *
import numpy as np
import numpy
import h5py
def dense_to_one_hot(labels_dense, num_classes=9):   #这个函数里的num_classes也是类别
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):

  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      ##assert images.shape[3] == 1
      assert images.shape[3] == 1
#      images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2]*images.shape[3])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      #images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(103)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
class SemiDataSet(object):         #数据分配的重点函数
    def __init__(self, images, labels, n_labeled):
#        self.n_labeled = n_labeled
        self.n_labeled = 100                        #an zhao bai fen bi xuan qu shu ju zhe ge di fang gai le
        # Unlabled DataSet
        self.unlabeled_ds = DataSet(images, labels)

        # Labeled DataSet
        self.num_examples = self.unlabeled_ds.num_examples
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)
        images = images[shuffled_indices]
        labels = labels[shuffled_indices]
        y = numpy.array([numpy.arange(9)[l==1][0] for l in labels])   #这个地方的10是类别个数
        idx = indices[y==0][:5]
        n_classes = y.max() + 1
#        n_from_each_class = int(n_labeled / n_classes)
        n_from_each_class = n_labeled
        i_labeled = []
        for c in range(n_classes):
            percent=int(numpy.shape(indices[y==c])[0]*n_from_each_class)  #an zhao bai fen bi xuan qu shu ju zhe ge di fang gai le
            i = indices[y==c][:percent]
#            i = indices[y==c][:n_from_each_class]
            i_labeled += list(i)
        l_images = images[i_labeled]
        l_labels = labels[i_labeled]
        self.labeled_ds = DataSet(l_images, l_labels)

    def next_batch(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
        images = numpy.vstack([labeled_images, unlabeled_images])
        return images, labels
def read_data_sets(n_labeled = 100, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    VALIDATION_SIZE=0
#    paviau_data=r'/home/zj/Documents/ladder-master/paviau_percent80.mat' 
#    paviau_data=loadmat(paviau_data)
#    train_images=paviau_data.get('train_images')
#    train_labels=paviau_data.get('train_labels')
#    test_images=paviau_data.get('test_images')
#    test_labels=paviau_data.get('test_labels')
#    train_images=train_images.reshape(34218,1,103,1)
#    test_images=test_images.reshape(8558,1,103,1)
#    train_labels=train_labels.reshape(34218,)
#    test_labels=test_labels.reshape(8558,)
    cifar_data=h5py.File('./Data/ladder_paviau103_per20_imdb.mat', 'r')
    train_images=cifar_data['train_images'][:]
    train_labels=cifar_data['train_labels'][:]
    test_images=cifar_data['test_images'][:]
    test_labels=cifar_data['test_labels'][:]
    #train_images=train_images.reshape(400000,28,28,4)
    #test_images=test_images.reshape(100000,28,28,4)
    
#    train_images=train_images.reshape(17106,1,103,1)
#    test_images=test_images.reshape(25670,1,103,1)
#    train_labels=train_labels.reshape(17106,)
#    test_labels=test_labels.reshape(25670,)
   
#    train_images=train_images.reshape(34218,1,103,1)
#    test_images=test_images.reshape(8558,1,103,1)
#    train_labels=train_labels.reshape(34218,)
#    test_labels=test_labels.reshape(8558,)

    train_images=train_images.reshape(8551,1,103,1)
    test_images=test_images.reshape(34225,1,103,1)
    train_labels=train_labels.reshape(8551,)
    test_labels=test_labels.reshape(34225,)
   
    train_images=train_images.astype(np.int32)
    test_images=test_images.astype(np.int32)
    train_labels=train_labels.astype(np.int32) 
    test_labels=test_labels.astype(np.int32)

    train_labels=dense_to_one_hot(train_labels)
    test_labels=dense_to_one_hot(test_labels)
  
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.train = SemiDataSet(train_images, train_labels, n_labeled)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)

    return data_sets
#if __name__=="__main__":
#    cifar=read_data_sets(n_labeled=0.1, one_hot=True)
