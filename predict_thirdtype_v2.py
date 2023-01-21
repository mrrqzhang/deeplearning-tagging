# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import numpy as np
import metrics
import tensorflow as tf
import time

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score # add by maohui
from sklearn.metrics import classification_report # add by maohui
from map import tag_mapp, get_labels_list # add by maohui

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flags = tf.flags
FLAGS = flags.FLAGS

import random
from sys import path
path.append("/data6/maohui/workspace/sampleProcessing/")
from tagMapping import TagMapper
tagmapper = TagMapper()
Token_List = tagmapper.token_list
Token_Num = len(Token_List)

BERT_BASE_DIR = "/data6/maohui/workspace/bert_base_dir/chinese_L-12_H-768_A-12/"

## Required parameters
flags.DEFINE_string(
    "data_dir", '/data6/maohui/workspace/data/thirdtype_mulity/',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", BERT_BASE_DIR + '/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'wb', "The name of the task to train.")

flags.DEFINE_string("vocab_file", BERT_BASE_DIR + 'vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", '../model_output/thirdtype_mulity',
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", '/Users/chunhua5/Documents/bert_tf/bert/chinese_L-12_H-768_A-12/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 200,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")
flags.DEFINE_integer("keep_checkpoint_max", 15,
                     "save num max.")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file) as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class WeiboProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    
    def get_labels(self,lables_dir):
        """See base class."""
        labels = []
        for line in open(lables_dir):
            tagid = line.strip().split()[2]
            labels.append(tagid)
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[3])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

processors = {"wb":WeiboProcessor,}
class ThirdTypeModel():
  def __init__(self,checkpoint_dir,lables_dir):
    # 一般变量
    self.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    task_name = FLAGS.task_name.lower()
    self.processor = processors[task_name]()
    self.tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    self.is_training=False
    self.use_one_hot_embeddings=False
    self.batch_size=1
    self.gpu_config = tf.ConfigProto()
    self.gpu_config.gpu_options.allow_growth = True
    self.sess=tf.Session(config=self.gpu_config)
    self.model=None
    self.example=None
    self.feature=None
    self.output_weights = None

    # 独特变量
    self.checkpoint_dir=checkpoint_dir
    self.label_list = self.processor.get_labels(lables_dir)
    self.num_labels=len(self.label_list)
    self.index2label={i:self.label_list[i] for i in range(len(self.label_list))}
    #self.id2name = self.processor.getmap(lables_dir)
    self.shell = 0.75

    # 模型
    self.input_ids_p, self.input_mask_p, self.label_ids_p, self.segment_ids_p = None, None, None, None
    #if not os.path.exists(FLAGS.init_checkpoint + "checkpoint"):
    if not os.path.exists(self.checkpoint_dir):
      print (self.checkpoint_dir)
      raise Exception("failed to get checkpoint. going to return ")

    global graph
    graph = tf.get_default_graph()
    with graph.as_default():
      print("going to restore checkpoint")
      #sess.run(tf.global_variables_initializer())
      self.input_ids_p = tf.placeholder(tf.int32, [self.batch_size, FLAGS.max_seq_length], name="input_ids")
      self.input_mask_p = tf.placeholder(tf.int32, [self.batch_size, FLAGS.max_seq_length], name="input_mask")
      self.label_ids_p = tf.placeholder(tf.int32, [self.batch_size], name="label_ids")
      self.segment_ids_p = tf.placeholder(tf.int32, [FLAGS.max_seq_length], name="segment_ids")
      self.probabilities = self.create_model(
          self.bert_config, self.is_training, self.input_ids_p, self.input_mask_p, self.segment_ids_p,
          self.label_ids_p, self.num_labels, self.use_one_hot_embeddings)
      
      self.saver = tf.train.Saver()
      self.saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir))

  def convert_single_example(self, ex_index, example, label_list, max_seq_length,tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i
    #  print('example.text_a: ', example.text_a, example.text_b)
    thirdname = example.text_b

    fields = example.text_a.split(thirdname) 
    tokens_a = []
    for i,phrase in enumerate(fields):
      tokens_a += tokenizer.tokenize(phrase)
      if (i+1)!=len(fields): 
        tokens_a.append('[SEP]')
        tokens_a += tokenizer.tokenize(thirdname)
        tokens_a.append('[SEP]')

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

#    tokens_b = None # add by maohui 20211129
    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]


    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    #label_id = label_map[example.label]
    label_id = -1
    # if ex_index < 5:
    #   '''
    #   tf.logging.info("*** Example ***")
    #   tf.logging.info("guid: %s" % (example.guid))
    #   tf.logging.info("tokens: %s" % " ".join(
    #       [tokenization.printable_text(x) for x in tokens]))
    #   tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #   tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #   tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #   tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    #   '''
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature

  def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
      total_length = len(tokens_a) + len(tokens_b)
      if total_length <= max_length:
        break
      if len(tokens_a) > len(tokens_b):
        tokens_a.pop()
      else:
        tokens_b.pop()

  def create_int_feature(self, values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def create_model(self, bert_config, is_training, input_ids, input_mask, segment_ids,
                  labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire segment. If you want to use the token-level output, use model.get_sequence_output() instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    self.output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

      logits = tf.matmul(output_layer, self.output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      probabilities = tf.nn.softmax(logits, axis=-1)
      return probabilities

  def predict_example(self, example):
    feature = self.convert_single_example(0, example, self.label_list, FLAGS.max_seq_length, self.tokenizer)
    input_ids = np.reshape([feature.input_ids],(1,FLAGS.max_seq_length))
    input_mask = np.reshape([feature.input_mask],(1,FLAGS.max_seq_length))
    segment_ids =  np.reshape([feature.segment_ids],(FLAGS.max_seq_length))
    label_ids =[feature.label_id]

    global graph
    with graph.as_default():
        feed_dict = {self.input_ids_p: input_ids, self.input_mask_p: input_mask, self.segment_ids_p: segment_ids, self.label_ids_p: label_ids}
        possibility = self.sess.run([self.probabilities], feed_dict)
        possibility=possibility[0][0] # get first label
        label_index=np.argmax(possibility)
        tagid=self.index2label[label_index]
        prob = possibility[label_index]
        pred_list = []
        if prob > 0.5:
            pred_list.append(tagid)

    return pred_list

def read_tsv(input_file, quotechar=None):
  """Reads a tab separated value file."""
  with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
          lines.append(line)
      return lines

def PredictFile(predict_file_path, save_path):
  third_model = ThirdTypeModel('/data6/maohui/workspace/model_output/thirdtype_mulity_v2/', '/data6/maohui/workspace/data/thirdtype_mulity/thirdtype.labels')
  fo = open(save_path, 'w')
  lines = read_tsv(predict_file_path)
  for line in lines:
    example= InputExample(guid=0, text_a=line[1],text_b=line[3])
    pred_type = third_model.predict_example(example)
    pred_types = '|'.join(pred_type)
    try:
        fo.write(pred_types + '\t' + '\t'.join(line) + '\n')
    except: continue
  pass

if __name__ == "__main__":
  predict_file_path = FLAGS.data_dir + 'test.tsv'
  save_path = 'result/test.tsv_result'
  PredictFile(predict_file_path, save_path)


