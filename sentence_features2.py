# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling_dw
import optimization
import tokenization
import tensorflow as tf
import codecs

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"data_dir", None,
	"The input data dir. Should contain the .tsv files (or other data files) "
	"for the task.")

flags.DEFINE_string(
	"bert_config_file", None,
	"The config json file corresponding to the pre-trained BERT model. "
	"This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
					"The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
	"output_dir", None,
	"The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
	"init_checkpoint", None,
	"Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
	"do_lower_case", True,
	"Whether to lower case the input text. Should be True for uncased "
	"models and False for cased models.")

flags.DEFINE_integer(
	"max_seq_length", 128,
	"The maximum total input sequence length after WordPiece tokenization. "
	"Sequences longer than this will be truncated, and sequences shorter "
	"than this will be padded.")

flags.DEFINE_integer(
	"exinfos_length", 0,
	"External feature vector dimensions. "
)

flags.DEFINE_integer(
	"num_exinfos_hidden_layers", 4,
	"external info linear layer number"
)

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
	"do_predict", False,
	"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

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

flags.DEFINE_integer("iterations_per_loop", 1000,
			"How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("tag_embedding", False, "Whether to generate tag embedding.")

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

flags.DEFINE_integer(
	"num_variant_tags", 10,
	"maximal number of tags every sentence.")


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
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
		self.text_c = text_c
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

	def __init__(self,
				input_ids_l,
				input_ids_r,
				input_mask_l,
				input_mask_r,
				segment_ids,
				ex_infos,
				label_id,
				is_real_example=True):
		self.input_ids_l = input_ids_l
		self.input_ids_r = input_ids_r
		self.input_mask_l = input_mask_l
		self.input_mask_r = input_mask_r
		self.segment_ids = segment_ids
		if ex_infos:
			self.ex_infos = ex_infos
		self.label_id = label_id
		self.is_real_example = is_real_example


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
		with tf.gfile.Open(input_file, "rb") as f:
			reader = csv.reader(codecs.iterdecode(f, 'utf-8', errors='ignore'), delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines


#==WB_binary_a_Processor
class WBTrashProcessor(DataProcessor):
	"""Processor for the Weibo data set ."""

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
		if  FLAGS.tag_embedding:
			return self._create_examples(
				self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
		else:
			return self._create_examples(self._read_tsv(os.path.join(data_dir, "midtext.tsv")), "test")
	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		FLAGS.exinfos_length = 0
		FLAGS.num_variant_tags = 2
		for (i, line) in enumerate(lines):
			#if i == 0:
			#	continue
			guid = "%s-%s" % (set_type, i)
			text_a = tokenization.convert_to_unicode(line[1])
			text_b = None
			text_c = None
			if set_type == "test":
				label = ['0','0']
			else:
				yes = tokenization.convert_to_unicode(line[-1])
				no = '1' if yes=='0' else '0' 
				label = [yes, no]
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
		return examples


class Wb_Dual_Tower_Processor(DataProcessor):
	"""Processor for the Weibo data set ."""

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
		if  FLAGS.tag_embedding:
			return self._create_examples(
				self._read_tsv(os.path.join(data_dir, "thirdtag.tsv")), "test")
		else:
			return self._create_examples(
				self._read_tsv(os.path.join(data_dir, "midtext.tsv")), "test")

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		FLAGS.exinfos_length = 0
		for (i, line) in enumerate(lines):
			#if i == 0:
			#	continue
			guid = "%s-%s" % (set_type, i)
			text_a = tokenization.convert_to_unicode(line[1])
			text_b = None
			text_c = None
			if set_type == "test":
				if FLAGS.tag_embedding:
					text_b = tokenization.convert_to_unicode(line[2])
				else: text_b='0'
				label = '0'
			else:
				text_b = '0'
				label = '0'
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
		return examples

class WbmzProcessor(DataProcessor):
	"""Processor for the Weibo data set ."""

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

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, i)
			text_a = tokenization.convert_to_unicode(line[1])
			text_b = tokenization.convert_to_unicode(line[2])
			if set_type == "test":
				label = [0 for i in range(FLAGS.num_variant_tags)]
			else:
				label = tokenization.convert_to_unicode(line[-1]).split(',')
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=None, label=label))
		return examples


class WideProcessor(DataProcessor):
	"""Processor for the Weibo data set ."""

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

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, i)
			text_a = tokenization.convert_to_unicode(line[1])
			text_b = tokenization.convert_to_unicode(line[2])
			text_c = tokenization.convert_to_unicode(line[3])
			if line[3] == "":
				text_c = ','.join(['0'] * FLAGS.exinfos_length)
			if set_type == "test":
				label = [0 for i in range(FLAGS.num_variant_tags)]
			else:
				label = tokenization.convert_to_unicode(line[-1]).split(',')
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
		return examples


class ExinfosProcessor(DataProcessor):
	"""Processor for the Weibo data set ."""

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

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, i)
			text_a = tokenization.convert_to_unicode(line[5])
			text_b = tokenization.convert_to_unicode(line[2]).split('@')[0] + ' ' + tokenization.convert_to_unicode(
				line[1])
			text_c = tokenization.convert_to_unicode(line[4])
			if set_type == "test":
				label = [0 for i in range(FLAGS.num_variant_tags)]
			else:
				label = tokenization.convert_to_unicode(line[-1]).split(',')
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
		return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
	"""Converts a single `InputExample` into a single `InputFeatures`."""

	if isinstance(example, PaddingInputExample):
		return InputFeatures(
			input_ids_l=[0] * max_seq_length,
			input_ids_r=[0] * max_seq_length,
			input_mask_l=[0] * max_seq_length,
			input_mask_r=[0] * max_seq_length,
			segment_ids=[0] * max_seq_length,
			ex_infos=[0] * FLAGS.exinfos_length,
			label_id=0,
			is_real_example=False)

	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i


	#  tokens_a = tokenizer.tokenize(example.text_a)

	tokens_a = tokenizer.tokenize(example.text_a)
	tokens_b = tokenizer.tokenize(example.text_b)
	tokens_a.insert(0,'[CLS]')
	tokens_b.insert(0,'[CLS]')
	tokens_a = tokens_a[0:max_seq_length]  #only max_seq_length used if length>max_seq_length
	tokens_b = tokens_b[0:max_seq_length]
	input_ids_l = tokenizer.convert_tokens_to_ids(tokens_a)
	input_ids_r = tokenizer.convert_tokens_to_ids(tokens_b)
	input_mask_l=[1] * len(input_ids_l)
	input_mask_r=[1] * len(input_ids_r)
	while len(input_ids_l) < max_seq_length:
		input_ids_l.append(0)
		input_mask_l.append(0)
	while len(input_ids_r) < max_seq_length:
		input_ids_r.append(0)
		input_mask_r.append(0)
	
	segment_ids=[0] * max_seq_length
		
	ex_infos = []
	

	label_id = label_map[example.label]

	assert len(input_ids_l) == max_seq_length
	assert len(input_mask_l) == max_seq_length
	assert len(input_ids_r) == max_seq_length
	assert len(input_mask_r) == max_seq_length
	assert len(segment_ids) == max_seq_length

	if ex_index < 20:
		tf.logging.info("*** Example ***")
		tf.logging.info("guid: %s" % (example.guid))
		tf.logging.info("tokens_a: %s" % " ".join(
			[tokenization.printable_text(x) for x in tokens_a]))

		tf.logging.info("tokens_b: %s" % " ".join(
			[tokenization.printable_text(x) for x in tokens_b]))

		tf.logging.info("input_ids_l: %s" % " ".join([str(x) for x in input_ids_l]))
		tf.logging.info("input_ids_r: %s" % " ".join([str(x) for x in input_ids_r]))
		tf.logging.info("input_mask_l: %s" % " ".join([str(x) for x in input_mask_l]))
		tf.logging.info("input_mask_r: %s" % " ".join([str(x) for x in input_mask_r]))
		tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
		tf.logging.info("label: %s " % (example.label))

	feature = InputFeatures(
		input_ids_l=input_ids_l,
		input_ids_r=input_ids_r,
		input_mask_l=input_mask_l,
		input_mask_r=input_mask_r,
		segment_ids=segment_ids,
		ex_infos=ex_infos,
		label_id=label_id,
		is_real_example=True)
	return feature


def file_based_convert_examples_to_features(
		examples, label_list, max_seq_length, tokenizer, output_file):
	"""Convert a set of `InputExample`s to a TFRecord file."""

	writer = tf.python_io.TFRecordWriter(output_file)

	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		feature = convert_single_example(ex_index, example, label_list,
						max_seq_length, tokenizer)

		def create_int_feature(values):
			f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
			return f

		def create_float_feature(values):
			f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
			return f

		features = collections.OrderedDict()
		features["input_ids_l"] = create_int_feature(feature.input_ids_l)
		features["input_ids_r"] = create_int_feature(feature.input_ids_r)
		features["input_mask_l"] = create_int_feature(feature.input_mask_l)
		features["input_mask_r"] = create_int_feature(feature.input_mask_r)
		features["segment_ids"] =  create_int_feature(feature.segment_ids) 
		if FLAGS.exinfos_length != 0:
			features["ex_infos"] = create_float_feature(feature.ex_infos)
		features["is_real_example"] = create_int_feature(
			[int(feature.is_real_example)])
		if isinstance(feature.label_id, list):
			label_ids = feature.label_id
		else:
			label_ids = feature.label_id
		features["label_ids"] = create_int_feature([label_ids])

		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		writer.write(tf_example.SerializeToString())
	writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
								drop_remainder):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""

	name_to_features = {
		"input_ids_l": tf.FixedLenFeature([seq_length], tf.int64),
		"input_ids_r": tf.FixedLenFeature([seq_length], tf.int64),
		"input_mask_l": tf.FixedLenFeature([seq_length], tf.int64),
		"input_mask_r": tf.FixedLenFeature([seq_length], tf.int64),
		"segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"ex_infos": tf.FixedLenFeature([FLAGS.exinfos_length], tf.float32),
		"label_ids": tf.FixedLenFeature([], tf.int64),
		"is_real_example": tf.FixedLenFeature([], tf.int64),
	}
	if FLAGS.exinfos_length == 0:
		name_to_features = {
			"input_ids_l": tf.FixedLenFeature([seq_length], tf.int64),
			"input_ids_r": tf.FixedLenFeature([seq_length], tf.int64),
			"input_mask_l": tf.FixedLenFeature([seq_length], tf.int64),
			"input_mask_r": tf.FixedLenFeature([seq_length], tf.int64),
			"segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
			"label_ids": tf.FixedLenFeature([], tf.int64),
			"is_real_example": tf.FixedLenFeature([], tf.int64),
		}


	def _decode_record(record, name_to_features):
		"""Decodes a record to a TensorFlow example."""
		example = tf.parse_single_example(record, name_to_features)

		# tf.Example only supports tf.int64, but the TPU only supports tf.int32.
		# So cast all int64 to int32.
		for name in list(example.keys()):
			t = example[name]
			if t.dtype == tf.int64:
				t = tf.to_int32(t)
			example[name] = t

		return example

	def input_fn(params):
		"""The actual input function."""
		batch_size = params["batch_size"]
		#tf.logging.info("batch size: %s", batch_size)

		# For training, we want a lot of parallel reading and shuffling.
		# For eval, we want no shuffling and parallel reading doesn't matter.
		d = tf.data.TFRecordDataset(input_file)
		if is_training:
			d = d.repeat()
			d = d.shuffle(buffer_size=100)

		d = d.apply(
			tf.contrib.data.map_and_batch(
				lambda record: _decode_record(record, name_to_features),
				batch_size=batch_size,
				drop_remainder=drop_remainder))

		return d

	return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):  # rq: no tokens_a pop()
			tokens_a.pop()
		else:
			tokens_b.pop()

def create_model(bert_config, is_training, input_ids_l, input_ids_r, input_mask_l, input_mask_r, segment_ids, ex_infos,
                                labels, num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        with tf.variable_scope(tf.get_variable_scope()):
                model_l = modeling_dw.BertModel(
                        config=bert_config,
                        is_training=is_training,
                        input_ids=input_ids_l,
                        input_mask=input_mask_l,
                        token_type_ids=segment_ids,
                        input_wides=ex_infos,
                        scope="bert",
                        use_one_hot_embeddings=use_one_hot_embeddings)


        def nonzero_average(output_layer, input_mask):
                tmp_input_mask = tf.reshape(input_mask,[-1, FLAGS.max_seq_length, 1])
                output_layer = output_layer * tf.cast(tmp_input_mask, tf.float32) #[batchsize, sequence lengt, 1]
                output_layer = tf.reduce_sum(output_layer, axis=1) #[batchsize, hiddensize)
                count_nonzero = tf.cast(tf.math.count_nonzero(input_mask, axis=1), tf.float32) + 0.001 #[batchsize]
     
                output_layer = output_layer / tf.reshape(count_nonzero, [-1,1])
                return output_layer
        
        output_layer_l = model_l.get_sequence_output() #[batchsize, sequence leng, hiddensize]
        hidden_size = output_layer_l.shape[-1].value
        output_layer_l = nonzero_average(output_layer_l, input_mask_l)

        #sentence_features_pos, sentence_features_neg = output_layer_l*output_weights_pos, output_layer_l*output_weights_neg
        #tf.logging.info("  sentence_features_pos shape = %s" % (sentence_features_pos.shape))

        #sentence_features_pos, sentence_features_neg = output_layer_l*output_weights_pos, output_weights_neg
        #return (sentence_features_pos, sentence_features_neg)
        #return (output_weights_pos, output_weights_neg)
        return(output_layer_l)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
					num_train_steps, num_warmup_steps, use_tpu,
					use_one_hot_embeddings):
	"""Returns `model_fn` closure for TPUEstimator."""

	def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
		"""The `model_fn` for TPUEstimator."""

		tf.logging.info("*** Features ***")
		for name in sorted(features.keys()):
			tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

		input_ids_l = features["input_ids_l"]
		input_ids_r = features["input_ids_r"]
		input_mask_l = features["input_mask_l"]
		input_mask_r = features["input_mask_r"]
		segment_ids = features["segment_ids"]
		ex_infos = None
		if FLAGS.exinfos_length != 0:
			ex_infos = features["ex_infos"]
		label_ids = features["label_ids"]
		is_real_example = None
		if "is_real_example" in features:
			is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
		else:
			is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

		is_training = (mode == tf.estimator.ModeKeys.TRAIN)

		output_layer = create_model(

			bert_config, is_training, input_ids_l, input_ids_r, input_mask_l,input_mask_r, segment_ids, ex_infos, label_ids,
			num_labels, use_one_hot_embeddings)

		tvars = tf.trainable_variables()
		initialized_variable_names = {}
		scaffold_fn = None
		if init_checkpoint:
			(assignment_map, initialized_variable_names) = modeling_dw.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
			if use_tpu:

				def tpu_scaffold():
					tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
					return tf.train.Scaffold()

				scaffold_fn = tpu_scaffold
			else:
				tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

		tf.logging.info("**** Trainable Variables ****")
		for var in tvars:
			init_string = ""
			if var.name in initialized_variable_names:
				init_string = ", *INIT_FROM_CKPT*"
			tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
							init_string)

		output_spec = None
		if mode == tf.estimator.ModeKeys.TRAIN:

			train_op = optimization.create_optimizer(
				total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

			logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=50)
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				loss=total_loss,
				training_hooks=[logging_hook],
				train_op=train_op,
				scaffold_fn=scaffold_fn)
		elif mode == tf.estimator.ModeKeys.EVAL:
			def metric_fn(per_example_loss, label_ids, logits, is_real_example):
					predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
					accuracy = tf.metrics.accuracy(
					labels=label_ids, predictions=predictions, weights=is_real_example)
					loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
					return {
						"eval_accuracy": accuracy,
						"eval_loss": loss,
					}

			eval_metrics = (metric_fn,
				[per_example_loss, label_ids, logits, is_real_example])
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				loss=total_loss,
				eval_metrics=eval_metrics,
				scaffold_fn=scaffold_fn)
		
		else:
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				predictions={"output_layer": output_layer},
				scaffold_fn=scaffold_fn)
		return output_spec

	return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""

	all_input_ids_l = []
	all_input_ids_r = []
	all_input_mask_l = []
	all_input_mask_r = []
	all_segment_ids = []
	all_label_ids = []

	for feature in features:
		all_input_ids_l.append(feature.input_ids_l)
		all_input_ids_r.append(feature.input_ids_r)
		all_input_mask_l.append(feature.input_mask_l)
		all_input_mask_r.append(feature.input_mask_r)
		all_segment_ids.append(feature.segment_ids)
		all_label_ids.append(feature.label_id)

	def input_fn(params):
		"""The actual input function."""
		batch_size = params["batch_size"]

		num_examples = len(features)

		# This is for demo purposes and does NOT scale to large data sets. We do
		# not use Dataset.from_generator() because that uses tf.py_func which is
		# not TPU compatible. The right way to load data is with TFRecordReader.
		d = tf.data.Dataset.from_tensor_slices({
			"input_ids_l":
				tf.constant(
					all_input_ids_l, shape=[num_examples, seq_length],
					dtype=tf.int32),
			"input_ids_r":
				tf.constant(
					all_input_ids_r, shape=[num_examples, seq_length],
					dtype=tf.int32),
			"input_mask_l":
				tf.constant(
					all_input_mask_l,
					shape=[num_examples, seq_length],
					dtype=tf.int32),
			"input_mask_r":
				tf.constant(
					all_input_mask_r,
					shape=[num_examples, seq_length],
					dtype=tf.int32),
			"segment_ids":
				tf.constant(
					all_segment_ids,
					shape=[num_examples, seq_length],
					dtype=tf.int32),
			"label_ids":
				tf.constant(all_label_ids, shape=[num_examples, len(LABEL_COLUMNS)], dtype=tf.int32),
		})

		if is_training:
			d = d.repeat()
			d = d.shuffle(buffer_size=100)

		d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
		return d

	return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
								tokenizer):
	"""Convert a set of `InputExample`s to a list of `InputFeatures`."""

	features = []
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

		feature = convert_single_example(ex_index, example, label_list,
										max_seq_length, tokenizer)

		features.append(feature)
	return features


def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)

	processors = {
		"wbtrash": WBTrashProcessor,
		"wb_binary_ab": WBTrashProcessor,
		"wb_dual_tower": Wb_Dual_Tower_Processor,
		"wbmz": WbmzProcessor,
		"exinfos": ExinfosProcessor,
		"wide": WideProcessor,
	}

	tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
											FLAGS.init_checkpoint)

	if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
		raise ValueError(
			"At least one of `do_train`, `do_eval` or `do_predict' must be True.")

	bert_config = modeling_dw.BertConfig.from_json_file(FLAGS.bert_config_file)

	if FLAGS.max_seq_length > bert_config.max_position_embeddings:
		raise ValueError(
			"Cannot use sequence length %d because the BERT model "
			"was only trained up to sequence length %d" %
			(FLAGS.max_seq_length, bert_config.max_position_embeddings))

	tf.gfile.MakeDirs(FLAGS.output_dir)

	task_name = FLAGS.task_name.lower()

	if task_name not in processors:
		raise ValueError("Task not found: %s" % (task_name))

	processor = processors[task_name]()

	label_list = processor.get_labels()

	tokenizer = tokenization.FullTokenizer(
		vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

	tpu_cluster_resolver = None
	if FLAGS.use_tpu and FLAGS.tpu_name:
		tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

	is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
	run_config = tf.contrib.tpu.RunConfig(
		cluster=tpu_cluster_resolver,
		master=FLAGS.master,
		model_dir=FLAGS.output_dir,
		save_checkpoints_steps=FLAGS.save_checkpoints_steps,
		tpu_config=tf.contrib.tpu.TPUConfig(
			iterations_per_loop=FLAGS.iterations_per_loop,
			num_shards=FLAGS.num_tpu_cores,
			per_host_input_for_training=is_per_host))

	train_examples = None
	num_train_steps = None
	num_warmup_steps = None

	model_fn = model_fn_builder(
		bert_config=bert_config,
		num_labels=FLAGS.num_variant_tags,  # len(label_list),
		init_checkpoint=FLAGS.init_checkpoint,
		learning_rate=FLAGS.learning_rate,
		num_train_steps=num_train_steps,
		num_warmup_steps=num_warmup_steps,
		use_tpu=FLAGS.use_tpu,
		use_one_hot_embeddings=FLAGS.use_tpu)

	# If TPU is not available, this will fall back to normal Estimator on CPU
	# or GPU.
	estimator = tf.contrib.tpu.TPUEstimator(
		use_tpu=FLAGS.use_tpu,
		model_fn=model_fn,
		config=run_config,
		train_batch_size=FLAGS.train_batch_size,
		eval_batch_size=FLAGS.eval_batch_size,
		predict_batch_size=FLAGS.predict_batch_size)

	if FLAGS.do_predict and not FLAGS.tag_embedding:
		eval_examples = processor.get_test_examples(FLAGS.data_dir)
		num_actual_eval_examples = len(eval_examples)
		if FLAGS.use_tpu:
			# TPU requires a fixed batch size for all batches, therefore the number
			# of examples must be a multiple of the batch size, or else examples
			# will get dropped. So we pad with fake examples which are ignored
			# later on. These do NOT count towards the metric (all tf.metrics
			# support a per-instance weight, and these get a weight of 0.0).
			while len(eval_examples) % FLAGS.eval_batch_size != 0:
				eval_examples.append(PaddingInputExample())

		eval_file = os.path.join(FLAGS.output_dir, "midtext.tf_record")
		file_based_convert_examples_to_features(
			eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

		tf.logging.info("***** Running evaluation *****")
		tf.logging.info("  Num examples = %d (%d actual, %d padding)",
						len(eval_examples), num_actual_eval_examples,
						len(eval_examples) - num_actual_eval_examples)
		tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

		eval_input_fn = file_based_input_fn_builder(
			input_file=eval_file,
			seq_length=FLAGS.max_seq_length,
			is_training=False,
			drop_remainder=False) #for deep wide 

		result = estimator.predict(input_fn=eval_input_fn)

		num_written_lines=0
		output_eval_file = os.path.join(FLAGS.output_dir, "midtext.tsv")
		with tf.gfile.GFile(output_eval_file, "w") as writer:
			for (i, prediction) in enumerate(result):
				num_written_lines += 1
				output_layer = prediction["output_layer"]
				assert len(output_layer) == 768
				output_line =  " ".join( str(format(x, '.4f')) for x in output_layer) + "\n"
				writer.write(output_line)
		#assert num_written_lines == num_actual_eval_examples


	if FLAGS.do_predict and FLAGS.tag_embedding:
		predict_examples = processor.get_test_examples(FLAGS.data_dir)
		num_actual_predict_examples = len(predict_examples)
		output_fvs = {}
		index_fvs = []
		count_fvs = {}
		for line in predict_examples:
			tag = line.text_b
			output_fvs[tag] = [0]*768
			count_fvs[tag] = 0
			index_fvs.append(tag)



		if FLAGS.use_tpu:
			# TPU requires a fixed batch size for all batches, therefore the number
			# of examples must be a multiple of the batch size, or else examples
			# will get dropped. So we pad with fake examples which are ignored
			# later on.
			while len(predict_examples) % FLAGS.predict_batch_size != 0:
				predict_examples.append(PaddingInputExample())

		predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
		file_based_convert_examples_to_features(predict_examples, label_list,
												FLAGS.max_seq_length, tokenizer,
												predict_file)

		tf.logging.info("***** Running prediction*****")
		tf.logging.info("  Num examples = %d (%d actual, %d padding)",
						len(predict_examples), num_actual_predict_examples,
						len(predict_examples) - num_actual_predict_examples)
		tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

		predict_drop_remainder = True if FLAGS.use_tpu else False
		predict_input_fn = file_based_input_fn_builder(
			input_file=predict_file,
			seq_length=FLAGS.max_seq_length,
			is_training=False,
			drop_remainder=True)

		result = estimator.predict(input_fn=predict_input_fn)

		num_written_lines=0
		for (i, prediction) in enumerate(result):
			num_written_lines += 1
			output_layer = prediction["output_layer"]
			assert len(output_layer) == 768
			count_fvs[index_fvs[i]] += 1
			#print (i, index_fvs[i].encode('utf-8'))
			output_fvs[index_fvs[i]] = [ x+y for x,y in zip(output_fvs[index_fvs[i]], output_layer) ]
		#assert num_written_lines == num_actual_predict_examples


		output_predict_file_pos = os.path.join(FLAGS.output_dir, "tag_embedding.tsv")	
		with tf.gfile.GFile(output_predict_file_pos, "w") as writer:
			tf.logging.info("***** Predict results *****")
			for key in output_fvs:
			  #assert len(output_fvs[key]) == 768
				output_line = key + "\t" + str(count_fvs[key]) + "\t" +  " ".join( str(format(x, '.4f')) for x in output_fvs[key] ) + "\n"
				writer.write(output_line)



if __name__ == "__main__":
	flags.mark_flag_as_required("data_dir")
	flags.mark_flag_as_required("task_name")
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("bert_config_file")
	flags.mark_flag_as_required("output_dir")
	tf.app.run()