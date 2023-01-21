#coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#			http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import modeling
import optimization
import tensorflow as tf
import tokenization
import csv, codecs

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
		"bert_config_file", None,
		"The config json file corresponding to the pre-trained BERT model. "
		"This specifies the model architecture.")

flags.DEFINE_string(
		"input_file", None,
		"Input TF example files (can be a glob or comma separated).")



flags.DEFINE_string(
		"output_dir", None,
		"The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
            "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
                "do_lower_case", True,
                "Whether to lower case the input text. Should be True for uncased "
                "models and False for cased models.")


## Other parameters
flags.DEFINE_string(
		"init_checkpoint", None,
		"Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
		"max_seq_length", 128,
		"The maximum total input sequence length after WordPiece tokenization. "
		"Sequences longer than this will be truncated, and sequences shorter "
		"than this will be padded. Must match data generation.")

flags.DEFINE_integer(
		"max_predictions_per_seq", 6,
		"Maximum number of masked LM predictions per sequence. "
		"Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 10000,
										 "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
										 "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")

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

def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature


def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
	"""Truncates a pair of sequences to a maximum sequence length."""
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_num_tokens:
			break

		trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
		assert len(trunc_tokens) >= 1
		trunc_tokens.pop()



def write_examples_to_features(input_files, max_seq_length,
								max_predictions_per_seq,  tokenizer, output_file):
	max_num_tokens = max_seq_length - 2  #[CLS] [SEP]
	writers = tf.python_io.TFRecordWriter(output_file)
	total_written = 0

	with tf.gfile.Open(input_files, "rb") as f:
		reader = csv.reader(codecs.iterdecode(f, 'utf-8', errors='ignore'), delimiter="\t", quotechar=None)
		for i,line in enumerate(reader):
			text_a = tokenization.convert_to_unicode(line[1])
			tokens_a = tokenizer.tokenize(text_a)
			tokens_b = ['[MASK]','[MASK]','[MASK]','[MASK]','[MASK]','[MASK]','[MASK]']
			#tf.logging.info("text a:  %s", text_a)
			#tf.logging.info("text b:  %s", text_b)
			#tf.logging.info("tokens a:  %s", ' '.join(tokens_a))
			#tf.logging.info("tokens b:  %s", ' '.join(tokens_b))
			truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
			if not len(tokens_a) >= 1 or not len(tokens_b) >= 1:	
					tf.logging.info('input file format error: text too short\n')
					exit(-1)

			tokens = []
			segment_ids = []
			tokens.append("[CLS]")
			segment_ids.append(0)
			for token in tokens_a:
				tokens.append(token)
				segment_ids.append(0)
			tokens.append("[SEP]")
			segment_ids.append(0)

			masked_lm_positions = []
			for token in tokens_b:
				tokens.append(token)
				masked_lm_positions.append(len(tokens)-1)
				segment_ids.append(1)
			#tokens.append("[SEP]")
			#segment_ids.append(1)
			masked_lm_weights = [0] * max_predictions_per_seq
			masked_lm_ids = [0]*max_predictions_per_seq
			masked_lm_labels = ['0']*max_predictions_per_seq
			while len(masked_lm_positions) < max_predictions_per_seq:
				masked_lm_positions.append(0)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			input_mask = [1] * len(input_ids)
			assert len(input_ids) <= max_seq_length

			while len(input_ids) < max_seq_length:
					input_ids.append(0)
					input_mask.append(0)
					segment_ids.append(0)

			next_sentence_label = 0
			features = collections.OrderedDict()
			features["input_ids"] = create_int_feature(input_ids)
			features["input_mask"] = create_int_feature(input_mask)
			features["segment_ids"] = create_int_feature(segment_ids)
			features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
			features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
			features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
			features["next_sentence_labels"] = create_int_feature([next_sentence_label])

			tf_example = tf.train.Example(features=tf.train.Features(feature=features))

			writers.write(tf_example.SerializeToString())

			total_written += 1

			if total_written < 20:
				tf.logging.info("*** Example %d ***" % total_written)
				tf.logging.info("tokens: %s" % " ".join(
					[tokenization.printable_text(x) for x in tokens]))

				for feature_name in features.keys():
					feature = features[feature_name]
					values = []
					if feature.int64_list.value:
						values = feature.int64_list.value
					elif feature.float_list.value:
							values = feature.float_list.value
					tf.logging.info(
						"%s: %s" % (feature_name, " ".join([str(x) for x in values])))

	writers.close()

	tf.logging.info("Wrote %d total instances", total_written)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
										 num_train_steps, num_warmup_steps, use_tpu,
										 use_one_hot_embeddings):
	"""Returns `model_fn` closure for TPUEstimator."""

	def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
		"""The `model_fn` for TPUEstimator."""

		tf.logging.info("*** Features ***")
		for name in sorted(features.keys()):
			tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		masked_lm_positions = features["masked_lm_positions"]
		masked_lm_ids = features["masked_lm_ids"]
		masked_lm_weights = features["masked_lm_weights"]
		next_sentence_labels = features["next_sentence_labels"]

		is_training = (mode == tf.estimator.ModeKeys.TRAIN)

		model = modeling.BertModel(
				config=bert_config,
				is_training=is_training,
				input_ids=input_ids,
				input_mask=input_mask,
				token_type_ids=segment_ids,
				use_one_hot_embeddings=use_one_hot_embeddings)

		topn, topn_probs = get_masked_lm_output(
				 bert_config, model.get_sequence_output(), model.get_embedding_table(),
				 masked_lm_positions, masked_lm_ids, masked_lm_weights)

		tvars = tf.trainable_variables()

		initialized_variable_names = {}
		scaffold_fn = None
		if init_checkpoint:
			(assignment_map, initialized_variable_names
			) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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
		output_spec = tf.contrib.tpu.TPUEstimatorSpec(
					mode=mode,
					predictions={"topn": topn, "topn_probs": topn_probs},
					scaffold_fn=scaffold_fn)

		return output_spec

	return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
												 label_ids, label_weights):
	"""Get loss and log probs for the masked LM."""
	#input_tensor = gather_indexes(input_tensor, positions)
	input_tensor = tf.gather(input_tensor, positions,axis=1,batch_dims=1)

	with tf.variable_scope("cls/predictions"):
		# We apply one more non-linear transformation before the output layer.
		# This matrix is not used after pre-training.
		with tf.variable_scope("transform"):
			input_tensor = tf.layers.dense(
					input_tensor,
					units=bert_config.hidden_size,
					activation=modeling.get_activation(bert_config.hidden_act),
					kernel_initializer=modeling.create_initializer(
							bert_config.initializer_range))
			input_tensor = modeling.layer_norm(input_tensor)

		# The output weights are the same as the input embeddings, but there is
		# an output-only bias for each token.
		output_bias = tf.get_variable(
				"output_bias",
				shape=[bert_config.vocab_size],
				initializer=tf.zeros_initializer())
		logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		probs = tf.nn.softmax(logits, axis=-1) #[batch, masked, vocab_size]
		sorted_probs = tf.argsort(probs,axis=-1, direction='DESCENDING', stable=False, name=None)
		topn = tf.slice(sorted_probs,[0,0,0],[-1,-1,4])
		topn_probs = tf.gather(probs,topn, axis=2, batch_dims=2)

		#tokenizer.convert_ids_to_tokens(ids)
	return  (topn, topn_probs)




def gather_indexes(sequence_tensor, positions):
	"""Gathers the vectors at the specific positions over a minibatch."""
	sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
	batch_size = sequence_shape[0]
	seq_length = sequence_shape[1]
	width = sequence_shape[2]

	flat_offsets = tf.reshape(
			tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
	flat_positions = tf.reshape(positions + flat_offsets, [-1])
	flat_sequence_tensor = tf.reshape(sequence_tensor,
																		[batch_size * seq_length, width])
	output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
	return output_tensor


def input_fn_builder(input_files,
										 max_seq_length,
										 max_predictions_per_seq,
										 is_training,
										 num_cpu_threads=4):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""

	def input_fn(params):
		"""The actual input function."""
		batch_size = params["batch_size"]

		name_to_features = {
				"input_ids":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"input_mask":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"segment_ids":
						tf.FixedLenFeature([max_seq_length], tf.int64),
				"masked_lm_positions":
						tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
				"masked_lm_ids":
						tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
				"masked_lm_weights":
						tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
				"next_sentence_labels":
						tf.FixedLenFeature([1], tf.int64),
		}

		# For training, we want a lot of parallel reading and shuffling.
		# For eval, we want no shuffling and parallel reading doesn't matter.
		if is_training:
			d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
			d = d.repeat()
			d = d.shuffle(buffer_size=len(input_files))

			# `cycle_length` is the number of parallel files that get read.
			cycle_length = min(num_cpu_threads, len(input_files))

			# `sloppy` mode means that the interleaving is not exact. This adds
			# even more randomness to the training pipeline.
			d = d.apply(
					tf.contrib.data.parallel_interleave(
							tf.data.TFRecordDataset,
							sloppy=is_training,
							cycle_length=cycle_length))
			d = d.shuffle(buffer_size=100)
		else:
			d = tf.data.TFRecordDataset(input_files)
			# Since we evaluate for a fixed number of steps we don't want to encounter
			# out-of-range exceptions.
			#d = d.repeat() #predict no repeat

		# We must `drop_remainder` on training because the TPU requires fixed
		# size dimensions. For eval, we assume we are evaluating on the CPU or GPU
		# and we *don't* want to drop the remainder, otherwise we wont cover
		# every sample.
		d = d.apply(
				tf.contrib.data.map_and_batch(
						lambda record: _decode_record(record, name_to_features),
						batch_size=batch_size,
						num_parallel_batches=num_cpu_threads,
						drop_remainder=True))
		return d

	return input_fn


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


def main(_):
	tf.logging.set_verbosity(tf.logging.INFO)

	if not FLAGS.do_train and not FLAGS.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

	tf.gfile.MakeDirs(FLAGS.output_dir)

	input_files = []
	for input_pattern in FLAGS.input_file.split(","):
		input_files.extend(tf.gfile.Glob(input_pattern))

	tf.logging.info("*** Input Files ***")
	for input_file in input_files:
		tf.logging.info("  %s" % input_file)

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

	model_fn = model_fn_builder(
			bert_config=bert_config,
			init_checkpoint=FLAGS.init_checkpoint,
			learning_rate=FLAGS.learning_rate,
			num_train_steps=FLAGS.num_train_steps,
			num_warmup_steps=FLAGS.num_warmup_steps,
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
			predict_batch_size=FLAGS.eval_batch_size)


	if FLAGS.do_eval:
		tf.logging.info("***** Running evaluation *****")
		tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

		predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
		tokenizer = tokenization.FullTokenizer(
				vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

		write_examples_to_features(FLAGS.input_file, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, tokenizer,
				predict_file)

		eval_input_fn = input_fn_builder(
				input_files=predict_file,
				max_seq_length=FLAGS.max_seq_length,
				max_predictions_per_seq=FLAGS.max_predictions_per_seq,
				is_training=False)

		result = estimator.predict(
				input_fn=eval_input_fn)

		output_eval_file = os.path.join(FLAGS.output_dir, "test_results.txt")
		with tf.gfile.GFile(output_eval_file, "w") as writer:
			tf.logging.info("***** Eval results *****")
			for i,prediction in enumerate(result):
				topn = prediction["topn"]
				topn_probs = prediction["topn_probs"]
				tokens = []
				for x in topn.tolist():
					tokens.append(tokenizer.convert_ids_to_tokens(x))
				tied = zip( tokens, topn_probs.tolist())
				writer.write("%s\n\n" % '\n'.join([str(x)+'\t'+str(y) for x,y in tied]))
				#writer.write("%s" % '\t'.join([ str(x) for x in topn.tolist()]))
				#writer.write("\t%s\n" % '\t'.join([ str(x) for x  in topn_probs.tolist()]))


if __name__ == "__main__":
	flags.mark_flag_as_required("input_file")
	flags.mark_flag_as_required("bert_config_file")
	flags.mark_flag_as_required("output_dir")
	tf.app.run()
