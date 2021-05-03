"""
__info: This file is to convert Vietnamese summarization raw data into tokenized chunks for summarization model.
        This file is modified and mostly borrowed code from https://github.com/becxer/cnn-dailymail/
        Added some modifications to adapt to Vietnamese dataset (https://github.com/ThanhChinhBK/vietnews/blob/master/data/)
__original-author__ = "Abigail See" + converted to python3 by Becxer
__modified-author__ = "Vy Thai"
__email__ = "vythai@stanford.edu"
"""

import tensorflow as tf
from tensorflow.core.example import example_pb2
import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
from os import listdir
import collections

finished_files_dir = "finished_files"

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data
dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence
chunks_dir = os.path.join(finished_files_dir, "chunked")

#kept from original version
def chunk_file(set_name):
  in_file = 'finished_files/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1

#kept from original version
def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train','val','test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)

#This function is to load the file
def load_document(filename):
    print(filename)
    file = open(filename, encoding='utf-8')
    text = file.read()
    file.close()
    return text

#kept from original version, to add missing period symbol.
def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

# This function is used to split each datapoint to abstraction and article, and then clean it.
def split_document(doc):

    #split into title + abstract + text
    splits = doc.split('\n\n')
    title = splits[0]
    abstract = splits[1]
    text = splits[2]

    return clean_lines(text.split('\n'), "story"), clean_lines(abstract.split('\n'), "abs")

# This function is to take dir and load data and process them.
def load_articles(directory):
    documents = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_document(filename)
        # split into story and highlights
        article, abstract = split_document(doc)
        # store
        documents.append({'name': filename, 'story': article, 'highlights': abstract})
    return documents

#This function is used to clean the datapoint
def clean_lines(lines, type_):
    cleaned = list()
    #lowercase
    lines = [line.lower() for line in lines]
    if type_ == 'abs':
        # abstraction has multi-sentence but on one line
        line = lines[0]
        line = line.split(' .')
        line = [l.strip() for l in line]
        line = [c for c in line if len(c) > 0]
        line = [ l+ " ." for l in line]

        # Make abstract into a signle string, putting <s> and </s> tags around the sentences
        cleaned = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in line])
    else:
        article = [fix_missing_period(n) for n in lines]
        cleaned = ' '.join(article)
    return cleaned

# mostly kept from original version
def write_to_bin(in_file_dir, out_file, makevocab=False):

  if makevocab:
    vocab_counter = collections.Counter()

  count = 0

  list_dir = listdir(in_file_dir)
  num_stories = len(list_dir)

  with open(out_file, 'wb') as writer:
    for name in list_dir:
      if(name != ".DS_Store"):
        if count % 1000 == 0:
          print("Writing story %i of %i; %.2f percent done" % (count, num_stories, float(count)*100.0/float(num_stories)))

        # Get the strings to write to .bin file
        filename = in_file_dir + '/' + name
        # load document
        doc = load_document(filename)
        # split into story and highlights
        article, abstract = split_document(doc)
        print(abstract)
        print("\n")
        print(article)



        # Write to tf.Example
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
        tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))

        # Write the vocab to file, if applicable
        if makevocab:
          art_tokens = article.split(' ')
          abs_tokens = abstract.split(' ')
          abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
          tokens = art_tokens + abs_tokens
          tokens = [t.strip() for t in tokens] # strip
          tokens = [t for t in tokens if t!=""] # remove empty
          vocab_counter.update(tokens)

  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")

# load stories
directory = 'viet'
#stories = load_stories(directory)
#print('Loaded Stories %d' % len(stories))
#print(stories[1]['highlights'])
#for example in stories:#
#	example['story'] = clean_lines(example['story'].split('\n'), "story")#
#	example['highlights'] = clean_lines(example['highlights'].split('\n'), "abs")

#print(stories[1]['highlights'])

if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
write_to_bin(os.path.join(directory, "val"), os.path.join(finished_files_dir, "val.bin"), makevocab=True)
write_to_bin(os.path.join(directory, "train"), os.path.join(finished_files_dir, "train.bin"), makevocab=True)
write_to_bin(os.path.join(directory, "test"), os.path.join(finished_files_dir, "test.bin"), makevocab=True)

#write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
#write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"), makevocab=True)

chunk_all()
