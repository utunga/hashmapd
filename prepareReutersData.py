from hashmapd import *

import numpy
import shelve
import sys
import os
import csv
import collections
import cPickle
import gzip
import theano

# Some fileconstants

#reuters data
CODES_DESCRIPTION_FILE = "data/reuters/rcv1-v2.topics.qrels.gz";
DOC_CODES_FILE = "data/reuters/rcv1-v2.topics.qrels.gz";
WORD_VECTORS_FILE = "data/reuters/lyrl2004_tokens_train.dat.gz";

PICKLED_WORDS_FILE = "data/reuters_words.pkl.gz";
PICKLED_DOC_CODES_FILE = "data/reuters_doc_codes.pkl.gz";
PICKLED_WORD_VECTORS_FILE = "data/reuters_data.pkl.gz";
PICKLED_WORD_VECTORS_RENDER_FILE = "data/reuters_data_render.pkl.gz";

DOC_FILES = ["data/reuters/lyrl2004_tokens_test_pt0.dat.gz","data/reuters/lyrl2004_tokens_test_pt1.dat.gz","data/reuters/lyrl2004_tokens_test_pt2.dat.gz","data/reuters/lyrl2004_tokens_test_pt3.dat.gz","data/reuters/lyrl2004_tokens_train.dat.gz"];

CATEGORIES = ["GDIS","E212","C151","M143","G154"]

# plan of attack:

# PREPARE STEP
# 1) unzip codes description file
# 2) parse codes description file and build dictionary of "topic_id -> topic_desc" (ignore hierarchy information for now)
# 3) save a pkl file "topic_id -> topic_desc"

# 4) unzip doc codes file
# 5) parse doc codes file and build dictionary of "doc_code -> topic_code"
# 6) save a pkl file "doc_id -> topic_code"

# 7) unzip doc word vectors file
# 8) parse doc word vectors file and build the following dictionaries:
#      "word_id -> word, total_frequency"
#      "doc_id,word_id -> , doc_word_frequency"
# 9) generate a list of the top x words with the highest frequency 
# 10) remove from the word info and doc word count dictionary any words not in this list
# 11) save a pkl file "word_id -> word, total_frequency"
# 12) save a pkl file "doc_id,word_id -> doc_word_frequency"

# TRAIN STEP
# 1) run through the hashmapd autoencoder training process

def read_topic_descriptions():
    f = gzip.open(CODES_DESCRIPTION_FILE,'r');
    
    # parse the topic codes and descriptions
    # file format:
    #   parent: <topic_id> child: <topic_id> child-description: <topic_desc>
    # (ignore the parent, and ignore "Root" topic_id)
    
    topic_descriptions = {}
    
    for line in f :
        # ignore blank lines
        if line == "\n" :
            break;
        # extract topic info
        data = line.split('\t',2);
        topic_descriptions[data[1][7:]] = data[2][19:];
        break;
    
    return topic_descriptions;


def output_topic_descriptions():
    # todo
    
    
    return;


def determine_most_used_words(num_words = 2000):
    
    total_word_counts = collections.defaultdict(int);
    
    iter = 0;
    for file in DOC_FILES :
        print 'attempting to read file '+ file;
        f = gzip.open(file,'r');
        
        while True :
            line = f.readline();
            # when the end of the file is reached, break
            if (line == "") :
                break;
            # debug
            if iter%10000 == 0 :
                print 'reading row '+ str(iter) + '..';
            iter += 1;
            # ignore doc_id and .W line
            f.readline();
            # extract words
            while True :
                line = f.readline();
                # when an empty line is encountered, break
                if (line == "\n") :
                    break;
                data = line.split(' ');
                for word in data :
                    if word[-1:] == "\n" :
                        word = word[:-1];
                    # update total word count
                    total_word_counts[word] += 1;
        
        f.close();
    
    # determine the top 2000 most frequently used words
    print 'total no of different words found = ' + str(len(total_word_counts));
    print 'determining the '+str(num_words)+' most frequently used words...';
    
    # get a list of tuples, with value in first position, key second
    sortedWordCounts = [(v,k) for k,v in total_word_counts.items()];
    max_word_counts = {};
    # sort the list
    sortedWordCounts.sort();
    sortedWordCounts = sortedWordCounts[-num_words:];
    # retain only the m highest values, from the end of the list
    for v,k in sortedWordCounts :
        max_word_counts[k] = v;
    
    # assign ids to remaining words
    words_ids = {};
    iter = 0;
    for word,count in max_word_counts.iteritems() :
        words_ids[iter] = word,count;
        iter += 1;
    
    print 'done reading words:';
    list.reverse(sortedWordCounts);
    for count,word in sortedWordCounts :
        print word+": "+str(count);
    
    return words_ids;


def output_pickled_words(word_ids):
    
    print "outputting words";
    print '...  pickling and zipping data to '+ PICKLED_WORDS_FILE;
    
    f = gzip.open(PICKLED_WORDS_FILE,'wb');
    cPickle.dump(word_ids,f, cPickle.HIGHEST_PROTOCOL);
    f.close();


def read_doc_word_counts(words_to_ids, batch_size, num_words = 2000, num_train = 1000, num_valid = 0, num_test = 0):
    
    # TODO finish batch splitting (load into memory in pieces and save as shelve persistent dictionary)
    
    # determine total amount of data to load in
    num_total = num_train + num_valid + num_test;
    # determine how many batches of data should be loaded in at once (assume 200MB of memory for now)
    mini_batches_per_mega_batch = (200*1024*1024)/(num_words*batch_size*4);
    words_per_mega_batch = mini_batches_per_mega_batch*batch_size;
    # determine no. mega-batches
    no_mega_batches = math.ceil(num_total/float(words_per_mega_batch));
    
    # dictionary of mega-batch -> data/labels/sums
    #persistentData = shelve.open(PICKLED_WORD_VECTORS_FILE);
    
    """
    Reads in data from reuters/lyrl2004_tokens_train.dat.gz
    """
    print "attempting to read " + WORD_VECTORS_FILE;
    
    # STEP 1) parse the document word vectors
    
    # file format:
    # .I <doc_id>
    # .W
    # <space/line separated list of stemmed words>
    # [blankline]
    
    f = gzip.open(WORD_VECTORS_FILE,'r');
    
    reuters_ids_to_doc_ids = {}; # for quick access to doc ids
    raw_counts = numpy.zeros((num_total,num_words), dtype=theano.config.floatX);
    
    mega_batch_iter = 0;
    doc_iter = 0;
    for file in DOC_FILES :
        if doc_iter >= num_total:
            break;
            
        print 'attempting to read file '+ file;
        f = gzip.open(file,'r');
        
        while True :
            if doc_iter >= num_total:
                break;
            
            line = f.readline();
            # when the end of the file is reached, break
            if (line == "") :
                break;
            # debug
            if doc_iter%10000 == 0 :
                print 'reading doc '+ str(doc_iter) + '..';
            doc_iter += 1;
            # check if its time to save the data
#            if doc_iter > 0 and doc_iter%(words_per_mega_batch-1)==0:
#                print 'saving batch '+ str(mega_batch_iter) + '..';
#                
#                sums = raw_counts.sum(axis=1)
#                
#                train_set_x = raw_counts[0:num_train,:]
#                valid_set_x = raw_counts[num_train:num_train+num_valid,:]
#                test_set_x = raw_counts[num_train+num_valid:num_train+num_valid+num_test,:]
#                
#                train_set_y = labels[0:num_train]
#                valid_set_y = labels[num_train:num_train+num_valid]
#                test_set_y = labels[num_train+num_valid:num_train+num_valid+num_test]
#                
#                train_sums = sums[0:num_train]
#                valid_sums = sums[num_train:num_train+num_valid]
#                test_sums = sums[num_train+num_valid:num_train+num_valid+num_test]
#                
#                persistentData[mega_batch_iter] = (train_set_x,train_set_y,train_sums),(valid_set_x,valid_set_y,valid_sums),(test_set_x,test_set_y,test_sums);
#                
#                mega_batch_iter += 1;
            # extract reuters_id
            reuters_id = line[3:-1];
            if reuters_id in reuters_ids_to_doc_ids :
                doc_id = reuters_ids_to_doc_ids[reuters_id];
            else :
                doc_id = doc_iter;
                reuters_ids_to_doc_ids[reuters_id] = doc_id;
                doc_iter += 1;
            # ignore .W line
            f.readline();
            
            # extract words
            while True :
                line = f.readline();
                # when an empty line is encountered, break
                if (line == "\n") :
                    break;
                
                data = line.split(' ');
                for word in data :
                    if word[-1:] == "\n" :
                        word = word[:-1];
                    if word in words_to_ids :
                        # update doc word count
                        raw_counts[doc_id,words_to_ids[word]] += 1;
    
    f.close();
    # persistentData.close();
    
    # STEP 2) Parse the document topics (while we have the assigned doc_id info available)
    
    print 'attempting to read document topics from ' + DOC_CODES_FILE;
    
    # file format:
    #   <topic_id> <doc_id> 1
    
    f = gzip.open(DOC_CODES_FILE,'r');
    labels = numpy.zeros(num_total, dtype=theano.config.floatX);
    doc_topics = {};
    topic_ids = {};
    
    iter = 0;
    topic_iter = 0;
    for line in f :
        # debug
        if iter%100000 == 0 :
            print 'reading row '+ str(iter) + '..';
        iter += 1;
        # ignore blank lines
        if line == "\n" :
            break;
        # extract doc info
        data = line.split(' ',1);
        reuters_id = data[1][:-3];
        if reuters_id in reuters_ids_to_doc_ids :
            doc_id = reuters_ids_to_doc_ids[reuters_id];
            # for now, record the first label for the file only (later we will have to get pre-determined topic ids)
            if data[0] not in topic_ids :
                topic_ids[data[0]] = topic_iter;
                topic_iter += 1;
            
            if doc_id in doc_topics :
                doc_topics[doc_id].append(data[0]);
            else :
                doc_topics[doc_id] = [];
                doc_topics[doc_id].append(data[0]);
                labels[doc_id] = topic_ids[data[0]];
    
    sums = raw_counts.sum(axis=1);
    
    train_set_x = raw_counts[0:num_train,:]
    valid_set_x = raw_counts[num_train:num_train+num_valid,:]
    test_set_x = raw_counts[num_train+num_valid:num_train+num_valid+num_test,:]
    
    train_set_y = labels[0:num_train]
    valid_set_y = labels[num_train:num_train+num_valid]
    test_set_y = labels[num_train+num_valid:num_train+num_valid+num_test]
    
    train_sums = sums[0:num_train]
    valid_sums = sums[num_train:num_train+num_valid]
    test_sums = sums[num_train+num_valid:num_train+num_valid+num_test]
    
    # quick debug (total word counts in these documents)
    word_sums = raw_counts.sum(axis=0);
    word_ids = {}
    for word,id in words_to_ids.iteritems() :
        word_ids[id] = word;
    list = []
    for i in xrange(len(word_sums)):
        list.append((word_sums[i],word_ids[i]));
    list.sort();
    for i in xrange(len(list)):
        list.append(list[i]);
        
    print list
    
    print 'done reading input';
    
    print train_set_x;
    print valid_set_x;
    print test_set_x;
    
    print train_set_y;
    print valid_set_y;
    print test_set_y;
    
    return doc_topics,train_set_x,valid_set_x,test_set_x,train_set_y,valid_set_y,test_set_y,train_sums,valid_sums,test_sums;


def output_pickled_doc_topics(doc_topics):
    
    print "outputting doc topics";
    print '...  pickling and zipping data to '+ PICKLED_DOC_CODES_FILE;
    
    f = gzip.open(PICKLED_DOC_CODES_FILE,'wb');
    cPickle.dump(doc_topics,f, cPickle.HIGHEST_PROTOCOL);
    f.close();


def output_pickled_data(train_set,valid_set,test_set,train_set_y,valid_set_y,test_set_y,train_sums,valid_sums,test_sums):
    
    print "outputting full data set";
    print '...  pickling and zipping data to '+ PICKLED_WORD_VECTORS_FILE;
    
    f = gzip.open(PICKLED_WORD_VECTORS_FILE,'wb');
    cPickle.dump(((train_set,train_set_y,train_sums),(valid_set,valid_set_y,valid_sums),(test_set,test_set_y,test_sums)),f, cPickle.HIGHEST_PROTOCOL);
    f.close();


def test_pickling(dataset=PICKLED_WORD_VECTORS_FILE):

    f = gzip.open(dataset,'rb');
    train_set,valid_set,test_set = cPickle.load(f);
    f.close();

    def shared_dataset(data_x):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX));
        return shared_x;
    
    train_set_x = shared_dataset(train_set[0]);
    valid_set_x = shared_dataset(valid_set[0]);
    test_set_x = shared_dataset(test_set[0]);
    
    train_set_y = shared_dataset(train_set[1]);
    valid_set_y = shared_dataset(valid_set[1]);
    test_set_y = shared_dataset(test_set[1]);
    
    train_sums = shared_dataset(train_set[2]);
    valid_sums = shared_dataset(valid_set[2]);
    test_sums = shared_dataset(test_set[2]);
    
    print(train_set_x.value.shape);
    print(valid_set_x.value.shape);
    print(test_set_x.value.shape);
    
    print(train_set_y.value.shape);
    print(valid_set_y.value.shape);
    print(test_set_y.value.shape);
    
    print(train_sums.value.shape);
    print(valid_sums.value.shape);
    print(test_sums.value.shape);


def main(argv = sys.argv):
    opts, args = getopt.getopt(argv[1:], "h", ["help"])
    
    cfg = LoadConfig("reuters")
    
    if (args[0]=='max_words'):
        # determine the most used words in the entire reuters data set
        word_ids = determine_most_used_words(cfg.shape.input_vector_length)
        
        output_pickled_words(word_ids)
        
    elif (args[0]=='prepare_data'):
        # load word ids
        f = gzip.open(PICKLED_WORDS_FILE,'rb');
        ids_to_words = cPickle.load(f);
        f.close();
        
        # flip dictionary (this is the one time we need quick access via words to word_ids)
        words_to_ids = {}
        for id,(word,count) in ids_to_words.iteritems() :
            words_to_ids[word] = id;
        
        # read in the data
        doc_topics,train_set_x,valid_set_x,test_set_x,train_set_y,valid_set_y,test_set_y,train_sums,valid_sums,test_sums = \
            read_doc_word_counts(words_to_ids,cfg.train.train_batch_size,cfg.shape.input_vector_length,cfg.input.number_for_training,cfg.input.number_for_validation,cfg.input.number_for_testing);
        
        # output the data
        output_pickled_doc_topics(doc_topics);
        output_pickled_data(train_set_x,valid_set_x,test_set_x,train_set_y,valid_set_y,test_set_y,train_sums,valid_sums,test_sums);
        
        test_pickling()


if __name__ == '__main__':
    sys.exit(main())



    