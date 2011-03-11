from hashmapd import *

import numpy
import sys
import os
import csv
import collections
import cPickle
import gzip
import theano

#reuters data
CODES_DESCRIPTION_FILE = "data/reuters/rcv1-v2.topics.qrels.gz";
DOC_CODES_FILE = "data/reuters/rcv1-v2.topics.qrels.gz";
DOC_FILES = ["data/reuters/lyrl2004_tokens_test_pt0.dat.gz","data/reuters/lyrl2004_tokens_test_pt1.dat.gz","data/reuters/lyrl2004_tokens_test_pt2.dat.gz","data/reuters/lyrl2004_tokens_test_pt3.dat.gz","data/reuters/lyrl2004_tokens_train.dat.gz"];

PICKLED_WORDS_FILE = "data/reuters/reuters_words.pkl.gz";                  # word_id -> word, total_frequency
PICKLED_TOPICS_FILE = "data/reuters/reuters_topics.pkl.gz";                # topic_id -> reuters_topic_id
PICKLED_DOCS_FILE = "data/reuters/reuters_docs.pkl.gz";                    # doc_id -> reuters_doc_id
PICKLED_WORD_VECTORS_TRAINING_FILE_PREFIX = "data/reuters/reuters_training_data";   # doc_id,word_id -> doc_word_frequency
PICKLED_WORD_VECTORS_VALIDATION_FILE_PREFIX = "data/reuters/reuters_validation_data"; # doc_id,word_id -> doc_word_frequency
PICKLED_WORD_VECTORS_TESTING_FILE_PREFIX = "data/reuters/reuters_testing_data";    # doc_id,word_id -> doc_word_frequency
PICKLED_WORD_VECTORS_FILE_POSTFIX = ".pkl.gz";

# data info (no train/validate/test batches, no files, no batches per file, mean word count)
PICKLED_WORD_VECTORS_FILE_INFO = "data/reuters_data_info.pkl.gz"



# TOPIC DESCRIPTIONS
# 1) unzip codes description file
# 2) parse codes description file and build dictionary of "reuters_topic_id -> topic_desc" (ignore hierarchy information for now)
# 3) save a pkl file "reuters_topic_id -> topic_desc"

# TOTAL WORD COUNTS
# 1) unzip all document files
# 2) parse the documents files and build a dictionary of "word_id -> word,word_count" for the top 2,000 (most used) words
# 3) save a pkl file "word_id -> word,word_count"

# DOC_IDS, TOPIC_IDS, AND DOC_WORD_COUNTS
# 1) unzip all document files
# 2) parse the documents files and build a dictionary of:
# 3)   "doc_id -> topic_id"
# 4)   "doc_id -> reuters_doc_id"
# 5)   "doc_id,word_id" -> doc_word_count"
# 6) while parsing, save a pkl file per batch containing "doc_id,word_id" -> doc_word_count"
# 7) save a pkl file "doc_id -> topic_id"
# 8) save a pkl file "doc_id -> reuters_doc_id"


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
    cPickle.dump(word_ids, f, cPickle.HIGHEST_PROTOCOL);
    f.close();


def read_doc_word_counts(counts, words_to_ids, batch_size = 10, num_words = 2000, num_total = 1000, num_training = 800, num_validation = 150, num_testing = 50):
    """
    Reads in data from reuters/lyrl2004_tokens_train.dat.gz
    """
    
    # truncate num_total to fit batch_sizes if necessary
    no_training_batches = int(math.floor(num_training/batch_size));
    no_validation_batches = int(math.floor(num_validation/batch_size));
    no_testing_batches = int(math.floor(num_testing/batch_size));
    no_mini_batches = no_training_batches+no_validation_batches+no_testing_batches;
    num_total = no_mini_batches*batch_size;
    # determine how many mini batches of data should be stored in each file (assume 500MB of memory for now)
    batches_per_file = int(math.floor((512*1024*1024)/((num_words+1+103)*batch_size*4)));
    # determine how many words are going to be stored in each file
    words_per_file = batches_per_file*batch_size;
    # determine the number of files needed
    no_training_files = int(math.ceil(num_training/float(words_per_file)));
    no_validation_files = int(math.ceil(num_validation/float(words_per_file)));
    no_testing_files = int(math.ceil(num_testing/float(words_per_file)));
    no_files = no_training_files+no_validation_files+no_testing_files;
    
    print ''
    print 'storing data in '+str(no_files)+' segment(s)'
    print 'each segment has (up to) '+str(batches_per_file)+' batches containing '+str(batch_size)+' docs each ('+str(words_per_file*num_words*4/float(1024*1024))+'MB)'
    print ''
    
    #===========================================================================
    # STEP 1) parse and save the document word vectors
    # 
    #  Note this involves reading through the files of data in sequence and once
    #  batches_per_file batches of data have been read, the data is saved.
    # 
    #  Separate files are created for the training, validation, and testing sets.
    #===========================================================================
    
    # file format:
    # .I <doc_id>
    # .W
    # <space/line separated list of stemmed words>
    # [blankline]
    
    mean_doc_size = 0; # used to determine the mean training document size
    
    doc_ids_to_reuters_ids = {}; # save this information for reference
    reuters_ids_to_doc_ids = {}; # for quick access to doc ids
    raw_counts = numpy.zeros((min(words_per_file,num_training),num_words), dtype=theano.config.floatX);
    
    set_iter = 0;   # 0 = training, 1 = validation, 2 = testing
    file_iter = 0;  # the number of files saved so far in this data set
    file_count = 0; # number of documents parsed that will be saved in this file
    doc_iter = 0;   # total number of documents parsed
    for file in DOC_FILES :
        if doc_iter >= num_total:
            break;
        
        print 'attempting to read file '+ file;
        f = gzip.open(file,'r');
        
        while True :
            # check if all the data has been read
            if doc_iter >= num_total :
                break;
            
            line = f.readline();
            # when the end of the file is reached, break
            if (line == "") :
                break;
            # debug
            if doc_iter%10000 == 0 :
                print 'reading doc '+ str(doc_iter) + '..';
            # extract reuters_id
            reuters_id = line[3:-1];
            doc_id = doc_iter;
            reuters_ids_to_doc_ids[reuters_id] = doc_id;
            doc_ids_to_reuters_ids[doc_id] = reuters_id;
            # ignore .W line
            f.readline();
            
            # extract words
            no_words = 0;
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
                        no_words += 1;
                        # update doc word count
                        raw_counts[file_count,words_to_ids[word]] += 1;
            
            if set_iter == 0 : mean_doc_size += no_words;
            
            doc_iter += 1;
            file_count += 1;
            
            # check if its time to save the data
            if (file_count > 0 and file_count%(words_per_file)==0) or \
                    (doc_iter == num_training) or \
                    (doc_iter == num_validation+num_training) or \
                    (doc_iter == num_total) :
                
                print 'saving set '+str(set_iter)+', batch '+ str(file_iter) + '..';
                
                # save batch information
                if set_iter == 0 :   file_prefix = PICKLED_WORD_VECTORS_TRAINING_FILE_PREFIX;
                elif set_iter == 1 : file_prefix = PICKLED_WORD_VECTORS_VALIDATION_FILE_PREFIX;
                elif set_iter == 2 : file_prefix = PICKLED_WORD_VECTORS_TESTING_FILE_PREFIX;
                g = gzip.open(file_prefix+str(file_iter)+PICKLED_WORD_VECTORS_FILE_POSTFIX,'wb');
                cPickle.dump((raw_counts,raw_counts.sum(axis=1)), g, cPickle.HIGHEST_PROTOCOL);
                g.close();
                
                # update counts
                file_iter += 1;
                file_count = 0;
                
                if doc_iter == num_training or doc_iter == num_validation+num_training or doc_iter == doc_iter == num_validation+num_training:
                    set_iter += 1;
                    file_iter = 0;
                
                if set_iter == 0 :   raw_counts = numpy.zeros((min(words_per_file,num_training-doc_iter),num_words), dtype=theano.config.floatX);
                elif set_iter == 1 : raw_counts = numpy.zeros((min(words_per_file,num_validation+num_training-doc_iter),num_words), dtype=theano.config.floatX);
                elif set_iter == 2 : raw_counts = numpy.zeros((min(words_per_file,num_total-doc_iter),num_words), dtype=theano.config.floatX);
    
        f.close();
    
    mean_doc_size = mean_doc_size/doc_iter;
    
    print 'mean doc size: ' + str(mean_doc_size) + ' words';
    print ''
    
    #===========================================================================
    # STEP 2) parse the document topics
    #===========================================================================
    
    print 'attempting to read document topics from ' + DOC_CODES_FILE;
    
    num_topics = 103
    
    # file format:
    #   <topic_id> <doc_id> 1
    
    f = gzip.open(DOC_CODES_FILE,'r');
    doc_topics = {};
    reuters_data_to_topic_ids = {};
    topic_ids_to_reuters_data = {};
    
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
        reuters_data = data[0];
        reuters_id = data[1][:-3];
        if reuters_id in reuters_ids_to_doc_ids :
            doc_id = reuters_ids_to_doc_ids[reuters_id];
            # assign an id to this topic if necessary
            if reuters_data not in reuters_data_to_topic_ids :
                topic_id = topic_iter;
                reuters_data_to_topic_ids[reuters_data] = topic_id;
                topic_ids_to_reuters_data[topic_id] = reuters_data;
                topic_iter += 1;
            # store the topic id with this doc id
            if doc_id in doc_topics :
                doc_topics[doc_id].append(reuters_data_to_topic_ids[reuters_data]);
            else :
                doc_topics[doc_id] = [];
                doc_topics[doc_id].append(reuters_data_to_topic_ids[reuters_data]);
    
    print 'num topics discovered: '+str(topic_iter)
    print ''
    
    #===========================================================================
    # STEP 3) save the labels along with the word counts
    #===========================================================================
    
    print 'saving document labels'
    
    doc_iter = 0;
    file_iter = 0;
    file_prefix = PICKLED_WORD_VECTORS_TRAINING_FILE_PREFIX;
    for i in xrange(no_files) :
        # load the relevant word count file
        if i == no_training_files :
            file_prefix = PICKLED_WORD_VECTORS_VALIDATION_FILE_PREFIX;
            file_iter = 0;
        elif i == no_training_files+no_validation_files :
            file_prefix = PICKLED_WORD_VECTORS_TESTING_FILE_PREFIX;
            file_iter = 0;
        f = gzip.open(file_prefix+str(file_iter)+PICKLED_WORD_VECTORS_FILE_POSTFIX,'rb');
        raw_counts,sums = cPickle.load(f);
        f.close();
        # for each category type a document belongs to, set the corresponding label value to one
        labels = numpy.zeros((raw_counts.shape[0],num_topics), dtype=theano.config.floatX);
        for j in xrange(raw_counts.shape[0]) :
            for k in doc_topics[doc_iter] :
                labels[j,k] = 1.0;
            doc_iter += 1;
        # save labels with data
        f = gzip.open(file_prefix+str(file_iter)+PICKLED_WORD_VECTORS_FILE_POSTFIX,'wb');
        if (counts): cPickle.dump((raw_counts,sums,labels), f, cPickle.HIGHEST_PROTOCOL);
        else:        cPickle.dump((raw_counts/numpy.array([x_sums]*(x.shape[1])).transpose(),sums,labels), f, cPickle.HIGHEST_PROTOCOL);
        f.close();
        
        file_iter += 1;
    
    print ''
    print 'saving data info'
    
    f = gzip.open(PICKLED_WORD_VECTORS_FILE_INFO,'wb');
    cPickle.dump((PICKLED_WORD_VECTORS_TRAINING_FILE_PREFIX,no_training_files,no_training_batches,
                    PICKLED_WORD_VECTORS_VALIDATION_FILE_PREFIX,no_validation_files,no_validation_batches,
                    PICKLED_WORD_VECTORS_TESTING_FILE_PREFIX,no_testing_files,no_testing_batches,
                        batches_per_file,mean_doc_size), f, cPickle.HIGHEST_PROTOCOL);
    f.close();
    
    print ''
    print 'done reading input';
    
    return no_training_files,no_validation_files,no_testing_files,topic_ids_to_reuters_data,doc_ids_to_reuters_ids;
    

def output_pickled_topics(topic_ids):
    
    print "outputting topics";
    print '...  pickling and zipping data to '+ PICKLED_TOPICS_FILE;
    
    f = gzip.open(PICKLED_TOPICS_FILE,'wb');
    cPickle.dump(topic_ids, f, cPickle.HIGHEST_PROTOCOL);
    f.close();


def output_pickled_docs(doc_ids):
    
    print "outputting docs";
    print '...  pickling and zipping data to '+ PICKLED_DOCS_FILE;
    
    f = gzip.open(PICKLED_DOCS_FILE,'wb');
    cPickle.dump(doc_ids, f, cPickle.HIGHEST_PROTOCOL);
    f.close();


def test_pickling(filename):
    
    f = gzip.open(filename,'rb');
    x,x_sums,y = cPickle.load(f);
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
    
    train_set_x = shared_dataset(x);
    train_sums = shared_dataset(x_sums);
    train_set_y = shared_dataset(y);
    
    print ''
    print 'batch '+str(filename)+' restored data sizes:'
    print('x: '+str(train_set_x.value.shape));
    print('x_sums: '+str(train_sums.value.shape));
    print('y: '+str(train_set_y.value.shape));


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
        no_training_files,no_validation_files,no_testing_files,topic_ids,doc_ids = read_doc_word_counts(cfg.train.first_layer_type=='poisson',words_to_ids,cfg.train.train_batch_size,cfg.shape.input_vector_length,cfg.input.number_of_examples,cfg.input.number_for_training,cfg.input.number_for_validation,cfg.input.number_for_testing);
        
        # output the data
        output_pickled_topics(topic_ids);
        output_pickled_docs(doc_ids);
        
        for i in xrange(no_training_files):
            test_pickling(PICKLED_WORD_VECTORS_TRAINING_FILE_PREFIX+str(i)+PICKLED_WORD_VECTORS_FILE_POSTFIX);
        for i in xrange(no_validation_files):
            test_pickling(PICKLED_WORD_VECTORS_VALIDATION_FILE_PREFIX+str(i)+PICKLED_WORD_VECTORS_FILE_POSTFIX);
        for i in xrange(no_testing_files):
            test_pickling(PICKLED_WORD_VECTORS_TESTING_FILE_PREFIX+str(i)+PICKLED_WORD_VECTORS_FILE_POSTFIX);


if __name__ == '__main__':
    sys.exit(main())



    