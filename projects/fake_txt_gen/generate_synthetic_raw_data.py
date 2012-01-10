from __future__ import division
import sys, os, bisect, numpy, numpy.random
from hashmapd.load_config import LoadConfig
from hashmapd.utils import tiled_array_image

def generate(USERS, WORDS):
    random = numpy.random.RandomState(seed=1)
    DISTINCT = True
    TOPICS = 4
    SAMPLES = WORDS * 500

    probabilities = random.uniform(size=[WORDS, TOPICS])
    
    # normalize so sum of P(word|topic) over all words == 1
    probabilities /= probabilities.sum(axis=0)

    if TOPICS == 3:
        # Order so that trace images are a smooth gradient
        probabilities = list(probabilities)
        probabilities.sort(key=lambda (a,b):a/(a+b))

    probabilities = numpy.array(probabilities)
    partitions = numpy.add.accumulate(probabilities, axis=0)
    counts = numpy.zeros([USERS, WORDS], int)
    labels = []

    for user in range(USERS):
        if DISTINCT:
            # Absolute interests, everyone talks about 0 or 1
            topic = random.randint(TOPICS)
            for x in random.uniform(size=[SAMPLES]):
                word = bisect.bisect_left(partitions[:,topic], x)
                counts[user, word] += 1
            labels.append(topic)
        
        else:
            # Mixed interests
            interests = random.uniform(size=[TOPICS])
            weights = probabilities.dot(interests)
            partition = numpy.add.accumulate(weights / weights.sum())
    
            for x in numpy.random.uniform(size=[SAMPLES]):
                word = bisect.bisect_left(partition, x)
                counts[user, word] += 1
            
            order = numpy.argsort(interests)[::-1]
            labels.append(''.join(str(i) for i in order[:3]))

    return (counts, labels)

def output(counts, labels, cfg):
    vectors = open(cfg.raw.csv_data, 'w')
    users = open(cfg.raw.users_file, 'w')
    labels_file = open(cfg.output.labels_file, 'w')

    print >>users, "user_id,screen"
    print >>vectors, "user_id,word_id,count" 

    for user, label in enumerate(labels):
        print >>users, user, ",", label 
        if user < cfg.input.number_for_training:
            print >>labels_file, label
        for (word, count) in enumerate(counts[user]):
            if count:
                print >>vectors, (("%s,%s,%s" % (user, word, count)))

def export_input_image(array, file_name, mirroring=False):
        # Construct image from the weight matrix
        #array = self.W.get_value()
        if not mirroring:
            array = array.T
        image = tiled_array_image(array)
        image.save(file_name)

def output_img_debug(counts, labels):
    # insert label at start of each row so we can sort
    sortable = []
    for i in xrange(len(counts)):
        row = counts[i].tolist()
        row.insert(0,labels[i])
        sortable.append(row)

    # sort by label
    sortable = sorted(sortable, key=lambda row:row[0], reverse=True)

    # remove labels again
    data_x = numpy.array([row[1:] for row in sortable])

    #normalize
    sums_x = data_x.sum(axis=1)[:, numpy.newaxis]
    normalized = data_x / sums_x

    print data_x.shape
    export_input_image(normalized,'test.png', False)
    export_input_image(normalized,'test_mirrored.png', True)

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()

    cfg = LoadConfig(options.config)
    (counts, labels) = generate(cfg.input.number_of_examples, cfg.shape.input_vector_length)
    output(counts, labels, cfg)
    output_img_debug(counts, labels)

