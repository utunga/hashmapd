from __future__ import division
import sys, os, bisect, numpy, numpy.random

def get_git_home():
    testpath = '.'
    while not '.git' in os.listdir(testpath) and not os.path.abspath(testpath) == '/':
        testpath = os.path.sep.join(('..', testpath))
    if not os.path.abspath(testpath) == '/':
        return os.path.abspath(testpath)
    else:
        raise ValueError, "Not in git repository"

HOME = get_git_home()
sys.path.append(HOME)

from hashmapd.load_config import LoadConfig


def generate(USERS, WORDS):
    random = numpy.random.RandomState(seed=1)
    DISTINCT = True
    TOPICS = 2
    SAMPLES = WORDS * 100

    probabilities = random.uniform(size=[WORDS, TOPICS])
    
    # normalize so sum of P(word|topic) over all words == 1
    probabilities /= probabilities.sum(axis=0)

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
    
            labels.append(int(interests[1] *10 / interests.sum()))

    return (counts, labels)

def output(counts, labels, cfg):
    vectors = open(cfg.raw.csv_data, 'w')
    users = open(cfg.raw.users_file, 'w')
    users2 = open(cfg.output.labels_file, 'w')

    print >>users, "user_id,screen"
    print >>vectors, "user_id,word_id,count" 

    for user, label in enumerate(labels):
        print >>users, user, ",", label 
        if user < cfg.input.number_for_training:
            print >>users2, label
        for (word, count) in enumerate(counts[user]):
            if count:
                print >>vectors, (("%s,%s,%s" % (user, word, count)))

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="config", default="synth_config",
        help="Path of the config file to use")
    (options, args) = parser.parse_args()

    cfg = LoadConfig(options.config)
    (counts, labels) = generate(cfg.input.number_of_examples, cfg.shape.input_vector_length)
    output(counts, labels, cfg)
    
    
    
