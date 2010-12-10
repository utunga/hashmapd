
*Due to an appalling structural design, or rather, lack thereof, here are some instructions that should give you a rough idea of what the various python script files are for and how to get started*

*Generally, one would expect to run the python file at each step by simply typing the name of the 'main python file' without any command line parameters eg*
> python prepareUserInput.py

*You should be able to get started at any point in the below process as I have, with luck, checked the relevant intermediate files into git. For instance.*

*A good quick start would be to jump to the presentation (variations) step and have a look at the map for the 5000_user_labels.csv/user_coords_perplexity_50.csv variation.* 


* Quick review of basic data flow *
=======

**DATA PARSE STEP** [see xtract repository, C#]
{twitter stream} -> raw tweets

**DATA PARSE STEP** [see xtract repository, C#]
raw tweets -> user+word counts (in db), plus some stats

**DATA PREPREP STEP** [xtract repository, SQL]
user+word counts (in db) -> user/word vectors (csv)
user+word counts (in db) -> user labels (csv)

**DATA PREP STEP** 
user/word vectors (csv) -> training data (pkl.gz)
user/word vectors (csv) -> presentation data (pkl.gz)

**TRAIN STEP**
training data (pkl.gz) -> trained weights (pkl.gz)

**RUN STEP**
trained weights (pkl.gz) 
+ presentation data (pkl.gz) -> coords (csv)

**PRESENTATION STEP**
coords (csv) + user labels (csv) -> visual (java applet)

* Details of data flow *
====

DATA PREP STEP
---
**user/word vectors (csv) -> training data (pkl.gz)**
**user/word vectors (csv) -> presentation data (pkl.gz)**

main python file:
**prepareUserVectorsData.py**
- reads data/user_word_vectors.csv 
- writes data/word_vectors.pkl.gz
- writes data/word_vectors_display.pkl.gz

data/user_word_vectors.csv
 input file, contains rows of all ints:
 [user_id, word_id, count]

data/word_vectors.pkl.gz is an array of numpy arrays, divded into unsupervised, train, validation and testing data sets 
[train_set, valid_set, test_set]

data/word_vectors_display.pkl.gz includes all (at this stage 5000 rows of) user data in one array

each row of above contains word counts, normalized so that the total count per row == 1 (except for some empty case exceptions where total count=0)

**VARIATIONS FOR DATA PREP STEP**

**prepareFakeImageInputData.py**
**prepareFakeInputData.py**
**prepareInputData.py**
**prepareTruncatedMnistData.py** 

**all respectively, reading from**
- **data/mnist.pkl.gz** - the original full mnist data set
- **data/user_word_vectors_small.csv** - smaller number of users (768 I think)
- **data/fake_img_data.csv** - a very small set of data representing crosses (in same shape as mnist data)

**and writing to:**
- **data/truncated_mnist.pkl.gz**
in same format as mnist.pkl and same format as  tutorial expects
[[train_set_x,train_set_y],[valid_set_x, valid_set_y],[test_set_x, test_set_y]]
*where the xxx_set_x consists of rows of floats not necessarily adding to 1, (28*28 pixels per row)* and the xxx_set_y consist of a series of ints, from 0,9 representing the correct 'class' for the digit
 - **data/truncated_unsupervised_mnist.pkl.gz**
same as above, except we drop the y values... thus
[train_set_x,valid_set_x, test_set_y]
- **data/fake_img_data.pkl.gz**
same format as above, but much smaller set consisting of variations on crosses

TRAIN STEP
---
**training data (pkl.gz) -> trained weights (pkl.gz)**

*main python file:*
**train_SMH.py**
- reads data/word_vectors.pkl.gz
- writes data/word_vectors_weights.pkl.gz

this is the main algorithm file, network configuration (ie 3000,2000,50,20,50,2000,3000 is hard coded in here)
consists of pre-training step followed by classic neural net gradient descent on training data as an auto-encoder. doesn't use the test_data at all, but uses valid_data to avoid over fitting. lots of parameters in here need to (manually) be made to match the parameters in run_SMH.py

**VARIATIONS ON TRAIN STEP**

**read data/truncated_unsupervised_mnist.pkl.gz
write data/unsuperivsed_mnist_weights.pkl.gz**

when run in this mode also generates png files in the trace directory that give some idea of whether digits after reconstruction actually look like digits (also the first layer weights are interesting to look at)

RUN STEP
---
**trained weights (pkl.gz) 
+ presentation data (pkl.gz) -> coords (csv)**

main python file:
**run_SMH.py**
- reads data/word_vectors_weights.pkl.gz
- reads data/word_vectors_display.pkl.gz
- writes out/coords.csv

this step does two main things
- run the SMH 'forward' on display data generating 'output codes' for each input vector 
- give the 'output codes' to the tsne binary which then 'refines' the tsne alg then output to coords.csv 

one of the crucial parameters of the tsne algorithm is the 'perplexity' which is described as 'optimal number of neighbours'. i have found that low perplexity results in a single big blob but high perplexity results in slightly more islands and variation to the chart

PRESENTATION STEP
---
**coords (csv) + labels (csv) -> visual (java applet)**

**out/VOSviewer.jar**
- loads out/coords.csv
- loads out/5000_user_labels.csv
- generates visual display

Click on VOSViewer.jar and it launches a java app. click 'open map..' then choose the above csv files. set the type to 'coordinates'. Bob is your proverbial. 

if there is no coords.csv yet (because you haven't run the above, see variations below)

**PRESENTATION STEP - VARIATIONS**

layout of *unsupervised* network on the mnist digit recognition problem
**out/mnist_coords.csv
out/mnist_labels.csv**
mnist_labels.csv is created directly from the supervised dataset in the relevant mnist data set (in prepareMnistData.py)

A couple of examples of running on the 5000 user/word vectors with different values for perplexity
**out/coords_perplexity_50.csv**
**out/user_coords_perplexity_5.csv**
