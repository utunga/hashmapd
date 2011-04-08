
** Tweet Data Flow **
=================

> **Add download request(s) to database** *(twextract/request_queue.py)*
The main method in "twextract/request_queue.py" adds download requests (username+page) for a single username to the queue. If no page is specified, a request will be added for each of the latest cfg.n_pages.
(this could easily be extended to add requests for a list of usernames)

*Download Request Format: uuid -> {username,request_time,doc_type="download_request"}*
("attempts" field added and incremented for each failed download attempt)
("started" field added when download begins)
("completed" field added when download successfully completed)

> **Download and store tweets** *(twextract/download_tweets.py)*
*(NB: This grabs the consumer_token, consumer_secret, session_key, and session_secret from the secrets.cfg file, and authenticates with twitter using oAuth)*
A manager will be initiated which takes the oldest download requests off the queue, and spawns worker threads to: download the relevant tweets and store them in the database, as well as storing the user's info (if it's not already stored). Once all download requests for a given user are completed, a hash request will be created. NB: The program remains open until the user types "exit".

*Tweet Format: "tweet_"+tweetid -> {all_tweet_data,provider_namespace="twitter",doc_type="raw_tweet"}*
*User Format: "twuser_"+username -> {all_username_data,request_time,hash=null,coords=null,provider_namespace="twitter",doc_type="user"}*
*Hash Request Format: uuid -> {username,request_time,provider_namespace="twitter",doc_type="user"}*
("attempts" field added and incremented for each failed download attempt)
("started" field added when download begins)
("completed" field added when download successfully completed)

> **Tokenize and Computer Hashes* *(twextract/compute_hashes.py)*
A manager will be initiated which takes the oldest hash requests off the queue, and spawns worker threads that: tokenize the user's tweet data (done in a view), compute word histograms by combining this information with the words list used by the SMH, then feed the histograms through the SMH to get a hash code (the tSNE algorithm is also run to produce co-ordinates, but this doesn't work at the moment). The hash is then stored in the database. NB: The program remains open until the user types "exit".


> The **tweepy** library which is used to download information from twitter is currently stored inside the twextract folder.
> There are several tests in the test folder that should complete successfully.


** SMH Data Flow **
===============

The .cfg files specify the parameters used when preparing, training, and running the data. The parameters in the specified cfg file are loaded, and any missing paramters are obtained from the base.cfg file.

> **Prepare Data** *(prepare#.py)*
Prepares the relevant dataset for training. The data will be stored in the following files (pickled):

1) cfg.input.train_data_info:
training_file_name (not including index or file extension), number_training_files, total_no_training_batches,
validation_file_name (not including index or file extension), number_validation_files, total_no_validation_batches,
testing_file_name (not including index or file extension), number_testing_files, total_no_testing_batches,
batches_per_file (may be less in last file), mean_doc_size

2) data_files:
normalized_document_word_counts (or raw_counts if using poisson layer), document_sums (all one, unless using poisson layer), labels (may be blank)

> **Train** *(train.py)*
Loads in the specified config file, which loads the relevant data files (all into memory at once if there is only one training file, or one file at a time per epoch if the data is split over several files) and performs pre-training (RBMs/deep belief net) and (after unrolling the network into a stacked auto-encoder) fine-tuning (gradient descent) on the data, before saving the weights (cfg.train.weights_file).

Note the new config options:
- train.method = 'cd' or 'pcd'
- train.k = number of iterations of cd/pcd
- first_layer_type = 'bernoulli' or 'poisson' (the poisson layer is still not working properly :/)
- noise_std_dev = how much gaussian noise to add to codes during fine tuning - 0.0 for none (untested)
- cost = 'squared_diff' or 'cross_entropy' (cross entropy is untested)

*(i have found that low perplexity results in a single big blob but high perplexity results in slightly more islands and variation to the chart #MKT)*

> **Run** *(run.py)*
Loads in the specified config file, which loads the relevant weights to reproduce the SMH, as well as the **first** file of training data. This data is then fed forward through the SMH to produce hashes which are saved (cfg.output.codes_file), before calling the tSNE.py file (a slowish python implementation of tSNE) to produce co-ordinates which are also saved (cfg.output.coords_file). The prepared labels are also saved in a format that is readable by the VOSViewer (cfg.output.labels_file). NB: The standard tSNE algorithm is very memory intensive and can't handle more than around 5,000 elements of data at once.

> **View** *(out/vosviewer)*
You can run the VOSviewer.jar/jnlp app and select the coords.csv and (optionally) the labels.csv files to view the final output (2d co-ordinates).





** Datasets **
==========

**prepareReutersData.py** > see reuters.cfg
**prepareTruncatedMnistData.py** > see unsupervised_mnist.cfg
**prepare.py** > see base.cfg

(the other prepare files probably aren't updated to match the expected new multi-file format)

