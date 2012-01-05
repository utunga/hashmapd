from hashmapd.token_counts import TokenCounts
data_dir = 'projects/word_vectors' #'amdata'
data_dir = 'amdata'
truncate_at_num_users = 200
truncate_at_num_tokens = 400

# load all data
data = TokenCounts(file_prefix='inc_',data_dir = data_dir)
data.load_from_csv()

# define a truncate function
def truncate_user_filter(user):
    return (data.user_ids[user]<truncate_at_num_users)

def truncate_token_filter(token):
    return (data.token_ids[token]<truncate_at_num_tokens)

# this is a really crappy way to do things but anyway
data.write_to_csv(file_prefix='truncated_', token_filter_fun=truncate_token_filter, user_filter_fun=truncate_user_filter)
data.load_from_csv()

data.write_to_training_csv()
