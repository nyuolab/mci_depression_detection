test_demo = current_demo.groupby('pat_owner_id').agg({
                                              'birthdate' : list,
                                              'ageinyears' : list,
                                              'ethnicity' : list,
                                              'firstrace' : list,
                                              })

test_demo['ceil_index'] = test_demo['ageinyears'].apply(lambda x: np.argmax(x))
test_demo['birthdate'] = test_demo.apply(lambda row: row['birthdate'][row['ceil_index']],axis=1)
test_demo['ageinyears'] = test_demo.apply(lambda row: row['ageinyears'][row['ceil_index']],axis=1)
test_demo['firstrace'] = test_demo.apply(lambda row: row['firstrace'][row['ceil_index']],axis=1)
test_demo['ethnicity'] = test_demo.apply(lambda row: row['ethnicity'][row['ceil_index']],axis=1)
test_demo.drop('ceil_index',axis=1,inplace=True)


df['birthdate'] = None
df['ageinyears'] = None
df['ethnicity'] = None
df['firstrace'] = None

df.set_index('pat_owner_id',inplace=True)
df.update(test_demo)
df.reset_index(inplace=True)

df.set_index('pat_owner_id',inplace=True)
len_msgs_df.set_index('pat_owner_id',inplace=True)
df['len_msgs'] = 0
df.update(len_msgs_df)
df.reset_index(inplace=True)
df['average_len_msgs'] = df['len_msgs'] / df['total_msgs_from_pt']
