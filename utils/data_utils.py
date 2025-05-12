import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from datetime import datetime

#REGEXES for cleaning patient messages
post_2020_med_rq_regex = 
question_regex = 
questionairre_str =
greeting_str = 
imposter_str = 
flu_str = 
pre_change_med_rq_regex = 
post_2020_delimiter = 
change_date = 


def word_count(text):
    """
    Get the (crude) word count of a sentence
    Input: string
    Output: int
    """
    words = re.findall(r'\b\w+\b', text.lower())
    return len(words)
    
def date_formatter(date_str):
    """
    Normalizes the timestamp if provided in iso format originally
    Input: string
    Output: string
    """
    if type(date_str) == str:
        if 'T' or 'Z' in date_str:
            return datetime.fromisoformat(date_str).strftime("%Y-%m-%d %H:%M:%S")
        else:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S") 

    else:
        return date_str


def cleaning_algo_2015_2020(raw_msg_str, cleaned_set,match_cutoff=50):
    """
    Cleaning threads created before change in Epic formatting of threads
    Input: raw_msg_str: string, cleaned_set: set of strings, match_cutoff: int
    Output: list of strings
    """
    processed_input = raw_msg_str.strip().replace('\\',"").replace("'", "").replace('"',"").replace('\n',"").strip() 
    output = processed_input
    for subsequence in cleaned_set:
        if subsequence != processed_input and len(subsequence) >= match_cutoff:
            output = output.replace(subsequence,'[ZOSBOS]')
            
    return output.split('[ZOSBOS]',1)[0].strip()

def cleaning_algo_2020_2024(raw_msg_str):
    """
    Cleaning threads created after change in Epic formatting of threads
    Input: string
    Output: list of strings
    """

    processed_input = raw_msg_str.strip().replace('\\',"").replace("'", "").replace('"',"").replace('\n',"").strip() 
    output = re.sub(post_2020_delimiter, '[ZOSBOS]',processed_input)
    return output.split('[ZOSBOS]',1)[0].strip()

def process_pat_data(patient_id,patient_data):
    """
    Function to retrieve and clean patient messages.
    Input: patient_id: string, patient_data: list of dicts
    Output: list of dicts
    """

    patient_results = {
        'pat_to_doctor_total_words': 0,
        'pat_to_doctor_total_char_count' : 0,
        'pat_to_doctor_msgs': 0,
        'refined_pat_encounters' :[]
    }

    #Create set of unique messages to clean away duplicate sequences in messages
    cleaning_set = list(set([message['content'].strip().replace('\\',"").replace("'", "").replace('"',"").replace('\n',"").strip() 
                 for encounter in patient_data for message in encounter['messages'] if message['content'] is not None]))

    #Remove the largest sequences first (less likely to be repeated intentionally)
    cleaning_set.sort(key=len,reverse=True) 
    
    for encounter in patient_data:
        enc = {
            'pat_enc_csn_id': encounter['pat_enc_csn_id'],
            'pat_messages' : []
        }
        for message in encounter['messages']:
            if message['from'] == 'patient':
                pat_msg = {
                    'std_message_id' : message['std_message_id'],
                    'timestamp' : date_formatter(message['timestamp']),
                    'speaker' : message['speaker'],
                    'caretaker': False,
                    'subject' : message['subject']
                }
                subjects = set([message['subject']]) if type(message['subject']) == str else set(message['subject'])

                #Check if before or after Epic formatting change (Currently identified as 2020-05-28)

                #if after change, then use post-2020 regex to retrieve patient content
                if datetime.strptime(pat_msg['timestamp'], "%Y-%m-%d %H:%M:%S").date() >= datetime.fromisoformat('2020-05-28').date():
                    if any("Medication Renewal Request".lower() in s.lower().strip(r" \[\]\'")for s in subjects):
                        
                        try:
                            content = re.search(post_2020_med_rq_regex,message['content'],re.MULTILINE | re.IGNORECASE).group().strip()
                        except:
                            continue
                        if len(content) > 0:
                            pat_msg['content'] = content
                        else:
                            continue
                    elif any(questionairre_str.lower() in s.lower() for s in subjects): 
                        continue
                    else:
                        content = cleaning_algo_2020_2024(message['content'])
                        content = re.sub(greeting_str, "",re.sub(r"\s+", " ", content))
                        content = re.sub(flu_str, "", content)
                        pat_msg['content'] = content
                        
                #if before change, then use pre-2020 regex to retrieve patient content
                else:
                    if any("Medication Renewal Request".lower() in s.lower().strip(r" \[\]\'")for s in subjects):
                        try:
                            content = re.search(pre_2020_med_rq_regex,message['content'],re.MULTILINE | re.IGNORECASE).group().strip()
                        except:
                            continue
                        pat_msg['content'] = content
                    elif any(questionairre_str.lower() in s.lower() for s in subjects): 
                        continue
                    else:
                        content = cleaning_algo_2015_2020(message['content'], cleaning_set)
                        content = re.sub(greeting_str, "",re.sub(r"\s+", " ", content))
                        content = re.sub(flu_str, "", content)
                        pat_msg['content'] = content

                #Move on if no content left
                if pat_msg['content'] == '':
                    continue

                #Try to check if the patient content is from a caretaker (need to refine)
                if re.search(imposter_str, pat_msg['content']):
                    pat_msg['caretaker'] = True
               
                #Get message statistics
                pat_msg['word_count'] = word_count(content)
                pat_msg['char_count'] = len(content.replace(" ", ""))
                enc['pat_messages'].append(pat_msg)

            #If nothing left after cleaning, move on
            if enc['pat_messages'] == []:
                continue
            patient_results['refined_pat_encounters'].append(enc)

    #removing duped encs
    final_list = []
    hash_set = set()
    for enc in patient_results['refined_pat_encounters']:
        if enc['pat_enc_csn_id'] not in hash_set:
            hash_set.add(enc['pat_enc_csn_id'])
            final_list.append(enc)
    patient_results['refined_pat_encounters'] = final_list

    #Getting overall stats for patient 
    for enc in patient_results['refined_pat_encounters']:
        patient_results['pat_to_doctor_msgs'] += len(enc['pat_messages'])
        for msg in enc['pat_messages']:
            patient_results['pat_to_doctor_total_words'] += msg['word_count'] 
            patient_results['pat_to_doctor_total_char_count'] += msg['char_count']

    return patient_id, patient_results

def curr_msg_dict(pt_i, enc_i=None, msg_i=None, key=None):
    """
    (Unsafe) Utility function to explore patient conversations
    Input: pt_i: string, enc_i: int, msg_i: int, key: string
    Output: 
    if enc_i == None --> list of dicts
    if msg_i == None --> dict
    if key == None --> list of dicts
    else --> Depends on key
    """

    if enc_i == None:
        return df.iloc[pt_i]['encounters']
    if msg_i == None:
        return df.iloc[pt_i]['encounters'][enc_i]
    if key == None:
        return df.iloc[pt_i]['encounters'][enc_i]['messages'][msg_i]
    return df.iloc[pt_i]['encounters'][enc_i]['messages'][msg_i][key]

def from_patient_msg_filter(msg_list_elem):
    """
    Helper function to filter patient messages from one message
    Input: dict
    Output: dict
    """

    temp_dict = dict(filter(lambda item: item[0] == 'from' and item[1] == 'patient', msg_list_elem.items()))
    if temp_dict:
        temp_dict = temp_dict | dict(filter(lambda item1: item1[0] == 'timestamp' , msg_list_elem.items()))
        return temp_dict

def enc_wide_pt_msg_filter(enc_list_elem):
    """
    Helper function to filter patient messages across a list of messages of one encounter 
    Input: list of dicts
    Output: list of dicts
    """

    return list(map(from_patient_msg_filter, enc_list_elem['messages']))#['messages']))

def parallel_pt_msg_filter(enc_list):
    """
    Function to filter patient messages from one patient (name is misnomer, not in parallel)
    Input: list of dicts
    Output: list of lists
    """

    return list(map(enc_wide_pt_msg_filter,enc_list))

def freq_helper(list_elem):
    """
    Helper function to get general frequency of some list of elements
    Input: list
    Output: double
    """

    return sum(map(lambda x: int(bool(x)), list_elem))/ len(list_elem)

def element_wise_freq(list_of_list):
    """
    Function to get frequency for each list in a list
    Input: list of list
    Output: list of double
    """
    return list(map(freq_helper, list_of_list))

def sum_help(list_elem):
    """
    Helper function to get number of elements in list
    Input: list
    Output: int
    """
    
    return sum(map(lambda x: int(bool(x)), list_elem))

def element_wise_sum(list_of_list):
    """
    Function to get number of elements in each list in a list of list
    Input: list of list
    Output: list of int
    """

    return list(map(sum_help, from_pt_msg_list))


def mci_adrd_filter(diag_df, mci_bool, adrd_bool, logic_type):
    """
    (DO NOT USE) Filter ids based on boolean logic on phenotype
    Input: diag_df: pandas Dataframe, mci_bool: boolean, adrd_bool: boolean, logic_type: string
    Output: set of strings
    """

    if logic_type == 'or':
        return set(diag_df['pat_owner_id'][(diag_df['adrd_tf_or_none'] == adrd_bool) | (diag_df['mci_tf_or_none'] == mci_bool)])
    if logic_type == 'and':
        return set(diag_df['pat_owner_id'][(diag_df['adrd_tf_or_none'] == adrd_bool) & (diag_df['mci_tf_or_none'] == mci_bool)])

def phenotyper(input_df, phenotype_set, key):
    phenotype_df = pd.DataFrame(phenotype_set)
    phenotype_df[key] = True
    phenotype_df.rename(columns={0:'pat_owner_id'},inplace=True)
    phenotype_df.set_index('pat_owner_id',inplace=True)
    input_df[key] = False
    input_df.set_index('pat_owner_id', inplace=True)
    input_df.update(phenotype_df)
    input_df.reset_index(inplace=True)

def calculate_stats(arr):
    """
    Get mean, median, stddev, min, max, and 25%, 50%, 90% percentiles
    Input: list
    Output: dict
    """

    stats = {
        'mean': np.mean(arr),
        'median': np.median(arr),
        'stddev': np.std(arr),
        '25%': np.percentile(arr, 25),
        '75%': np.percentile(arr, 75),
        '90%': np.percentile(arr, 90),
        'max': np.max(arr),
        'min': np.min(arr)
    }
    return stats

def fuzzy_subsequence_match(seq1, seq2, threshold=100):
    """
    Find approximate subsequences in seq1 that match seq2 with a minimum threshold.
    Input: seq1: string, seq2: string, threshold: int
    Output: string
    """
    replace_idx_start = 0
    replace_idx_end = 0
    replacement = ""
    modified_seq = list(seq1)
    curr_highest = 0
    # Check all possible subsequences of seq1
    for i in range(len(seq1) - len(seq2) + 1):
        subseq = seq1[i:i+len(seq2)]
        match_score = fuzz.ratio(subseq, seq2)

        if match_score >= threshold and match_score >= curr_highest:
            replacement = "[ZOSBOS]"
            replace_idx_start = i
            replace_idx_end = len(subseq)
            curr_highest = match_score


    modified_seq[replace_idx_start:replace_idx_start+replace_idx_end] = list(replacement)


    return ''.join(modified_seq)


def extract_phenotyped_messages(enc_list, time, icd_code):
    """
    Extract messages with some icd associated with them
    Input: enc_list: list of dict, time: string
    Output: list of dict
    """

    before = []
    enc_filtered = list(map(lambda d: 1 if icd in d['icds'] else 0, enc_list))
    for i in range(len(enc_filtered)):
        if time == "before" and enc_filtered[i] == 0:
            before.append(enc_list[i])
        elif time == "after" and enc_filtered[i] == 1:
            before.append(enc_list[i])
    return before


def pad_the_end(a_list, b_list):
    """
    Pad list_a to be the the length of list_b based on the last element
    Input: a_list: list, b_list: list
    Output: list
    """

    len_a = len(a_list)
    len_b = len(b_list)
    if(len_a < len_b): #pad a to fit b
        a_list.extend([a_list[-1]]*(len_b-len_a))
        return a_list
    else:
        return a_list

#LEGACY functions, DO NOT USE

def rough_clean_old(raw_msg_str, cleaned_set):
    processed_input = raw_msg_str.strip().replace('\\',"").replace("'", "").replace('"',"").replace('\n',"").lower().strip() 
    output = processed_input
    for subsequence in cleaned_set:
        if subsequence != processed_input: output = output.replace(subsequence,'[ZOSBOS]')
            
    return output.split('[ZOSBOS]',1)[0]

def rough_clean_new(raw_msg_str, cleaned_set,match_cutoff=20):
    processed_input = raw_msg_str.strip().replace('\\',"").replace("'", "").replace('"',"").replace('\n',"").lower().strip() 
    output = processed_input
    for subsequence in cleaned_set:
        if subsequence != processed_input and len(subsequence) > match_cutoff:
            output = output.replace(subsequence,'[ZOSBOS]')
            
    return output.split('[ZOSBOS]',1)[0]
