from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.data_utils import *
from tqdm import tqdm
tqdm.pandas()

conversation_path = '/gpfs/data/mankowskilab/NYU_Consult_raw/merged_data_2015_2024.json'
if __name__ == "__main__":
    conversations = pd.read_json(conversation_path)
    conversations = conversations[['pat_owner_id','encounters']]
    result = defaultdict(lambda: defaultdict(int))
    tasks = [(row['pat_owner_id'], row['encounters']) for _, row in conversations.iterrows()]
    chunk_size = 100000

    with ProcessPoolExecutor(max_workers=64) as executor:
    
        for chunk in chunk_tasks(tasks, chunk_size):
            future_to_patient = {executor.submit(process_pat_data, patient_id, patient_data): patient_id for patient_id, patient_data in chunk}
            
            for future in tqdm(as_completed(future_to_patient), total=len(future_to_patient)):
                patient_id, patient_results = future.result()
                for key, value in patient_results.items():
                    result[patient_id][key] = value
    
    
    patients = list(result.keys())
    pat_to_doctor_total_words = [result[patient]['pat_to_doctor_total_words'] for patient in patients]
    pat_to_doctor_msgs = [result[patient]['pat_to_doctor_msgs'] for patient in patients]
    
    crude_word_df = pd.DataFrame({key: dict(value) for key, value in result.items()}).transpose().reset_index().rename(columns={'index':'pat_owner_id', 'refined_encounters':'refined_pat_encounters'})
    crude_word_df.to_json('./data/refined_pat_msgs_new_algo.json')
