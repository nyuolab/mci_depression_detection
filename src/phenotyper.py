import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
#Phenotyping script using ICD-10 codes
#Current method: Check for existence of ICD-10 codes associated with MCI

#DATA_PATHS TODO: REMOVE for PUBLIC RELEASE
pt_msg_path = "./data/pat_msgs.json"
icd_subquery_1_path = 
icd_subquery_2_path = 
nyuqa_demo_path = 
diag_data_path = 
scraped_icds_path = 
qa_meds_path = 
yifan_demo_path = 

#regexes used TODO: Allow for generalized regex
#diagnosis code = 
adrd_regex = r"(G30|F0[1-4])\.[^,]|(G23\.1)|(G31\.(0[19]|83|9))"
mci_regex = r"(G31\.(1|8[4|5]))"

#Load patient message data to be phenotyped
#df should be changed to pt_msgs
pt_msgs = pd.read_json(pt_msg_path)

#Load all pulled data sources
icd_subquery_1 = pd.read_csv(icd_subquery_1_path) #Pulled icd data from Xu

icd_subquery_2 = pd.read_csv(icd_subquery_2_path) #Pulled icd data from Xu

nyuqa_demo = pd.read_csv(nyuqa_demo_path) #Pulled demo data from Xu
nyuqa_demo.drop(['Unnamed: 0', 'age 40_60', 'age 80 above'],axis=1, inplace=True)

diag_data = pd.read_csv(diag_data_path) #Pulled diag data from Xu
diag_data.drop(["Unnamed: 0.1","Unnamed: 0"],axis=1)

yifan_meds = pd.read_csv(yifan_demo_path) #Newly requested demo data by yifan

pt_icds_2020_2024 = pd.read_json(scraped_icds_path) #Scraped icds from original nyuqa dataset
pt_icds_2020_2024 = pt_icds_2020_2024.transpose()

#Updating phenotype using all pulled data, starting with scraped icd data 
#Using scraped ICDs to phenotype
pt_icds_2020_2024['mci_codes'] = pt_icds_2020_2024.encounters.apply(lambda x: [s for s in sum([ d['concatenated'] for d in x],[]) if re.search(mci_regex, s)])
pt_icds_2020_2024['adrd_codes'] = pt_icds_2020_2024.encounters.apply(lambda x: [s for s in sum([ d['concatenated'] for d in x],[]) if re.search(adrd_regex, s)])
mci_cohort = pt_icds_2020_2024[(pt_icds_2020_2024.mci_codes.apply(len) != 0) | (pt_icds_2020_2024.adrd_codes.apply(len) != 0)]

#Finding earliest date of diagnosis for later use
mci_cohort['earliest_date_mci'] = mci_cohort.mci_codes.apply( lambda x: min([datetime.strptime(s.split('[ZOSBOS]')[0], "%Y-%m-%d") for s in x]) if len(x) != 0 else x)

mci_cohort['earliest_date_adrd'] = mci_cohort.adrd_codes.apply( lambda x: min([datetime.strptime(s.split('[ZOSBOS]')[0], "%Y-%m-%d") for s in x]) if len(x) != 0 else x)

mci_cohort = mci_cohort[['pat_owner_id','earliest_date_mci','earliest_date_adrd']]
mci_cohort['earliest_sign'] = mci_cohort.apply(lambda x: min(x['earliest_date_mci'], x['earliest_date_adrd']) if type(x['earliest_date_adrd']) == type(x['earliest_date_mci']) 
                                               else x['earliest_date_adrd'] if type(x['earliest_date_adrd']) != list else x['earliest_date_mci'],axis=1)
mci_cohort = mci_cohort[['pat_owner_id','earliest_sign']]


#Yifan data: Get only patients with adrd or mci codes
yifan_meds = yifan_meds[yifan_meds.dx_current_icd10_list.str.contains(adrd_regex) | yifan_meds.dx_current_icd10_list.str.contains(mci_regex)]
yifan_meds = yifan_meds[['pat_id','dx_contact_date']]
yifan_meds = yifan_meds.groupby('pat_id').agg(list).reset_index().rename({'dx_contact_date': 'earliest_sign'},axis=1)
yifan_meds.earliest_sign = yifan_meds.earliest_sign.apply(lambda x: min([datetime.strptime(str(d),"%Y-%m-%d") for d in x]))
yifan_meds.rename({'pat_id':'pat_owner_id'},axis=1,inplace=True)

#Pulled data from Xu: Updating phenotypes
icd_subquery_1['mci_tf'] = icd_subquery_1['icd'].str.contains(mci_regex).fillna(False)
icd_subquery_1['adrd_tf'] = icd_subquery_1['icd'].str.contains(adrd_regex).fillna(False)

icd_subquery_2['mci_tf'] = icd_subquery_2['icd'].str.contains(mci_regex).fillna(False)
icd_subquery_2['adrd_tf'] = icd_subquery_2['icd'].str.contains(adrd_regex).fillna(False)

diag_data['mci_tf'] = diag_data['icd'].str.contains(mci_regex).fillna(False)
diag_data['adrd_tf'] = diag_data['icd'].str.contains(adrd_regex).fillna(False)

#Filtering for mci patients and getting raw times
icd_subquery_1['mci'] = icd_subquery_1.apply(lambda x: 1 if (x.mci_tf == True or x.adrd_tf== True) else 0,axis=1)
icd_subquery_1 = icd_subquery_1[icd_subquery_1.mci == 1][['pat_owner_id','startdatekey']]
icd_subquery_1.rename({'startdatekey':'earliest_sign'},axis=1,inplace=True)

icd_subquery_2['mci'] = icd_subquery_2.apply(lambda x: 1 if (x.mci_tf == True or x.adrd_tf == True) else 0,axis=1)
icd_subquery_2 = icd_subquery_2[icd_subquery_2.mci == 1][['pat_owner_id','startdatekey']]
icd_subquery_2.rename({'startdatekey':'earliest_sign'},axis=1,inplace=True)

diag_data['mci'] = diag_data.apply(lambda x: 1 if (x.mci_tf == True or x.adrd_tf == True) else 0,axis=1)
diag_data = diag_data[diag_data.mci == 1][['pat_owner_id','diagnosis_time']]
diag_data = diag_data.groupby('pat_owner_id').agg(list).reset_index().rename({'diagnosis_time': 'earliest_sign'},axis=1)

#Normalizing times
diag_data.earliest_sign = diag_data.earliest_sign.apply(lambda x: min([datetime.strptime(str(d),"%Y-%m-%d") for d in x]))

icd_subquery_1 = icd_subquery_1.groupby('pat_owner_id').agg(list).reset_index()
icd_subquery_1.earliest_sign = icd_subquery_1.earliest_sign.apply(lambda x: [d for d in x if d != -1])
icd_subquery_1.earliest_sign = icd_subquery_1.earliest_sign.apply(lambda x: min([datetime.strptime(str(d),"%Y%m%d") for d in x]) if len(x) != 0 else datetime(1970, 1,1))

icd_subquery_2 = icd_subquery_2.groupby('pat_owner_id').agg(list).reset_index()
icd_subquery_2.earliest_sign = icd_subquery_2.earliest_sign.apply(lambda x: [d for d in x if d != -1])
icd_subquery_2.earliest_sign = icd_subquery_2.earliest_sign.apply(lambda x: min([datetime.strptime(str(d),"%Y%m%d") for d in x]) if len(x) != 0 else datetime(1970, 1,1))

#Find first time of diagnosis among all provided data
all_times = pd.concat([icd_subquery_1,icd_subquery_2,diag_data,yifan_meds,mci_cohort])
all_times = all_times.groupby('pat_owner_id').agg(lambda x: min(list(x))).reset_index()

#Output phenotype pt_ids with associated first times of diagnosis
all_times.to_csv('./data/mci_cohort_dates.csv')
