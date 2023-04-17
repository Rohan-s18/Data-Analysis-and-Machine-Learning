import numpy as np
import pandas
import math

import pubchempy
#from tdc.multi_pred import DDI
import pandas as pd
import pickle
import pubchempy as pcp
#from rdkit import Chem
#from rdkit.Chem import AllChem
#from rdkit import DataStructs
#from tdc.multi_pred import DTI
import re


def build_x_tensor():
    dd_disease_data = pd.read_csv('/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersection_pairs_dd_disease.csv')
    drug_data = pd.read_csv('/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/intersections.csv')
    all_cell_line_ls = dd_disease_data['cell_line'].tolist()
    cell_line_ls = remove_duplicate(all_cell_line_ls)

    len_cell_line = len(cell_line_ls)
    pair_num = len(dd_disease_data[dd_disease_data['classification'] == 'synergy'].index)
    drug_num = len(drug_data.index)

    result = {}
    for index, cell_line in enumerate(cell_line_ls):
        result[str(index) + ' ' + cell_line] = build_slice_x('cell_line', cell_line, dd_disease_data, drug_data)

    with open('./useful_data/tensor_x.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('total number of drugs: {}'.format(drug_num))
    print('number of target cell line: {}'.format(len_cell_line))
    print('total valid pair of drug disease: {}'.format(pair_num))
    print('finish building tensor for drug drug disease!, sparsity: {}'.format(pair_num/(drug_num*drug_num*len_cell_line)))


def build_slice_x(slice_col_name, slice_name, tensor_data, drug_data):
    len_drug = len(drug_data.index)
    drug_names = drug_data['name'].tolist()
    result = pd.DataFrame(np.zeros((len_drug, len_drug), dtype=int), columns=drug_names, index=drug_names)
    target = tensor_data[tensor_data[slice_col_name] == slice_name]
    for index, row in target.iterrows():
        drug1 = row['Drug1_name']
        drug2 = row['Drug2_name']
        classification = row['classification']
        if classification == 'synergy':
            result[drug1][drug2] = 1

    return result


def build_y_tensor():
    ddi_data = pd.read_csv('./useful_data/intersection_pairs_ddi.csv')
    drug_data = pd.read_csv('./useful_data/intersections.csv')
    all_ddi_type_ls = ddi_data['Y'].tolist()
    ddi_type_ls = remove_duplicate(all_ddi_type_ls)

    ddi_type_num = len(ddi_type_ls)
    pair_num = len(ddi_data.index)
    drug_num = len(drug_data.index)

    result = {}
    for index, ddi_type in enumerate(ddi_type_ls):
        result[str(index) + ' ' + str(ddi_type)] = build_slice_y('Y', ddi_type, ddi_data, drug_data)

    with open('./useful_data/tensor_y.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('total number of drugs: {}'.format(drug_num))
    print('number of ddi types: {}'.format(ddi_type_num))
    print('total valid pair of drugs combo: {}'.format(pair_num))
    print('finish building tensor for DDI!, sparsity: {}'.format(pair_num / (drug_num * drug_num * ddi_type_num)))


def build_slice_y(slice_col_name, slice_name, tensor_data, drug_data):
    len_drug = len(drug_data.index)
    drug_names = drug_data['name'].tolist()
    result = pd.DataFrame(np.zeros((len_drug, len_drug), dtype=int), columns=drug_names, index=drug_names)
    target = tensor_data[tensor_data[slice_col_name] == slice_name]
    for index, row in target.iterrows():
        drug1 = row['Drug1_name']
        drug2 = row['Drug2_name']
        result[drug1][drug2] = 1

    return result


def save_ddi_files_to_csv():
    data = DDI(name='DrugBank').get_data()
    data.to_csv('./useful_data/ddi_all.csv')


def find_intersection():
    ddi_dictionary = pd.read_csv('./useful_data/unique_ddi_base_dictionary.csv').dropna()
    dd_disease_dictionary = pd.read_csv('./useful_data/unique_dd_disease_base_dictionary.csv').dropna()
    len_ddi = len(ddi_dictionary.index)
    len_dd_disease = len(dd_disease_dictionary.index)
    intersection_count = 0
    names = []
    drugbank_id = []
    inchi = []
    inchikey = []
    smiles = []
    search_area = ddi_dictionary['inchikey'].tolist()
    for index, row in dd_disease_dictionary.iterrows():
        if row['inchikey'] in search_area:
            names.append(row['name'])
            drugbank_id.append(ddi_dictionary.loc[ddi_dictionary['inchikey']==row['inchikey']]['drugbank_id'].tolist()[0])
            inchi.append(row['inchi'])
            inchikey.append(row['inchikey'])
            smiles.append(row['smile'])
            print('find intersection for {}'.format(row['name']))
            intersection_count += 1

    result_dict = {'name': names,
                   'drugbank_id': drugbank_id,
                   'inchi': inchi,
                   'inchikey': inchikey,
                   'smile': smiles}

    result = pd.DataFrame(result_dict)
    result.to_csv('./useful_data/intersections.csv')
    print('there are {} unique drugs for ddi pairs'.format(len_ddi))
    print('there are {} unique drugs for drug drug disease paris'.format(len_dd_disease))
    print('there are {} intersected unique drugs'.format(intersection_count))


def find_intersection_ddi():
    intersection_base_data = pd.read_csv('./useful_data/intersections.csv')
    ddi_data = pd.read_csv('./useful_data/ddi_all.csv')
    len_ddi = len(ddi_data.index)
    num_intersected = 0

    drug1_names = []
    drug1_inchis = []
    drug1_inchikeys = []
    drug1_smiles = []

    drug2_names = []
    drug2_inchis = []
    drug2_inchikeys = []
    drug2_smiles = []

    y = []

    seen = []
    search_area = intersection_base_data['drugbank_id'].tolist()
    for index, row in ddi_data.iterrows():
        drug1_id = row['Drug1_ID']
        drug2_id = row['Drug2_ID']
        if drug1_id in search_area and drug2_id in search_area:
            if not seen_pair_detector(seen, [drug1_id, drug2_id]):
                # define drug1
                drug1_row = intersection_base_data.loc[intersection_base_data['drugbank_id'] == drug1_id]
                drug1_name = drug1_row['name'].tolist()[0]
                drug1_inchi = drug1_row['inchi'].tolist()[0]
                drug1_inchikey = drug1_row['inchikey'].tolist()[0]
                drug1_smile = drug1_row['smile'].tolist()[0]

                drug1_names.append(drug1_name)
                drug1_inchis.append(drug1_inchi)
                drug1_inchikeys.append(drug1_inchikey)
                drug1_smiles.append(drug1_smile)

                # define drug2
                drug2_row = intersection_base_data.loc[intersection_base_data['drugbank_id'] == drug2_id]
                drug2_name = drug2_row['name'].tolist()[0]
                drug2_inchi = drug2_row['inchi'].tolist()[0]
                drug2_inchikey = drug2_row['inchikey'].tolist()[0]
                drug2_smile = drug2_row['smile'].tolist()[0]

                drug2_names.append(drug2_name)
                drug2_inchis.append(drug2_inchi)
                drug2_inchikeys.append(drug2_inchikey)
                drug2_smiles.append(drug2_smile)

                # define ddi type
                y.append(row['Y'])

                # add pair to seen
                seen.append([drug1_id, drug2_id])

                # keep track of the count of intersection
                num_intersected += 1

                print('{} and {} pair intersected!'.format(drug1_name, drug2_name))

    result_dict = {'Drug1_name': drug1_names,
                   'Drug1_inchi': drug1_inchis,
                   'Drug1_inchikey': drug1_inchikeys,
                   'Drug1_smile': drug1_smiles,
                   'Drug2_name': drug2_names,
                   'Drug2_inchi': drug2_inchis,
                   'Drug2_inchikey': drug2_inchikeys,
                   'Drug2_smile': drug2_smiles,
                   'Y': y}
    # 65 DDI type
    result = pd.DataFrame(result_dict)
    result.to_csv('./useful_data/intersection_pairs_ddi.csv')
    print('in DDI data set, there are {} pairs of drugs'.format(len_ddi))
    print('{} pairs of drugs are intersected with the drug drug disease'.format(num_intersected))


def find_intersection_dd_disease():
    intersection_base_data = pd.read_csv('./useful_data/intersections.csv')
    dd_disease_data = pd.read_csv('./useful_data/Syner&Antag_voting.csv')
    len_dd_disease = len(dd_disease_data.index)
    num_intersected = 0

    drug1_names = []
    drug1_inchis = []
    drug1_inchikeys = []
    drug1_smiles = []

    drug2_names = []
    drug2_inchis = []
    drug2_inchikeys = []
    drug2_smiles = []

    cell_lines = []

    classifications = []

    seen = []
    search_area = intersection_base_data['name'].tolist()
    for index, row in dd_disease_data.iterrows():
        drug1_name = row['Drug1']
        drug2_name = row['Drug2']
        if drug1_name in search_area and drug2_name in search_area:
            if not seen_pair_detector(seen, [drug1_name, drug2_name]):
                # define drug1
                drug1_row = intersection_base_data.loc[intersection_base_data['name'] == drug1_name]
                drug1_name = drug1_row['name'].tolist()[0]
                drug1_inchi = drug1_row['inchi'].tolist()[0]
                drug1_inchikey = drug1_row['inchikey'].tolist()[0]
                drug1_smile = drug1_row['smile'].tolist()[0]

                drug1_names.append(drug1_name)
                drug1_inchis.append(drug1_inchi)
                drug1_inchikeys.append(drug1_inchikey)
                drug1_smiles.append(drug1_smile)

                # define drug2
                drug2_row = intersection_base_data.loc[intersection_base_data['name'] == drug2_name]
                drug2_name = drug2_row['name'].tolist()[0]
                drug2_inchi = drug2_row['inchi'].tolist()[0]
                drug2_inchikey = drug2_row['inchikey'].tolist()[0]
                drug2_smile = drug2_row['smile'].tolist()[0]

                drug2_names.append(drug2_name)
                drug2_inchis.append(drug2_inchi)
                drug2_inchikeys.append(drug2_inchikey)
                drug2_smiles.append(drug2_smile)

                # cell line and classification
                cell_lines.append(row['Cell line'])
                classifications.append(determine_classification(drug1_name, drug2_name, dd_disease_data))

                # add pair to seen
                seen.append([drug1_name, drug2_name])

                # keep track of the count of intersection
                num_intersected += 1

                print('{} and {} pair intersected!'.format(drug1_name, drug2_name))

    result_dict = {'Drug1_name': drug1_names,
                   'Drug1_inchi': drug1_inchis,
                   'Drug1_inchikey': drug1_inchikeys,
                   'Drug1_smile': drug1_smiles,
                   'Drug2_name': drug2_names,
                   'Drug2_inchi': drug2_inchis,
                   'Drug2_inchikey': drug2_inchikeys,
                   'Drug2_smile': drug2_smiles,
                   'cell_line': cell_lines,
                   'classification': classifications}

    result = pd.DataFrame(result_dict)
    result.to_csv('./useful_data/intersection_pairs_dd_disease.csv')
    print('in drug drug disease data set, there are {} pairs of drugs'.format(len_dd_disease))
    print('{} pairs of drugs are intersected with the DDI dataset'.format(num_intersected))


def determine_classification(drug1, drug2, df):
    target = df.groupby(['Drug1', 'Drug2'])
    target_1 = target.get_group((drug1, drug2))
    target_ls = target_1['classification'].tolist()
    synergy_count = 0
    antagonism_count = 0
    for classification in target_ls:
        if classification == 'synergy':
            synergy_count += 1
        else:
            antagonism_count += 1

    if synergy_count > antagonism_count:
        return 'synergy'
    else:
        return 'antagonism'


def seen_pair_detector(seen_ls, target_ls):
    target1 = target_ls[0]
    target2 = target_ls[1]
    flag = False
    for pair in seen_ls:
        if target1 in pair and target2 in pair:
            flag = True
            return flag
    return flag


def build_dd_disease_base_df():
    data = pd.read_csv('./useful_data/Syner&Antag_voting.csv')
    all_name = data['Drug1'].tolist() + data['Drug2'].tolist()
    unique_names = remove_duplicate(all_name)
    total_drug_num = len(unique_names)
    success_count = 0
    total_time_out_num = 0
    total_bad_request_num = 0
    total_server_busy_num = 0
    names = []
    inchi = []
    inchikey = []
    smiles = []
    for name in unique_names:
        keep_match = True
        while keep_match:
            try:
                compounds = pcp.get_compounds(str(name), 'name')
                keep_match = False
            except pubchempy.BadRequestError as br:
                print(br)
                print('bad request happens matching {} \'s name'.format(name))
                total_bad_request_num += 1
                keep_match = False
            except pubchempy.TimeoutError as te:
                print('time out happens matching {} \'s name, trying again'.format(name))
                total_time_out_num += 1
            except pubchempy.PubChemHTTPError as sb:
                print('server busy matching {} \'s name, trying again'.format(name))
                total_server_busy_num += 1
            except TypeError as type_error:
                print(type_error)
                print(name)
                print('type error found matching {}'.format(name))
                keep_match = False

        if len(compounds) != 0 and name:
            compound = compounds[0]
            names.append(name)
            inchi.append(compound.inchi)
            inchikey.append(compound.inchikey)
            smiles.append(compound.isomeric_smiles)
            print('match!  {}'.format(name))
            success_count += 1

    result_dict = {'name': names,
                   'inchi': inchi,
                   'inchikey': inchikey,
                   'smile': smiles}
    result = pd.DataFrame(result_dict)
    result.to_csv('./useful_data/unique_dd_disease_base_dictionary.csv')

    print('all process finished!')
    print('out of {} drugs from drug drug disease, total of {} drugs are added to dictionary'.format(total_drug_num,
                                                                                                     success_count))
    print('in the processing, {} times of bad request error happened'.format(total_bad_request_num))
    print('in the processing, {} time of time out request error happened'.format(total_time_out_num))
    print('in the processing, {} time of server busy errors happened'.format(total_server_busy_num))





def build_ddi_base_df():
    data = pd.read_csv('./useful_data/ddi_all.csv')
    base_data = pd.read_csv('./data/base_dictionary.csv')
    all_id = data['Drug1_ID'].tolist() + data['Drug2_ID'].tolist()
    unique_ids = remove_duplicate(all_id)
    total_drugs_ddi = len(unique_ids)
    success_count = 0
    name = []
    drugbank_id = []
    inchi = []
    inchikey = []
    smiles = []
    for drug_bank_id in unique_ids:
        target = base_data.loc[base_data['drugbank_id'] == drug_bank_id]
        if not target.empty:
            name.append(target['name'].to_list()[0])
            drugbank_id.append(target['drugbank_id'].to_list()[0])
            inchi.append(target['inchi'].to_list()[0])
            inchikey.append(target['inchikey'].to_list()[0])
            smiles.append(target['smile'].to_list()[0])
            print('match!  {}'.format(target['name'].to_list()[0]))
            success_count += 1

    print('out of {} unique drugs in DDI, {} are successfully identified'.format(total_drugs_ddi, success_count))
    result_dict = {'name': name,
                   'drugbank_id': drugbank_id,
                   'inchi': inchi,
                   'inchikey': inchikey,
                   'smile': smiles}
    result = pd.DataFrame(result_dict)
    result.to_csv('./useful_data/unique_ddi_base_dictionary.csv')


def build_base_dataframe():
    data = pd.read_csv('./data/drugbank.tsv', sep='\t')
    total_num_drugs = len(data.index)
    total_time_out_num = 0
    total_bad_request_num = 0
    total_server_busy_num = 0
    success_count = 0
    print(data.columns)
    drugbank_id = []
    cid = []
    name = []
    types = []
    groups = []
    inchi = []
    inchikey = []
    smile = []
    for index, row in data.iterrows():
        if not pd.isnull(row['inchi']):
            compound_inchi = row['inchi']
            keep_match = True
            while keep_match:
                try:
                    compounds = pcp.get_compounds(compound_inchi, 'inchi')
                    keep_match = False
                except pubchempy.BadRequestError as br:
                    print(br)
                    print('bad request happens matching {} \'s INCHI'.format(row['name']))
                    total_bad_request_num += 1
                    keep_match = False
                except pubchempy.TimeoutError as te:
                    print('time out happens matching {} \'s INCHI, trying again'.format(row['name']))
                    total_time_out_num += 1
                except pubchempy.PubChemHTTPError as sb:
                    print('server busy matching {} \'s INCHI, trying again'.format(row['name']))
                    total_server_busy_num += 1

        else:
            keep_match = True
            while keep_match:
                try:
                    compounds = pcp.get_compounds(row['name'], 'name')
                    keep_match = False
                except pubchempy.BadRequestError as br:
                    print(br)
                    print('bad request happens matching {} \'s name'.format(row['name']))
                    total_bad_request_num += 1
                    keep_match = False
                except pubchempy.TimeoutError as te:
                    print('time out happens matching {} \'s name, trying again'.format(row['name']))
                    total_time_out_num += 1
                except pubchempy.PubChemHTTPError as sb:
                    print('server busy matching {} \'s name, trying again'.format(row['name']))
                    total_server_busy_num += 1

        if len(compounds) != 0:
            compound = compounds[0]
            cid.append(compound.cid)
            drugbank_id.append(row['drugbank_id'])
            name.append(row['name'])
            groups.append(row['groups'])
            types.append(row['type'])
            inchi.append(compound.inchi)
            inchikey.append(compound.inchikey)
            smile.append(compound.isomeric_smiles)
            print('match!  {}'.format(row['name']))
            success_count += 1

    result_dict = {'drugbank_id': drugbank_id,
                   'cid': cid,
                   'name': name,
                   'group': groups,
                   'type': types,
                   'inchi': inchi,
                   'inchikey': inchikey,
                   'smile': smile}
    result = pd.DataFrame(result_dict)
    result.to_csv('./data/base_dictionary.csv')

    print('all process finished!')
    print('out of {} drugs from drug bank, total of {} drugs are added to dictionary'.format(total_num_drugs,
                                                                                             success_count))
    print('in the processing, {} times of bad request error happened'.format(total_bad_request_num))
    print('in the processing, {} time of time out request error happened'.format(total_time_out_num))
    print('in the processing, {} time of server busy errors happened'.format(total_server_busy_num))


def remove_duplicate(ls):
    seen = set()
    result = []
    for element in ls:
        if element not in seen:
            seen.add(element)
            result.append(element)
    return result


def generate_side_info_tanimoto():
    data = pd.read_csv('./useful_data/intersections.csv')
    drug_names = data['name']
    num_drugs = (len(data.index))
    result = np.zeros((num_drugs, num_drugs), dtype=float)
    for i in range(num_drugs):
        for j in range(i, num_drugs):
            drug1_smile = data.iloc[i]['smile']
            drug2_smile = data.iloc[j]['smile']
            distance = tanimoto_calc(drug1_smile, drug2_smile)
            result[i, j] = distance
            result[j, i] = distance
        print('{} th drugs\' pairs finished'.format(i))

    result_df = pd.DataFrame(result, columns=drug_names)
    result_df.to_csv('./useful_data/si_0.csv')


def tanimoto_calc(smi1, smi2):
    if smi1 == 'NA' or smi2 == 'NA' or pd.isna(smi1) or pd.isna(smi2):
        return -1
    else:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
        s = round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)
        return s


def print_matrix_sparsity():
    with open('./useful_data/tensor_x.pickle', 'rb') as tensor_x:
        x = pickle.load(tensor_x)
        total_sparsity_x = []
        for df in x.values():
            sparr_x= df.apply(pd.arrays.SparseArray)
            total_sparsity_x.append(sparr_x.sparse.density)
        print('sparsity of x tensor: {}'.format(sum(total_sparsity_x)/len(total_sparsity_x)))

    with open('./useful_data/tensor_y.pickle', 'rb') as tensor_y:
        y = pickle.load(tensor_y)
        total_sparsity_y = []
        for df in y.values():
            sparr_y = df.apply(pd.arrays.SparseArray)
            total_sparsity_y.append(sparr_y.sparse.density)
        print('sparsity of y tensor: {}'.format(sum(total_sparsity_y)/len(total_sparsity_y)))


def generate_side_info_sider_distance_prep():
    side_effects = pd.read_csv('./data/side-effects.tsv', sep='\t')
    side_effects_term = pd.read_csv('./data/side-effect-terms.tsv', sep='\t')
    intersection = pd.read_csv('./useful_data/intersections.csv')
    side_effect_one_hot = []
    drug_bank_id = intersection['drugbank_id'].values.tolist()
    for drug_id in drug_bank_id:
        side_effect_array = np.zeros(len(side_effects_term.index), dtype=int).tolist()
        candidates = side_effects[side_effects['drugbank_id'] == drug_id]['umls_cui_from_meddra'].values.tolist()
        for side_id in candidates:
            side_index = side_effects_term[side_effects_term['umls_cui_from_meddra'] == side_id].index.tolist()
            if len(side_index) > 1:
                print('error duplicate side id')
            side_effect_array[side_index[0]] = 1
        side_effect_one_hot.append(side_effect_array)
    side_info_dict = {'drugbank_id': drug_bank_id,
                      'side_effect_one_hot': side_effect_one_hot}
    with open('./useful_data/side_effect_dictionary.pickle', 'wb') as handle:
        pickle.dump(side_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_side_info_side_effect():
    data = pd.read_csv('./useful_data/intersections.csv')
    with open('./useful_data/side_effect_dictionary.pickle', 'rb') as handle:
        side_info = pickle.load(handle)
    side_effect_one_hot = side_info['side_effect_one_hot']
    drug_names = data['name']
    num_drugs = (len(data.index))
    result = np.zeros((num_drugs, num_drugs), dtype=float)
    for i in range(num_drugs):
        for j in range(i, num_drugs):
            denominater = np.dot(side_effect_one_hot[i], side_effect_one_hot[i])
            drug1_data = side_effect_one_hot[i]
            drug2_data = side_effect_one_hot[j]
            distance = np.dot(drug1_data, drug2_data) / denominater
            result[i, j] = distance
            result[j, i] = distance
        print('{} th drugs\' pairs finished'.format(i))

    result_df = pd.DataFrame(result, columns=drug_names)
    result_df.to_csv('./useful_data/si_1.csv')


def generate_side_info_target_sequence_prep():
    intersection = pd.read_csv('./useful_data/intersections.csv')
    target_data_ic50 = DTI(name='BindingDB_IC50').harmonize_affinities(mode='max_affinity')
    target_data_Kd = DTI(name='BindingDB_Kd').harmonize_affinities(mode='max_affinity')
    target_data_Ki = DTI(name='BindingDB_Ki').harmonize_affinities(mode='max_affinity')
    davis_data = DTI(name='DAVIS').get_data()
    kiba_data = DTI(name='KIBA').get_data()

    print(davis_data.head().to_string())
    print(kiba_data.head().to_string())

    drugbank_id = intersection['drugbank_id'].values.tolist()
    smiles = intersection['smile'].values.tolist()
    protein_seqs = []
    missing_counter = 0
    for smile in smiles:
        # candidates = target_data.filter(regex=re.compile(smile, re.IGNORECASE))
        candidates_ic50 = target_data_ic50[target_data_ic50['Drug'] == smile]
        candidates_Kd = target_data_Kd[target_data_Kd['Drug'] == smile]
        candidates_Ki = target_data_Ki[target_data_Ki['Drug'] == smile]
        candidates_davis = davis_data[davis_data['Drug'] == smile]
        candidates_kiba = kiba_data[kiba_data['Drug'] == smile]
        candidates = pd.concat([candidates_ic50, candidates_Kd, candidates_Ki, candidates_davis, candidates_kiba], axis=0)
        if candidates.empty:
            print('no drug find matching the smile')
            protein_seqs.append('')
            missing_counter += 1
        else:
            print(intersection[intersection['smile'] == smile].to_string())
            print(candidates.loc[[candidates['Y'].idxmax()]]['Drug_ID'])
            protein_seq = candidates.loc[[candidates['Y'].idxmax()]]['Target']
            protein_seqs.append(protein_seq)
    print('out of {} drugs, {} drugs has no target protein found'.format(len(drugbank_id), missing_counter))
    target_sequence_dict = {
        'drugbank_id': drugbank_id,
        'smile': smiles,
        'target': protein_seqs
    }

    with open('./useful_data/target_seq_dictionary.pickle', 'wb') as handle:
        pickle.dump(target_sequence_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # build_base_dataframe()
    # save_ddi_files_to_csv()
    # build_ddi_base_df()
    # build_dd_disease_base_df()
    # find_intersection()
    # find_intersection_ddi()
    # find_intersection_dd_disease()
    build_x_tensor()
    build_y_tensor()
    # generate_side_info_tanimoto()
    # generate_side_info_sider_distance_prep()
    # generate_side_info_side_effect()
    #generate_side_info_target_sequence_prep()


