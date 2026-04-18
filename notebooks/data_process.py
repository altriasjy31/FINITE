data_path = '/root/autodl-tmp/data-dgz/'
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import os
import obonet
import networkx

namespace = ['bp', 'cc', 'mf']



for name in namespace:
    with open(data_path+'swissprot_exp.pkl', 'rb') as file:
        swissprot_data = pickle.load(file)
    with open(data_path+ name +'/train_data.pkl', 'rb') as file:
        train_data = pickle.load(file)
    with open(data_path+ name +'/valid_data.pkl', 'rb') as file:
        valid_data = pickle.load(file)
    with open(data_path+ name +'/test_data.pkl', 'rb') as file:
        test_data = pickle.load(file)

    protein_namespace = train_data['proteins'].to_list() + valid_data['proteins'].to_list() + test_data['proteins'].to_list()
    esm_repr = np.load(data_path + 'esm-repr/esm_representations.npy')
    terms = pickle.load(open(data_path + name + '/terms.pkl', 'rb'))
    terms = terms['gos']
    protein_name_id_map = {}
    go_name_id_map = {}

    output_path = '/root/autodl-tmp/source/pprogo-flg/data-dgz/' + name
    node_path = output_path + '/node.dat'
    link_path = output_path + '/link.dat'
    link_test_path = output_path + '/link_test.dat'
    
    # create node.dat
    # protein nodes
    node_df = pd.DataFrame(columns=['id', 'name', 'type', 'feature'])
    for i in range(swissprot_data.shape[0]):
        line = swissprot_data.iloc[i,:]
        protein_name = line['proteins']
        feature = ','.join(map(str, esm_repr[i]))
        
        if(protein_name in protein_namespace):
            protein_name_id_map[protein_name] = i
            # node_df.append({'id':i, 'name':name, 'type':type, 'feature':feature})
            new_row = pd.DataFrame({'id':i, 'name':name, 'type':0, 'feature':feature}, index=[i])
            node_df = pd.concat([node_df, new_row])
    # go nodes
    for j, term in enumerate(terms):
        go_name_id_map[term] = i+j
        
        new_row = pd.DataFrame({'id':i+j, 'name':term, 'type':1, 'feature':None}, index=[i+j])
        node_df = pd.concat([node_df, new_row])
        # node_df.append({'id':i + j, 'name':term, 'type':1, 'feature':None})
        
    node_df.to_csv(node_path, sep='\t', index=False, header=False)
    
    # create link.dat
    link_df = pd.DataFrame(columns=['src_id', 'dst_id', 'type', 'score'])
    link_df_index = 0
    # train_data valid_data
    for i in range(train_data.shape[0]):
        gos = list(train_data['prop_annotations'])[i]
        for go_annotation in gos:
            src_id = protein_name_id_map[list(train_data['proteins'])[i]]
            dst_id = go_name_id_map[go_annotation]
            score = 1.0
            new_row = pd.DataFrame({'src_id':src_id, 'dst_id':dst_id, 'type':0, 'score':score}, index=link_df_index)
            link_df = pd.concat([link_df, new_row])
            link_df_index += 1
            # link_df.append({'src_id':src_id, 'dst_id':dst_id, 'type':0, 'score':score})
    for i in range(valid_data.shape[0]):
        for go_annotation in list(valid_data['prop_annotations'])[i]:
            src_id = protein_name_id_map[list(valid_data['proteins'])[i]]
            dst_id = go_name_id_map[go_annotation]
            score = 1.0
            # link_df.append({'src_id':src_id, 'dst_id':dst_id, 'type':0, 'score':score})
            new_row = pd.DataFrame({'src_id':src_id, 'dst_id':dst_id, 'type':0, 'score':score}, index=link_df_index)
            link_df = pd.concat([link_df, new_row])
            link_df_index += 1
            
    # ppi data
    esm_sim = ssp.load_npz(data_path + '/esm-repr/esm_sim_csr.npz')
    mmseqs_sim = ssp.load_npz(data_path + '/mmseqs/mmseqs_sim_csr.npz')
    def save_csr_top_100(csr, save_path):
        if os.path.exists(save_path):
            new_csr = ssp.load_npz(save_path)
            return new_csr
        def topk_row(row, k=100):
            non_zero_indices = row.nonzero()[1]
            non_zero_values = row.data
            topk_indices = np.argsort(-non_zero_values)[:k]
            topk_values = non_zero_values[topk_indices]
            topk_cols = non_zero_indices[topk_indices]
            return topk_values, topk_cols
        new_data = []
        new_indices = []
        new_indptr = [0]
        for i in range(csr.shape[0]):
            row = csr.getrow(i)
            topk_values, topk_cols = topk_row(row)
            new_data.extend(topk_values)
            new_indices.extend(topk_cols)
            new_indptr.append(len(new_data))

        new_csr = ssp.csr_matrix((new_data, new_indices, new_indptr), shape=csr.shape)
        np.savez(save_path, data=new_csr.data, indices=new_csr.indices, indptr=new_csr.indptr, shape=new_csr.shape)
        
        return new_csr
    esm_sim_csr_top_100 = save_csr_top_100(esm_sim, data_path+'/esm_sim_csr_top_100.npz')
    # mmseqs_sim_csr_top_100 = save_csr_top_100(mmseqs_sim, data_path+'/mmseqs_sim_csr_top_100.npz')
    # ppi = (esm_sim_csr_top_100 + mmseqs_sim_csr_top_100) / 2
    ppi = esm_sim_csr_top_100

    for row in range(ppi.shape[0]):
        col_indices, row_data = ppi[row].nonzero()[1], ppi[row].data
        for i in range(col_indices.shape[0]):
            col = col_indices[i]
            data = row_data[i]

            # link_df.append({'src_id':row, 'dst_id':col, 'type':1, 'score':data})
            new_row = pd.DataFrame({'src_id':row, 'dst_id':col, 'type':1, 'score':data}, index=link_df_index)
            link_df = pd.concat([link_df, new_row])
            link_df_index += 1

    # go.obo data
    go_path = data_path + '/go.obo'
    go_graph = obonet.read_obo(go_path)
    obo_name_to_id = {data.get('name'): id_ for id_, data in go_graph.nodes(data=True)}
    for term in terms:
        ancestors = networkx.descendants(go_graph, term)
        src_id = term
        dst_id = ancestors
        score = 1.0
        # link_df.append({'src_id':src_id, 'dst_id':dst_id, 'type':2, 'score':score})
        new_row = pd.DataFrame({'src_id':src_id, 'dst_id':dst_id, 'type':2, 'score':score}, index=link_df_index)
        link_df = pd.concat([link_df, new_row])
        link_df_index += 1
        
    link_df.to_csv(link_path, sep='\t', index=False, header=False)
    
        
    
    
    
    

    
    