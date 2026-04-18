import numpy as np
from tqdm import tqdm
import scipy.sparse as ssp
import pandas as pd
from numba import jit

@jit(nopython=True)
def extract_top_10_percent_numpy(data, row_indices, col_indices, num_rows, num_cols):
    result = []
    for row in range(num_rows):
        # 获取当前行的所有非零元素及其索引
        row_mask = row_indices == row
        cols = col_indices[row_mask]
        scores = data[row_mask]
        
        # 如果当前行没有非零元素，跳过
        if len(scores) == 0:
            continue
        
        # 获取前10%的元素
        num_top_elements = max(1, len(scores) // 10)
        top_indices = np.argsort(scores)[-num_top_elements:][::-1]  # 取前10%
        
        # 添加结果
        for i in tqdm(top_indices):
            result.append((row, cols[i], scores[i]))
    
    return result

def extract_links(matrix):
    # 获取CSR矩阵的非零元素和索引
    row_indices, col_indices = matrix.nonzero()
    data = matrix.data
    
    # 获取矩阵的形状
    num_rows, num_cols = matrix.shape
    
    # 调用Numba函数
    return extract_top_10_percent_numpy(data, row_indices, col_indices, num_rows, num_cols)
            
# ppi_links = extract_link(esm1_sim_csr)
esm1_prot_go_csr = ssp.load_npz('/root/autodl-tmp/source/pprogo-flg/data/esm1_mmseqs_prot_go_sim_csr.npz')

pg_links = extract_links(esm1_prot_go_csr)
pg_links = pd.DataFrame(pg_links, columns=['src', 'dst', 'score'])
pg_links.insert(loc=2, column='type', value=0)
print(pg_links)

pg_links.to_csv('/root/autodl-tmp/source/pprogo-flg/data/esm1_mmseqs_pg_10p.dat', sep='\t', index=False, header=True)