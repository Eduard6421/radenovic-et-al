
import os
import numpy as np
import copy 
import pandas as pd

from cirtorch.utils.general import get_results_root

def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[], qimages=None, images=None, qvecs=None, vecs=None, scores = None):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0
    
    q_df = pd.DataFrame(columns = ['query_path','results_path','query_emb','result_emb'])

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            print(f"{qimages[i]},0")
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        # sorted array of indexes which are in the positive
        
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.array([])
        
        current_idx = 0
        indices = []
        while(len(indices) < 100):
            if(current_idx not in junk):
                indices.append(current_idx)
            current_idx += 1
            
        
        #print(ranks[:100,i])
        im_names  = (np.array(images))[ranks[indices,i]].tolist()
        q_im_name = qimages[i]
        q_im_emb  = qvecs[i]
        im_emb    = np.swapaxes(vecs[:,ranks[indices,i]],0,1).tolist()
        
        res = {'query_path': q_im_name,'results_path': im_names,'query_emb': q_im_emb,'result_emb': im_emb, 'scores': scores[indices,i]} 
        
        q_df = pd.concat([q_df, pd.DataFrame.from_records([res])])

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq

        # Will print Average precision
        print("AP: {},{}".format(qimages[i],ap))
        # Will print Precision @ 100
        #print("P@100: {},{}".format(qimages[i],prs[i,0]))    
        
        pr = pr + prs[i, :]
        
    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    
    top_results_path = os.path.join(get_results_root(),'{}-top-100-results-and-scores.csv', index=False)
    q_df.to_csv(top_results_path, index=False)

    return map, aps, pr, prs


def compute_map_and_print(dataset, ranks, gnd, kappas=[100], qimages=None, images=None, qvecs = None, vecs = None, scores = None):
    
    
    # old evaluation protocol
    if dataset.startswith('oxford5k') or dataset.startswith('paris6k'):
        map, aps, _, _ = compute_map(ranks, gnd)
        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map*100, decimals=2)))

    # new evaluation protocol
    elif dataset.startswith('roxford5k') or dataset.startswith('rparis6k') or  dataset.startswith('pascalvoc') or dataset.startswith('caltech'):
        
        print('==================== Easy and Hard are considered Positives. Junk are considered  negatives =================')
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.array([])
            gnd_t.append(g)
        compute_map(ranks, gnd_t, kappas,qimages=copy.deepcopy(qimages), images=copy.deepcopy(images), qvecs = copy.deepcopy(qvecs), vecs = copy.deepcopy(vecs), scores = copy.deepcopy(scores))