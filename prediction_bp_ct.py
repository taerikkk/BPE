import sys
import pickle
import networkx as nx
import math
import numpy as np
import gzip
import random
from sklearn.model_selection import KFold
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# from sklearn.metrics.pairwise import rbf_kernel

def main():
    print("Start: " + str(datetime.now()))

    if len(sys.argv) == 9:
        DATA = sys.argv[1]
        GRAPH_TYPE = int(sys.argv[2])
        type_compat = sys.argv[3]
        type_emb = sys.argv[4]
        compat_threshold1 = float(sys.argv[5])
        compat_threshold2 = float(sys.argv[6])
        type_sim = sys.argv[7]
        ct_type = sys.argv[8]
    elif len(sys.argv) == 7:
        DATA = sys.argv[1]
        GRAPH_TYPE = int(sys.argv[2])
        type_compat = sys.argv[3]
        type_emb = sys.argv[4]
        compat_threshold1 = None
        compat_threshold2 = None
        type_sim = sys.argv[5]
        ct_type = sys.argv[6]
    elif len(sys.argv) == 5:
        DATA = sys.argv[1]
        GRAPH_TYPE = int(sys.argv[2])
        type_compat = sys.argv[3]
        ct_type = sys.argv[4]
        type_emb = None
        compat_threshold1 = None
        compat_threshold2 = None
        type_sim = None
    else:
        # type_compat = 'table1', 'table2', 'table3'
        # type_emb = None, 'deepwalk', 'node2vec', 'doc2vec', 'word2vec'
        # compat_threshold1, 2 = None, 0.3, 0.5, 0.7
        # type_sim = None, 'rbf', 'minmax', 'cos'
        # ct_type = 'ct1_2', 'ct1_3', 'ct1_4', 'ct2', 'ct3_2', 'ct3_3', 'ct3_4'
        DATA = 'data_all'
        GRAPH_TYPE = 5
        type_compat = 'table3'
        type_emb = 'deepwalk'
        compat_threshold1 = 0.7
        compat_threshold2 = 0.7
        type_sim = 'cos'
        ct_type = 'ct1_3'

    N_FOLDS = 5
    max_epoch = 5

    print("Reading data...")
    # read graph data
    if False:
        g = nx.Graph()  # just for intellisense
    g = pickle.load(gzip.open('data/{}.graph_type{}.gzpickle'.format(DATA, GRAPH_TYPE), 'rb'))

    # read ground truth data
    url_truth = pickle.load(gzip.open('data/{}.url_truth.gzpickle'.format(DATA), 'rb'))
    community_truth = pickle.load(gzip.open('data/community_truth.{}.gzpickle'.format(ct_type), 'rb'))
    data = list(url_truth.keys())
    has_ground_truth = set(url_truth.keys())

    print("Done.")

    # build folds
    kf = KFold(n_splits=N_FOLDS, shuffle=True)
    precision_sum = float(0)
    recall_sum = float(0)
    f1score_sum = float(0)
    accuracy_sum = float(0)

    print("Starting {}-fold cross-validation".format(N_FOLDS))
    for train, test in kf.split(data):
        # build training/tset set
        training_set = set(np.array(data)[train])
        test_set = set(np.array(data)[test])

        # split training&test set to each class(rel/irrel)
        relevant_training = set()
        irrelevant_training = set()
        relevant_test = set()
        irrelevant_test = set()
        for node in training_set:
            if url_truth[node] == 1:
                relevant_training.add(node)
            elif url_truth[node] == 0:
                irrelevant_training.add(node)
            else:
                print("error: ground truth error (" + str(ground_truth[node]) + ")")
        for node in test_set:
            if url_truth[node] == 1:
                relevant_test.add(node)
            elif url_truth[node] == 0:
                irrelevant_test.add(node)
            else:
                print("error: ground truth error (" + str(ground_truth[node]) + ")")

        # initilize node label
        for node in g.nodes():
            g.nodes[node]['label'] = None
            g.nodes[node]['best_label'] = -1
            g.nodes[node]['data_cost'] = [0.5, 0.5]
            # msg box is a dict
            g.nodes[node]['msgbox'] = {}
            g.nodes[node]['msg_comp'] = [0, 0]
            for nbr in list(g.neighbors(node)):
                g.nodes[node]['msgbox'][nbr] = [0, 0]

        # set training set node labels
        mal = 0
        bn = 0
        for node in has_ground_truth:
            if node in training_set:
                g.nodes[node]['label'] = url_truth[node]
                if url_truth[node] == 1:      # malicious
                    g.nodes[node]['data_cost'] = [0.99, 0.01]
                    mal+=1
                elif url_truth[node] == 0:    # benign
                    g.nodes[node]['data_cost'] = [0.01, 0.99]
                    bn+=1
        print(mal,bn)

        mal = 0
        bn = 0
        # set community truth
        has_community_truth = set(community_truth.keys())
        nodes = set(g.nodes())
        for node in nodes:
            if node in has_community_truth:
                g.nodes[node]['label'] = community_truth[node]
                if community_truth[node] == 1:      # malicious
                    g.nodes[node]['data_cost'] = [0.99, 0.01]
                    mal+=1
                elif community_truth[node] == 0:    # benign
                    g.nodes[node]['data_cost'] = [0.01, 0.99]
                    bn+=1
                else:
                    g.nodes[node]['data_cost'] = [community_truth[node], 1 - community_truth[node]]
        print(mal,bn)

        # set distances for all edges
        if type_emb == 'None':
            type_emb = None
        if type_emb != None:
            emb = pickle.load(gzip.open('data/{}.graph_type{}.{}_emb.gzpickle'.format(DATA, GRAPH_TYPE, type_emb), 'rb'))

            min_dist = float("inf")
            max_dist = -float("inf")

            for edge in g.edges():
                if type_sim == 'minmax':
                    # euclidean distance
                    g.edges[edge]['distance'] = np.linalg.norm(emb[edge[0]] - emb[edge[1]])
                    if g.edges[edge]['distance'] > max_dist:
                        max_dist = g.edges[edge]['distance']
                    if g.edges[edge]['distance'] < min_dist:
                        min_dist = g.edges[edge]['distance']
                elif type_sim == 'cos':
                    # cosine similarity
                    g.edges[edge]['sim'] = (np.dot(emb[edge[0]], emb[edge[1]]) / (np.linalg.norm(emb[edge[0]]) * np.linalg.norm(emb[edge[1]])))
                    g.edges[edge]['distance'] = 1 - g.edges[edge]['sim']
                elif type_sim == 'rbf':
                    # euclidean distance
                    g.edges[edge]['distance'] = np.linalg.norm(emb[edge[0]] - emb[edge[1]])
                    # rbf sim (see: https://en.wikipedia.org/wiki/Radial_basis_function_kernel )
                    g.edges[edge]['sim'] = np.exp((-1.0 / 2.0) * np.power(g.edges[edge]['distance'], 2.0))

            if type_sim == 'minmax':
                for edge in g.edges():
                    g.edges[edge]['sim'] = 1 - np.divide((g.edges[edge]['distance'] - min_dist), max_dist-min_dist)

        # if type_emb not provided...
        else:
            # set initial messages
            for edge in g.edges():
                #g.edges[edge]['msg'] = [0, 0]
                g.edges[edge]['distance'] = 1.0
                g.edges[edge]['sim'] = 0.5

        print("Done.")

        # print("Iteration: {} MAP: {}".format(0, MAP(g)))
        for epoch in range(max_epoch):
            precision = float(0)
            recall = float(0)
            f1score = float(0)
            accuracy = float(0)

            step(g, type_compat, compat_threshold1=compat_threshold1, compat_threshold2=compat_threshold2)
            print("Iteration: {} MAP: {}".format(epoch + 1, MAP(g)))

            relevant_correctness = 0
            relevant_incorrectness = 0
            for i in relevant_test:
                if g.nodes[i]['best_label'] == 1:
                    relevant_correctness += 1
                else:
                    relevant_incorrectness += 1

            irrelevant_correctness = 0
            irrelevant_incorrectness = 0
            for i in irrelevant_test:
                if g.nodes[i]['best_label'] == 0:
                    irrelevant_correctness += 1
                else:
                    irrelevant_incorrectness += 1

            print("rel_cor: " + str(relevant_correctness))
            print("rel_incor: " + str(relevant_incorrectness))
            print("irrel_cor: " + str(irrelevant_correctness))
            print("irrel_incor: " + str(irrelevant_incorrectness))

            print("Relevant Accuracy: {:.6}".format(relevant_correctness / (relevant_correctness + relevant_incorrectness)))
            print("Irrelevant Accuracy: {:.6}".format(irrelevant_correctness / (irrelevant_correctness + irrelevant_incorrectness)))

            if (relevant_correctness + irrelevant_incorrectness) == 0:
                precision = float(0)
            else:
                precision = relevant_correctness / (relevant_correctness + irrelevant_incorrectness)
            print("Precision: {:.6}".format(precision))

            if (relevant_correctness + relevant_incorrectness) == 0:
                recall = float(0)
            else:
                recall = relevant_correctness / (relevant_correctness + relevant_incorrectness)
            print("Recall: {:.6}".format(recall))

            if (precision + recall) == 0:
                f1score = float(0)
            else:
                f1score = 2 * precision * recall / (precision + recall)
            print("F1 score: {:.6}".format(f1score))

            accuracy = (relevant_correctness + irrelevant_correctness) / (relevant_correctness + relevant_incorrectness + irrelevant_correctness + irrelevant_incorrectness)
            print("Accuracy: {:.6}".format(accuracy))

        precision_sum += precision
        recall_sum += recall
        f1score_sum += f1score
        accuracy_sum += accuracy
        print()
    
    print("Done.")
    print()

    print("Averaged precision: {:.6}".format(precision_sum / N_FOLDS))
    print("Averaged recall: {:.6}".format(recall_sum / N_FOLDS))
    print("Averaged F1 score: {:.6}".format(f1score_sum / N_FOLDS))
    print("Averaged accuracy: {:.6}".format(accuracy_sum / N_FOLDS))

    print("End: " + str(datetime.now()))


def step(G, type_compat, compat_threshold1=None, compat_threshold2=None):
    for n in tqdm(G.nodes(), desc="Propagate from vertices with label", mininterval=0.5):
        if G.nodes[n]['label'] != None:
            for nbr in G.neighbors(n):
                # do not propagate to nodes with label
                if G.nodes[nbr]['label'] == None:
                    _send_msg_label(G, n, nbr)
    #for n in tqdm(G.nodes(), desc="Compiling message boxes 1", mininterval=0.5):
    #    G.nodes[n]['msg_comp'] = [0, 0]
    #    for nbr in G.neighbors(n):
    #        G.nodes[n]['msg_comp'][0] += G.nodes[n]['msgbox'][nbr][0]
    #        G.nodes[n]['msg_comp'][1] += G.nodes[n]['msgbox'][nbr][1]
    for n in tqdm(G.nodes(), desc="Propagate from vertices without label", mininterval=0.5):
        if G.nodes[n]['label'] == None:
            for nbr in G.neighbors(n):
                # do not propagate to nodes with label
                if G.nodes[nbr]['label'] == None:
                    _send_msg(G, type_compat, n, nbr, compat_threshold1=compat_threshold1, compat_threshold2=compat_threshold2)
    #for n in tqdm(G.nodes(), desc="Compiling message boxes 2", mininterval=0.5):
    #    G.nodes[n]['msg_comp'] = [0, 0]
    #    for nbr in G.neighbors(n):
    #        G.nodes[n]['msg_comp'][0] += G.nodes[n]['msgbox'][nbr][0]
    #        G.nodes[n]['msg_comp'][1] += G.nodes[n]['msgbox'][nbr][1]


def _min_sum(G, _from, _to, type_compat, compat_threshold1, compat_threshold2):
    eps = 0.001

    new_msg = [0] * 2
    for i in range(2):  # we only have 2 labels so far
        fromnode = G.nodes[_from]

        # initialize
        # related => label 1
        # not related => label 0
        p_not_related = 0
        p_related = 0

        # data cost
        #p_not_related += math.log(1 - fromnode['data_cost'][0])
        #p_related += math.log(1 - fromnode['data_cost'][1])
        p_not_related += fromnode['data_cost'][0]
        p_related += fromnode['data_cost'][1]

        #for nbr in G.neighbors(_from):
        #    if nbr == _to:
        #        continue
        #    p_not_related += fromnode['msgbox'][nbr][0]
        #    p_related += fromnode['msgbox'][nbr][1]
        p_not_related += fromnode['msg_comp'][0] - fromnode['msgbox'][_to][0]
        p_related += fromnode['msg_comp'][1] - fromnode['msgbox'][_to][1]

        # smoothness cost
        if type_compat == 'table1':
            # original (we think this version is for sum-product...)
            #p_not_related += 0.5 + eps if i == 0 else 0.5 - eps
            #p_related += 0.5 - eps if i == 0 else 0.5 + eps
            p_not_related += 0.5 - eps if i == 0 else 0.5 + eps
            p_related += 0.5 + eps if i == 0 else 0.5 - eps
        elif type_compat == 'table2':
            # original (this version works only when table2 && cos)
            #p_not_related += 0 if i == 0 else 1 - G[_from][_to]['distance']
            #p_related += 1 - G[_from][_to]['distance'] if i == 0 else 0
            #p_not_related += 0 if i == 0 else G[_from][_to]['sim']
            #p_related += G[_from][_to]['sim'] if i == 0 else 0
            p_not_related += 0 if i == 0 else G[_from][_to]['distance']
            p_related += G[_from][_to]['distance'] if i == 0 else 0
        elif type_compat == 'table3':
            # original (our sim are similarities -> same = 1 / completely different = 0)
            p_not_related += np.min([compat_threshold1, 1 - G[_to][_from]['sim']]) if i == 0 else np.max([compat_threshold2, G[_to][_from]['sim']])
            p_related += np.max([compat_threshold2, G[_to][_from]['sim']]) if i == 0 else np.min([compat_threshold1, 1 - G[_to][_from]['sim']])
            
        new_msg[i] = min(p_not_related, p_related)

    # Normalization
    # new_msg = np.exp(new_msg) / np.sum(np.exp(new_msg))

    return new_msg


def _send_msg_label(G, _from, _to):
    # if lable is given
    if G.nodes[_from]['label'] == 1:
        msg = [1, 0]
    elif G.nodes[_from]['label'] == 0:
        msg = [0, 1]
    else:
        # ct2 case
        msg = G.nodes[_from]['data_cost']

    to_node = G.nodes[_to]
    # subtract original msg
    to_node['msg_comp'][0] -= to_node['msgbox'][_from][0]
    to_node['msg_comp'][1] -= to_node['msgbox'][_from][1]
    # add new msg
    to_node['msg_comp'][0] += msg[0]
    to_node['msg_comp'][1] += msg[1]
    # orignal msg := new msg
    to_node['msgbox'][_from] = msg


def _send_msg(G, type_compat, _from, _to, compat_threshold1 = None, compat_threshold2 = None):
    # label not given
    msg = _min_sum(G, _from, _to, type_compat, compat_threshold1, compat_threshold1)

    to_node = G.nodes[_to]
    # subtract original msg
    to_node['msg_comp'][0] -= to_node['msgbox'][_from][0]
    to_node['msg_comp'][1] -= to_node['msgbox'][_from][1]
    # add new msg
    to_node['msg_comp'][0] += msg[0]
    to_node['msg_comp'][1] += msg[1]
    # orignal msg := new msg
    to_node['msgbox'][_from] = msg


def MAP(G):
    n_wrong_label = 0
    n_correct_label = 0

    for n in G.nodes():
        nodedata = G.nodes[n]

        cost_not_related = 0
        cost_related = 0

        #cost_not_related += math.log(1 - G.node[n]['data_cost'][0])
        #cost_related += math.log(1 - G.node[n]['data_cost'][1])
        cost_not_related += nodedata['data_cost'][0]
        cost_related += nodedata['data_cost'][1]

        #for nbr, msg in nodedata['msgbox'].items():
        #    cost_not_related += msg[0]
        #    cost_related += msg[1]
        #for edge, eattr in G[n].items():
        #    cost_not_related += eattr['msg'][0]
        #    cost_related += eattr['msg'][1]
        cost_not_related += nodedata['msg_comp'][0]
        cost_related += nodedata['msg_comp'][1]

        if cost_related < cost_not_related:
            nodedata['best_label'] = 1
        else:
            nodedata['best_label'] = 0

        if nodedata['label'] == 1 and nodedata['best_label'] == 0:
            #print("error2: wrong label!")
            n_wrong_label += 1
        elif nodedata['label'] == 0 and nodedata['best_label'] == 1:
            #print("error2: wrong label!")
            n_wrong_label += 1
        elif nodedata['label'] == 1 and nodedata['best_label'] == 1:
            n_correct_label += 1
        elif nodedata['label'] == 0 and nodedata['best_label'] == 0:
            n_correct_label += 1

    print("# wrong label: " + str(n_wrong_label))
    print("# correct label: " + str(n_correct_label))

    energy = 0
    for n in G.nodes():
        cur_label = G.nodes[n]['best_label']

        #energy += math.log(1 - G.node[n]['data_cost'][cur_label])
        energy += G.nodes[n]['data_cost'][cur_label]
        for nbr, eattr in G[n].items():
            energy += 0 if G.nodes[nbr]['best_label'] == cur_label else eattr['distance']

    return energy

   
if __name__ == '__main__':
	main()
