import numpy as np
import pandas as pd
from collections import defaultdict
import geopandas as gpd
from  paper_data.mapping import linear_baseline
import argparse
import sys
import os


TOTAL = 0
TOTAL_CORRECT = 0
RANDOM_FLAG = False
node_array = []

def load_data(data_dir="paper_data"):
    try:
        counties_gpd = gpd.read_file(os.path.join(data_dir, "map/map/county/la_county.shp"))
        df_probs = pd.read_csv(os.path.join(data_dir, "la_county.csv"))
        df_stats = pd.read_csv("us_county_reporting_probs.csv")
        df_la_true = pd.read_csv(os.path.join(data_dir, "LA_real_truth.csv"))
        df_la = pd.read_csv(os.path.join(data_dir, "LA_real_test.csv"))
        
        return counties_gpd, df_probs, df_stats, df_la_true, df_la
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please ensure all required data files are in the correct directories.")
        sys.exit(1)
        
def build_infection_tree(df_la, df_la_true):

    real_tree = defaultdict(list)
    if 'baseline_guess' not in df_la.columns:
        df_la['True_source'] = df_la_true["from_geoid"]
        df_la['True_target'] = df_la_true["to_geoid"]
        df_la['baseline_guess_s'] = None
        df_la['baseline_guess_t'] = None
        df_la['Source_raw'] = None  
        df_la['Target_raw'] = None
        df_la['constant'] = None
    
    for i, row in df_la.iterrows():
        parent_node, source_geoid = row['Source'], row['from_geoid']
        parent_id = getints(parent_node)
        infector_key = f"{source_geoid}-{parent_id}"
        parents_nodes = df_la.loc[df_la['Source'] == parent_node]
        
        if infector_key in real_tree:
            continue
            
        if i == 0:   
            print(f"The initial case is: {parent_id}")
            for _, row in parents_nodes.iterrows():    
                child_node, source_geoid, target_geoid, date_gap = row['Target'], row['from_geoid'], row['to_geoid'], row['date_gap']
                infector_key = f"{source_geoid}-{parent_id}"
                child_id = getints(child_node)
                infected_key = f"{target_geoid}-{child_id}"
                print(f"The infected key is: {infected_key}")
                
                obs = target_geoid != '???'
                real_tree["initial"].append((infector_key, 0, True)) 
                real_tree[infector_key].append((infected_key, date_gap, obs))
        else:
            for _, row in parents_nodes.iterrows():   
                child_node, source_id, target_geoid, date_gap = row['Target'], row['from_geoid'], row['to_geoid'], row['date_gap']
                obs = target_geoid != '???'
                infector_key = f"{source_geoid}-{parent_id}"
                child_id = getints(child_node)
                infected_key = f"{target_geoid}-{child_id}"
                real_tree[infector_key].append((infected_key, date_gap, obs))
    
    return real_tree, df_la

def setup_matrices(df_probs, df_stats):
    counties = [str(county) for county in df_probs['from'].unique()]
    n_counties = len(counties)
    county_index = {county: i for i, county in enumerate(counties)}
    transition_matrix = np.zeros((n_counties, n_counties))
    df_probs['from'] = df_probs['from'].astype(str).str[0:5]
    df_probs['to'] = df_probs['to'].astype(str).str[0:5]
    
    for _, row in df_probs.iterrows():
        from_county, to_county = row['from'], row['to']
        if from_county in county_index and to_county in county_index:
            i, j = county_index[from_county], county_index[to_county]
            transition_matrix[i, j] = row['prob']
            
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.where(row_sums != 0, transition_matrix / row_sums, 0)
    transition_matrix[np.isnan(transition_matrix)] = 0
    transition_matrix = np.where(transition_matrix.sum(axis=1, keepdims=True) == 0,
                                 1.0 / n_counties, transition_matrix)

    emission_probs = np.full(n_counties, 0.5)  
    for i, prob in enumerate(emission_probs):
        if prob == 0.5:
            current_county = counties[i]
            current_county_pop_matches = df_stats[df_stats['GeoId'] == current_county]['Population']      
            if current_county_pop_matches.empty:
                df_stats['GeoId'] = df_stats['GeoId'].astype(str)
                df_stats['population_diff'] = abs(df_stats['Population'] - int(current_county))
                closest_county = df_stats.loc[df_stats['population_diff'].idxmin()]
                emission_probs[i] = closest_county['pre_cal_prob'] / 5
            else:
                emission_probs[i] = emission_probs[i-1] / 5

    return counties, county_index, transition_matrix, emission_probs

     
def getints(input_string):
    digits = filter(str.isdigit, input_string)
    return int("".join(digits)) if any(digits) else 0
  

class TreeNode:
    def __init__(self, id, county, person_id, day=None, parent=None, observed=False):
        self.id = id  # 20200-5
        self.county = county
        self.person_id = person_id
        self.day = day
        self.parent = parent
        self.children = []
        self.observed = observed  
        self.guess = []
        self.baseline = []
        self.constant = []
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    
    def __str__(self):
        return f"{self.id} (Day {self.day})"
    def __repr__(self):
        return self.__str__()
    
class InfectionTree:
    def __init__(self):
        self.nodes = {}  
        self.root_nodes = []  # allows for multiple initial infections
        self.leaf_nodes = []  
    
    def add_node(self, id, county, person_id, day=None, parent_id=None, observed=False):
        if id not in self.nodes:
            node = TreeNode(id, county, person_id, day, observed=observed)
            self.nodes[id] = node
        else:
            node = self.nodes[id]
            if day is not None:
                node.day = day
            node.observed = observed 
        if parent_id and parent_id in self.nodes:
            parent = self.nodes[parent_id]
            parent.add_child(node)
        elif parent_id == "initial":  
            self.root_nodes.append(node)
        
        self.update_leaf_nodes()
        return node
               
    def update_all(self, curr_node):
        if self.nodes[curr_node].children is not None:
            children = self.root_nodes[curr_node].children
            for child in children:
                self.nodes[child].day += self.nodes[curr_node].day
                self.update_all(child)
            
    def update_leaf_nodes(self):
        self.leaf_nodes = [node for node in self.nodes.values() if node.is_leaf()]
        
    def update_constant(self):
         for root_node in self.root_nodes:
             root_node.constant = None
             self.update_constant_recursive(root_node)
     
    def update_constant_recursive(self, node):
         for child in node.children:
             child.constant = node.county
             self.update_constant_recursive(child)
    
    def update_parents(self):
        for node in self.nodes:
            if self.nodes[node].parent is None:
                self.root_nodes.append(self.nodes[node])
                #print(f"{node} is a root")
            else:
                self.nodes[node].constant = self.nodes[node].parent.county
     
     
    def update_days(self):
        for node in self.root_nodes:
            self.update_all(self.root_nodes[node])
            
    def create(self, tree_dict):
        infection_tree = InfectionTree()
        for item in tree_dict.get("initial", []):
            if len(item) == 2:  
                person, day = item
                observed = False  # Default to False for backward compatibility
            else:  # New format (person, day, observed)
                person, day, observed = item
                
            county, person_id = person.split('-')
            infection_tree.add_node(person, county, person_id, day, "initial", observed)
   
        for infector, infected_list in tree_dict.items():
            if infector != "initial":
                if infector not in infection_tree.nodes:
                    infector_county, infector_id = infector.split('-')
                    infection_tree.add_node(infector, infector_county, infector_id)
                
                for item in infected_list:
                    if len(item) == 2:  # Handle old format
                        infected, day = item
                        observed = False  # Default
                    else:  # New format
                        infected, day, observed = item
                        
                    infected_county, infected_id = infected.split('-')
                    infection_tree.add_node(infected, infected_county, infected_id, day, infector, observed)
        
        return infection_tree

def create_infection_tree(tree_dict):
    tree = InfectionTree()
    return tree.create(tree_dict)


def viterbi(vpath, counties, county_index, transition_matrix, emission_probs, counties_gpd, debug=False, p_results=False):
    global node_array

    n_counties = len(counties)
    first_ob = vpath[0]
    last_ob = vpath[-1]
    unobserved_nodes = []
    unobserved_positions = []
    for i in range(1, len(vpath)-1):
        if not vpath[i].observed:
            unobserved_nodes.append(vpath[i])
            unobserved_positions.append(i)
    
    n_unobserved = len(unobserved_nodes)
    num_unobserved_nodes=+n_unobserved
    if n_unobserved == 0:
        return
    #neccesary for some reason
    try:
        c_first_i = county_index[first_ob.county]
        c_last_i = county_index[last_ob.county]
    except KeyError as e:
        print(f"Warning: County not found in county_index: {e}. Skipping this path.")
        return
    
    #MAPPING LINEAR BASELINE
    unobserved_linear_interpolation = linear_baseline(first_ob.county, last_ob.county, counties_gpd, n_unobserved)
    #print(unobserved_linear_interpolation)
    #MAPPING LINEAR BASELINE
    
    log_unobserved_emission = np.log(1 - emission_probs + 1e-10)  # Add small epsilon to avoid log(0)
    log_transition_matrix = np.log(transition_matrix + 1e-10)
   
    log_probs = np.full((n_unobserved, n_counties), -np.inf)
    path_tracking = np.zeros((n_unobserved, n_counties), dtype=int)
    
    
    log_probs[0] = log_transition_matrix[c_first_i] + log_unobserved_emission
    
    for pos in range(1, n_unobserved):
        for curr_state in range(n_counties):
            trans_scores = log_probs[pos-1] + log_transition_matrix[:, curr_state]
            best_prev = np.argmax(trans_scores)
            
            log_probs[pos, curr_state] = trans_scores[best_prev] + log_unobserved_emission[curr_state]
            path_tracking[pos, curr_state] = best_prev
    
    final_scores = log_probs[n_unobserved-1] + log_transition_matrix[:, c_last_i]
    
    best_path = []
    current_state = np.argmax(final_scores)
    best_path.append(current_state)
    for pos in range(n_unobserved-1, 0, -1):
        current_state = path_tracking[pos, current_state]
        best_path.append(current_state)
    
    best_path.reverse()
    #log to prob
    regular_probs = np.exp(log_probs)
    final_probs = np.exp(final_scores)
    #normalize
    for pos in range(n_unobserved):
        if np.sum(regular_probs[pos]) > 0:
            regular_probs[pos] /= np.sum(regular_probs[pos])
    
    if np.sum(final_probs) > 0:
        final_probs /= np.sum(final_probs)
 
    #print(f"\nViterbi analysis for path: {vpath}")
    
    for pos in range(n_unobserved):
        actual_node = unobserved_nodes[pos]
     
        if pos == n_unobserved - 1:
            node_probs = final_probs
        else:
            node_probs = regular_probs[pos]
      
        top_indices = np.argsort(node_probs)[::-1][:5]
        top_prediction = counties[top_indices[0]]
        
        # note: This differs from the Viterbi path, top_prediction is the marginal best at this position
        # best_path[pos] is the best as part of the globally optimal sequence
        viterbi_best = counties[best_path[pos]]
        
        actual_node.guess = top_prediction 
        actual_node.baseline = unobserved_linear_interpolation[pos]
        if p_results:
            print(f"\nUnobserved Node {pos+1}, Actual node: {actual_node}")
            print(f"  BEST GUESS COUNTY (marginal): {actual_node.guess}")
            print(f"  VITERBI PATH COUNTY: {viterbi_best}")
            
        if actual_node.person_id not in node_array:
            print(f"Imputing {actual_node.person_id}")
            impute(actual_node, top_prediction)
        if p_results: 
            print("  Top 5 probabilities:")
            for idx in top_indices:
                county = counties[idx]
                prob = node_probs[idx]
                print(f"    {county}: {prob:.4f}")
                
'''
 These following three functions are what create an infection tree, for both real and simulated data
 It does find paths that are in the format: 
    (OB) -> ((UN)*+ -> (OB)*+)*+
    where *+ is 0 or more
'''
  
def startsearch(infection_tree, *args):
    roots = infection_tree.root_nodes #array
    for i in range(len(roots)):
        path = []
        path.append(roots[i])
        searchforpaths(path, infection_tree, *args)
    
def searchforpaths(path, infection_tree,*args):
    #print(f"Analysis of {path}")
    node = path[-1]
    if node.is_leaf():
        #print("\nleaf node")
        return path
    else: 
        for child in node.children:
            if child.observed: 
                path.append(child)
                searchforpaths(path, infection_tree, *args)
                path.pop()  
            else: 
                unobserved_path = [child]
                seek(path, unobserved_path, infection_tree, *args)
                

def seek(path, h_nodes, infection_tree, *args):
    current = h_nodes[-1]
    
    if not current.children:  # leaf 
        return
    
    for child in current.children:
        if child.observed:
            full_path = path.copy() + h_nodes + [child]
            if RANDOM_FLAG: 
                random_path(full_path, *args[:3])
            else: 
                viterbi(full_path, *args)

            prefix = path.copy() + h_nodes + [child]
            searchforpaths(prefix, infection_tree, *args)
        else:
            ext  = h_nodes.copy() + [child]
            seek(path, ext, infection_tree, *args)
            
def random_path(vpath, counties, county_index, transition_matrix, p_results=False):
    
    print(f"\Random analysis for path: {vpath}")
    unobserved_nodes = []
    unobserved_positions = []
    for i in range(1, len(vpath)-1):
        if not vpath[i].observed:
            unobserved_nodes.append(vpath[i])
            unobserved_positions.append(i)
    
    n_unobserved = len(unobserved_nodes)
    
    states = np.arange(len(transition_matrix))
    
    for pos in range(n_unobserved):
        
        curr_node = unobserved_nodes[pos]
        last_node = vpath[pos]
        parent_county = last_node.county
       
        try:
           node_index = county_index[parent_county]
        except KeyError:
           print(f"Warning: Parent county {parent_county} not found in county_index. Skipping this node.")
           continue
      
        probs = transition_matrix[node_index, :]
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones(len(states)) / len(states)
       
        random_pick = np.random.choice(states, p=probs)
        county_pick = counties[random_pick]
        
        print(f"random pick: {county_pick}")
        
        curr_node.county = county_pick
        if curr_node.person_id not in node_array:
            print(f"Imputing {curr_node.person_id}")
            impute(curr_node, county_pick, True)
            
            
'''
impute does a few things, it takes the node it passes and the top prediction (from viterbi) 
and inserts it into the relevent row/column in df_la. It also measures accuracy
'''
def impute(node, top_prediction, random_flag=False):
    global TOTAL, TOTAL_CORRECT
    
    if node.person_id in node_array:
        return
    node_array.append(node.person_id)
   
    for i, row in df_la.iterrows():
        parent_node, target_node = row['Source'], row['Target']
        parent_id = getints(parent_node)
        target_id = getints(target_node)
        df_la.loc[i, 'Source_raw'] = parent_id
        df_la.loc[i, 'Target_raw'] = target_id
    
    if node.county == '???' or random_flag == True:
        accuracy_counted = False
        rows_idx = df_la[df_la['Source_raw'] == int(node.person_id)].index
        if not rows_idx.empty:
            df_la.loc[rows_idx, 'from_geoid'] = top_prediction
            if node.baseline: df_la.loc[rows_idx, 'baseline_guess_s'] = node.baseline
            if node.constant: df_la.loc[rows_idx, 'constant'] = node.constant
            if not accuracy_counted:
                true_source = df_la.loc[rows_idx, 'True_source'].iloc[0]
                TOTAL += 1
                if int(top_prediction) == int(true_source):
                    TOTAL_CORRECT += 1
                accuracy_counted = True
     
        t_rows_idx = df_la[df_la['Target_raw'] == int(node.person_id)].index
        if not t_rows_idx.empty:
            df_la.loc[t_rows_idx, 'to_geoid'] = top_prediction
            if node.baseline: df_la.loc[t_rows_idx, 'baseline_guess_t'] = node.baseline
            if node.constant: df_la.loc[t_rows_idx, 'constant'] = node.constant
            if not accuracy_counted:
                true_target = df_la.loc[t_rows_idx, 'True_target'].iloc[0]
                TOTAL += 1
                if int(top_prediction) == int(true_target):
                    TOTAL_CORRECT += 1
                accuracy_counted = True
            
            if accuracy_counted:
                print(f"total_correct: {TOTAL_CORRECT}, TOTAL: {TOTAL}, accuracy: {TOTAL_CORRECT/TOTAL:.3f}")
    

def main():
    global RANDOM_FLAG, df_la
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='paper_data')
    parser.add_argument('--output', default='paper_data/imputed_data_paper_v3.csv')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    RANDOM_FLAG = args.random
    
    print("Loading data")
    counties_gpd, df_probs, df_stats, df_la_true, df_la = load_data(args.data_dir)
    
    print("Building infection tree")
    real_tree, df_la = build_infection_tree(df_la, df_la_true)
    counties, county_index, transition_matrix, emission_probs = setup_matrices(df_probs, df_stats)
    
    print("Creating infection tree structure...")
    real_infection_tree = create_infection_tree(real_tree)
    real_infection_tree.update_parents()
    real_infection_tree.update_constant()
    print(f"Starting {'Random' if RANDOM_FLAG else 'Viterbi'} analysis...")
    
    if RANDOM_FLAG:
        startsearch(real_infection_tree, counties, county_index, transition_matrix)
    else:
        startsearch(real_infection_tree, counties, county_index, transition_matrix, emission_probs, counties_gpd, args.debug, args.print)

    print(f"\nAnalysis complete")
    print(f"Final accuracy: {TOTAL_CORRECT/TOTAL:.3f} ({TOTAL_CORRECT}/{TOTAL})" if TOTAL > 0 else "No accuracy data")
    
    print(f"Saving results to {args.output}...")
    df_la.to_csv(args.output, index=False)
    print("Saved")

if __name__ == "__main__":
    main()

