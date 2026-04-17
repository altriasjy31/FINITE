from collections import deque, Counter
import warnings
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
import math

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

# CAFA4 Targets
CAFA_TARGETS = set([
    '287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090',
    '10116', '44689', '83333', '99287', '226900', '243273', '284812', '559292'])

def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES


def get_goplus_defs(filename='data/definitions.txt'):
    plus_defs = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            go_id, definition = line.split(': ')
            go_id = go_id.replace('_', ':')
            definition = definition.replace('_', ':')
            plus_defs[go_id] = set(definition.split(' and '))
    return plus_defs


class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.ic_norm = 0.0

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)
            self.ic_norm = max(self.ic_norm, self.ic[go_id])
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_norm_ic(self, go_id):
        return self.get_ic(go_id) / self.ic_norm

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
     
        return ont

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_prop_terms(self, terms):
        prop_terms = set()

        for term_id in terms:
            prop_terms |= self.get_anchestors(term_id)
        return prop_terms


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

def read_fasta(filename):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    seq = ''
                inf = line[1:].split()[0]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    return info, seqs


class DataGenerator(object):

    def __init__(self, batch_size, is_sparse=False):
        self.batch_size = batch_size
        self.is_sparse = is_sparse

    def fit(self, inputs, targets=None):
        self.start = 0
        self.inputs = inputs
        self.targets = targets
        if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
            self.size = self.inputs[0].shape[0]
        else:
            self.size = self.inputs.shape[0]
        self.has_targets = targets is not None

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
                res_inputs = []
                for inp in self.inputs:
                    if self.is_sparse:
                        res_inputs.append(
                            inp[batch_index, :].toarray())
                    else:
                        res_inputs.append(inp[batch_index, :])
            else:
                if self.is_sparse:
                    res_inputs = self.inputs[batch_index, :].toarray()
                else:
                    res_inputs = self.inputs[batch_index, :]
            self.start += self.batch_size
            if self.has_targets:
                if self.is_sparse:
                    labels = self.targets[batch_index, :].toarray()
                else:
                    labels = self.targets[batch_index, :]
                return (res_inputs, labels)
            return res_inputs
        else:
            self.reset()
            return self.next()
        

from collections import deque
from typing import Dict, Set, Optional


class WangGOSim:
    """
    Efficient Wang semantic similarity using:
      - precomputed ancestors per GO term (go2ancestors)
      - a GODag for parent relationships (goatools.godag.GODag)
    """

    dflt_rel2scf = {
        "is_a": 0.8,
        "part_of": 0.6,
        "regulates": 0.6,
        "negatively_regulates": 0.6,
        "positively_regulates": 0.6,
    }

    def __init__(
        self,
        godag,
        go2ancestors: Dict[str, Set[str]],
        rel2scf: Optional[Dict[str, float]] = None,
        use_relationships: bool = True,
    ):
        """
        Parameters
        ----------
        godag : goatools.godag.GODag
            GO DAG object from goatools.
        go2ancestors : dict
            Mapping GO -> set(ancestors), typically from get_go2ancestors.
            Usually DOES NOT contain the term itself; we will add it internally.
        rel2scf : dict, optional
            Edge-type weight map for Wang method. Missing keys fall back to defaults.
        use_relationships : bool
            If True and relationships are present in godag term objects,
            we use them with edge-type-specific weights; otherwise only 'is_a' (parents).
        """
        self.godag = godag
        self.go2anc = go2ancestors
        self.use_relationships = use_relationships

        # Merge default and user-provided edge weights
        self.rel2scf = dict(self.dflt_rel2scf)
        if rel2scf is not None:
            self.rel2scf.update(rel2scf)

        # Caches: per-root S-values and their total sums
        self._svals: Dict[str, Dict[str, float]] = {}
        self._sv_tot: Dict[str, float] = {}

    # ---------- Public API ----------

    def get_sim(self, go_a: str, go_b: str) -> Optional[float]:
        """Compute Wang semantic similarity between two GO terms."""
        if go_a not in self.godag or go_b not in self.godag:
            return None
        if go_a not in self.go2anc or go_b not in self.go2anc:
            return None

        s_a = self._get_svals(go_a)
        s_b = self._get_svals(go_b)
        if s_a is None or s_b is None:
            return None

        # Intersection of all nodes in both induced DAGs
        common = s_a.keys() & s_b.keys()
        if not common:
            return 0.0

        numer = sum(s_a[t] + s_b[t] for t in common)
        denom = self._sv_tot[go_a] + self._sv_tot[go_b]
        if denom == 0.0:
            return None
        return numer / denom

    # ---------- Internal helpers ----------

    def _get_svals(self, root: str) -> Optional[Dict[str, float]]:
        """Return S-values for a root term (computing and caching if needed)."""
        if root in self._svals:
            return self._svals[root]

        if root not in self.go2anc:
            return None

        # Build induced node set: ancestors + root
        nodes = set(self.go2anc[root])
        nodes.add(root)

        # S(root) = 1.0; others will be filled
        svals = {root: 1.0}

        # BFS from root upwards along parent edges.
        # For Wang: S(parent) = max_{child on path to root} (w_e * S(child))
        visited = set([root])
        queue = deque([root])

        while queue:
            child = queue.popleft()
            child_s = svals[child]
            term_child = self.godag[child]

            # 1) GOATOOLS "parents" attribute (is_a edges)
            for parent_term in term_child.parents:
                parent_id = parent_term.id
                if parent_id not in nodes:
                    continue
                w = self.rel2scf.get("is_a", 0.8)
                cand = w * child_s
                if cand > svals.get(parent_id, 0.0):
                    svals[parent_id] = cand
                if parent_id not in visited:
                    visited.add(parent_id)
                    queue.append(parent_id)

            # 2) Optional relationships with specific edge types
            if self.use_relationships and hasattr(term_child, "relationships"):
                rels = term_child.relationships or {}
                for rel_type, parents in rels.items():
                    w = self.rel2scf.get(rel_type, self.rel2scf.get("is_a", 0.8))
                    for parent_term in parents:
                        parent_id = parent_term.id
                        if parent_id not in nodes:
                            continue
                        cand = w * child_s
                        if cand > svals.get(parent_id, 0.0):
                            svals[parent_id] = cand
                        if parent_id not in visited:
                            visited.add(parent_id)
                            queue.append(parent_id)

        # Cache total sum for denominator
        self._svals[root] = svals
        self._sv_tot[root] = float(sum(svals.values()))
        return svals