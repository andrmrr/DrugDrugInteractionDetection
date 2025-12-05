#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse

from deptree import *


# import patterns


## -------------------
## -- Convert a pair of drugs and their context in a feature vector

def extract_features(tree, entities, e1, e2):
    feats = set()

    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

    # check if it's equal, since that's an error of bad offset parsing (e.g. 51-59;70-84 -> start: 51, end: 84)
    if tkE1 is not None and tkE2 is not None and tkE1 != tkE2:
        # make sure tkE1 is the first token
        if tkE1 > tkE2:
            t = tkE1
            tkE1 = tkE2
            tkE2 = t

        # features for tokens in between E1 and E2
        tk = tkE1
        realTokens = []
        nstopw = 0
        while tk <= tkE2:
            try:
                while True:
                    tk += 1
                    if tk > tkE2:
                        break
                    if not tree.is_stopword(tk):
                        realTokens.append(tk)
                        break
                    else:
                        nstopw += 1
            except Exception as e:
                raise e

        for i in range(3):
            if len(realTokens) <= i:
                word = ""
                lemma = ""
                tag = ""
                eib = False
            else:
                word = tree.get_word(realTokens[i])
                lemma = tree.get_lemma(realTokens[i]).lower()
                tag = tree.get_tag(realTokens[i])
                if tree.is_entity(realTokens[i], entities):
                    eib = True
                else:
                    eib = False

            feats.add("lib" + str(i) + "=" + lemma)
            feats.add("wib" + str(i) + "=" + word)
            feats.add("lpib" + str(i) + "=" + lemma + "_" + tag)
            feats.add("eib" + str(i) + "=" + str(eib))


        # feature for token before E1
        tk = tkE1 - 1
        while tk >= 0 and tree.is_stopword(tk):
            tk -= 1
        if tk < 0:
            word = ""
            lemma = ""
            tag = ""
            eib = False
        else:
            word = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            if tree.is_entity(tk, entities):
                eib = True
            else:
                eib = False

        feats.add("libprev=" + lemma)
        feats.add("wibprev=" + word)
        feats.add("lpibprev=" + lemma + "_" + tag)
        feats.add("eibprev=" + str(eib))


        # feature for token after E2
        tk = tkE2 + 1
        treeLen = tree.get_n_nodes()
        while tk < treeLen and tree.is_stopword(tk):
            tk += 1
        if tk == treeLen:
            word = ""
            lemma = ""
            tag = ""
            eib = False
        else:
            word = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            if tree.is_entity(tk, entities):
                eib = True
            else:
                eib = False

        feats.add("libnext=" + lemma)
        feats.add("wibnext=" + word)
        feats.add("lpibnext=" + lemma + "_" + tag)
        feats.add("eibnext=" + str(eib))


        # feature indicating the presence of an entity in between E1 and E2
        eib = False
        for tk in range(tkE1 + 1, tkE2):
            if tree.is_entity(tk, entities):
                eib = True
        feats.add('eib=' + str(eib))

        # features about paths in the tree
        lcs = tree.get_LCS(tkE1, tkE2)
        word = tree.get_word(lcs)
        lemma = tree.get_lemma(lcs).lower()
        lcs_lemma = lemma
        tag = tree.get_tag(lcs)
        if tree.is_entity(lcs, entities):
            eib = True
        else:
            eib = False
        feats.add("liblcs=" + lemma)
        feats.add("wiblcs=" + word)
        feats.add("lpiblcs=" + lemma + "_" + tag)
        feats.add("eiblcs=" + str(eib))

        path1 = tree.get_up_path(tkE1, lcs)
        lp1 = len(path1)
        path1 = "<".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path1])
        feats.add("path1=" + path1)

        path2 = tree.get_down_path(lcs, tkE2)
        lp2 = len(path2)
        path2 = ">".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path2])
        feats.add("path2=" + path2)

        path = path1 + "<" + tree.get_lemma(lcs) + "_" + tree.get_rel(lcs) + ">" + path2
        feats.add("path=" + path)

        feats.add("dist=" + str(tkE2 - tkE1))
        # feats.add("pathlen=" + str(lp1 + lp2))
        # feats.add("sentlen=" + str(treeLen))
        # feats.add("nstopw=" + str(nstopw))

        # entitiy pair information
        feats.add("typ1=" + entities[e1]["type"])
        feats.add("typ2=" + entities[e2]["type"])

        # connecting verb class
        clss = ""
        for k in verbs.keys():
            if lcs_lemma in verbs[k]:
                clss = k
                break
        feats.add("verbclss=" + clss)

    return feats


def verb_extraction(datadir):
    verbs = {}
    used = set()
    for f in listdir(datadir):
        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)

        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            # analyze sentence
            stext = s.attributes["text"].value
            analysis = deptree(stext)

            # parse entities
            entities = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                id = e.attributes["id"].value
                offs = e.attributes["charOffset"].value.split("-")
                entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

            # for each pair in the sentence, decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                # ground truth
                ddi = p.attributes["ddi"].value
                if (ddi == "true"):
                    dditype = p.attributes["type"].value
                else:
                    dditype = "null"
                    # continue

                # target entities
                e1 = p.attributes["e1"].value
                e2 = p.attributes["e2"].value
                tkE1 = analysis.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
                tkE2 = analysis.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

                # put lcs lemma in the dict
                lcs = analysis.get_LCS(tkE1, tkE2)
                if lcs is not None and analysis.get_tag(lcs)[0] == 'V' and tkE1 != tkE2:
                    lemma = analysis.get_lemma(lcs)
                    if dditype not in verbs: verbs[dditype] = set()
                    if lemma not in used:
                        found = False
                        for k in verbs.keys():
                            if dditype == k: continue
                            if lemma in verbs[k]:
                                found = True
                                used.add(lemma)
                                verbs[k].remove(lemma)
                        if not found:
                            verbs[dditype].add(lemma)

    return verbs


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
datadir = sys.argv[1]
hsdb_file = sys.argv[2]

# tokenize HSDB
f = open(hsdb_file, "r")
hsdb = set()
for line in f.readlines():
    for item in line.split():
        hsdb.add(item.strip().lower())
f.close()

# create a list of verb lemmas that are LCS of certain interaction class
# duplicates are removed from every class verbs
print("Verb extraction...", file=sys.stderr)
verbs = verb_extraction(datadir)
print("Verb extraction finished...", file=sys.stderr)

# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir + "/" + f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        stext = s.attributes["text"].value  # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents:
            id = e.attributes["id"].value
            typ = e.attributes["type"].value
            offs = e.attributes["charOffset"].value.split("-")
            entities[id] = {'start': int(offs[0]), 'end': int(offs[-1]), 'type': typ}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1: continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi == "true"):
                dditype = p.attributes["type"].value
            else:
                dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction

            feats = extract_features(analysis, entities, id_e1, id_e2)
            # resulting vector
            if len(feats) != 0:
                print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")
