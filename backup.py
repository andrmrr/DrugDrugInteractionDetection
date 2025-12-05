# lcs verb class
clss = ""
for k in verbs.keys():
    if lcs_lemma in verbs[k]:
        clss = k
        break
feats.add("verbclss=" + clss)


def verb_extraction(datadir):
    verbs = {}
    used = {}
    for f in listdir(datadir):

        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)

        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            # analyze sentence
            stext = s.attributes["text"].value  # get sentence text
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

                # target entities
                e1 = p.attributes["e1"].value
                e2 = p.attributes["e2"].value
                tkE1 = analysis.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
                tkE2 = analysis.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

                # put lcs lemma in the dict
                lcs = analysis.get_LCS(tkE1, tkE2)
                if analysis.get_tag(lcs)[0] == 'V' and tkE1 != tkE2:
                    lemma = analysis.get_lemma(lcs)
                    if dditype not in verbs: verbs[dditype] = set()
                    if lemma not in used:
                        verbs[dditype].add(lemma)
                        used[lemma] = True
                    else:
                        if used[lemma] is True:
                            used[lemma] = False
                            for k in verbs.keys():
                                if lemma in verbs[k]:
                                    verbs[k].remove(lemma)
                                    break
    return verbs
