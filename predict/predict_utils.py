def prepare_instances(xseq):
	features = []
	for interaction in xseq:
		token_dict = {feat.split('=')[0]:feat.split('=')[1] for feat in interaction[0:]}
		features.append(token_dict)
	return features