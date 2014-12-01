from feature_parser import *
fp = FeatureParser("info.tsv")
fp.parse()
fp.write_to_file()