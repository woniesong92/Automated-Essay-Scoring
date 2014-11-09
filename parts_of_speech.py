import re
tag_dict = {'CC':0,'CD':1,'DT':2,'EX':3,'FW':4,'IN':5,'JJ':6,'JJR':7,'JJS':8,
            'LS':9,'MD':10,'NN':11,'NNS':12,'NNP':13,'NNPS':14,'PDT':15,
            'POS':16,'PRP':17,'PRP$':18,'RB':19,'RBR':20,'RBS':21,'RP':22,
            'SYM':23,'TO':24,'UH':25,'VB':26,'VBD':27,'VBG':28,'VBN':29,
            'VBP':30,'VBZ':31,'WDT':32,'WP':33,'WP$':34,'WRB':35,'.':36,',':37,
            ':':38,'``':39,"''":40,'#':41,'$':42,'-NONE-':43}
            
def remove_symbols(essay):
    return re.sub(r'[^\w+.+\`+\'+#+!+,+$+:]', ' ', essay)

import nltk
def main():
    f1 = open('data/training_set_rel3.tsv','r')
    f2 = open('training_set_tagged_count.tsv','w')
    f1.readline()
    for line in f1:
        line_break = line.strip("\n").strip("\t").split("\t")
        text = line_break[2].strip('"')
        text = remove_symbols(text)
        #sentence_split = nltk.data.load('tokenizers/punkt/english.pickle')
        text = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(text)
        count = [0]*len(tag_dict)
        for token in tagged:
            count[tag_dict[token[1]]] += 1
        f2.write(str(line_break[0])+"\t"+str(line_break[1])+"\t")
        print str(line_break[0])
        for c in count:
            f2.write(str(c)+"\t")
        f2.write("\t"+line_break[6]+"\n")
        #tagged = nltk.tag.pos_tag_sents(sentences)
        #tokens = nltk.word_tokenize(line_break[2].strip('"'))
        #tagged = nltk.pos_tag(tokens)
        #print tagged
    f2.close()
        
if __name__ == '__main__': main()