from utils.main_utils import tokenize

def read_sentences(corpus_location):
    f = open(corpus_location, "r")
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
    f.close()
    return lines

def read_labeled_tokens(corpus_location):
    pas_actual = []
    
    f = open(corpus_location, "r")
    lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if (i == 0):
            i += 1
        elif (lines[i] == "\n"):
            i += 2
        else:
            labeled_tokens = []
            while (i < len(lines) and lines[i] != "\n"):
                clean_line = lines[i].replace("\n", "")
                clean_line = clean_line.split(" ")
                labeled_tokens.append((clean_line[0], clean_line[1]))
                i += 1
            pas_actual.append(labeled_tokens)
    f.close()
    
    return pas_actual

def load_pas_corpus(nlp, corpus_test, pas_actual, corpus_docs):
    for corpus_name in corpus_test:
        pas_actual[corpus_name] = read_labeled_tokens("corpus_pas/" + corpus_name + ".txt")
        sentences = read_sentences("corpus_pas/" + corpus_name + "-sentences.txt")
        for sentence in sentences:
            doc = nlp(sentence)
            sent = tokenize(doc, -1, -1)
            corpus_docs[corpus_name]["sentences"].extend(sent)

def write_argument(title, subtitle, tokens, arguments):
    for idx_argument, argument in enumerate(arguments):
        for word in argument:
            tokens[word].name.tag = title

def write_result(f, extracted_pas, pas_prediction):
    write_argument("SUBJ", "Subjek ke: ", extracted_pas.tokens, extracted_pas.pas.subjects)
    write_argument("VERB", "Predikat ke: ", extracted_pas.tokens, extracted_pas.pas.predicates)
    write_argument("OBJ", "Objek ke: ", extracted_pas.tokens, extracted_pas.pas.objects)
    write_argument("ADJ", "Keterangan ke: ", extracted_pas.tokens, extracted_pas.pas.explanations)
    write_argument("TMP", "K. Waktu ke: ", extracted_pas.tokens, extracted_pas.pas.times)
    write_argument("LOC", "K. Tempat ke: ", extracted_pas.tokens, extracted_pas.pas.places)
    labeled_tokens = []
    for token in extracted_pas.tokens:
        labeled_tokens.append((token.name.text, token.name.tag))
        f.write(token.name.text + " " + token.name.tag + "\r\n")
    pas_prediction.append(labeled_tokens)
    f.write("\r\n")