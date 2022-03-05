from .main_utils import tokenize
from ...spansrl.src.features import SRLData
from tensorflow.keras.models import load_model


def load_srl_model(config):
    srl_data = SRLData(config)
    return load_model(config['srl_model']), srl_data

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

# MMR
def get_argument_tokens(arguments):
    tokens = []
    for argument in arguments:
        tokens.extend(argument)
    return tokens

def get_argument_tokens_without_punctuation(pas_tokens, arguments):
    tokens = []
    for argument in arguments:
    	for word in argument:
            if (pas_tokens[word].name.pos_tag not in exception_pos_tags):
                tokens.append(word)
    return tokens

def get_tokens(extracted_pas):
    tokens = []
    tokens.extend(get_argument_tokens(extracted_pas.pas.agent))
    tokens.extend(get_argument_tokens(extracted_pas.pas.verb))
    tokens.extend(get_argument_tokens(extracted_pas.pas.patient))
    tokens.extend(get_argument_tokens(extracted_pas.pas.location))
    tokens.extend(get_argument_tokens(extracted_pas.pas.temporal))
    tokens.extend(get_argument_tokens(extracted_pas.pas.goal))
    tokens.extend(get_argument_tokens(extracted_pas.pas.cause))
    tokens.extend(get_argument_tokens(extracted_pas.pas.extent))
    tokens.extend(get_argument_tokens(extracted_pas.pas.adverbial))
    tokens.extend(get_argument_tokens(extracted_pas.pas.modal))
    tokens.extend(get_argument_tokens(extracted_pas.pas.negation))
    tokens.sort()
    return tokens

def get_tokens_without_punctuation(extracted_pas):
    tokens = []
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.agent))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.verb))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.patient))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.location))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.temporal))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.goal))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.cause))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.extent))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.adverbial))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.modal))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.negation))
    tokens.sort()
    return tokens
  
def get_subjects_tokens(extracted_pas):
    tokens = get_argument_tokens(extracted_pas.pas.agent)
    tokens.sort()
    return tokens

def get_subjects(extracted_pas):
    tokens = get_subjects_tokens(extracted_pas)

    subjects = [extracted_pas.tokens[token].name.text for token in tokens]
    return " ".join(subjects)

def get_first_subject_tokens(extracted_pas):
    tokens = []
    if (len(extracted_pas.pas.subjects) > 0):
        for subj in extracted_pas.pas.subjects[0]:
            if extracted_pas.tokens[subj].name.pos_tag not in exception_pos_tags:
                tokens.append(subj)

    return tokens

def get_first_subject(extracted_pas):
    tokens = get_first_subject_tokens(extracted_pas)

    subjects = [extracted_pas.tokens[token].name.text for token in tokens]
    return " ".join(subjects)

def get_tokens_without_first_subject(extracted_pas):
    tokens = []
    if (len(extracted_pas.pas.subjects) > 1):
        tokens.extend(get_argument_tokens(extracted_pas.pas.subjects[1:len(extracted_pas.pas.subjects)]))

    tokens.extend(get_argument_tokens(extracted_pas.pas.predicates))
    tokens.extend(get_argument_tokens(extracted_pas.pas.objects))
    tokens.extend(get_argument_tokens(extracted_pas.pas.times))
    tokens.extend(get_argument_tokens(extracted_pas.pas.places))
    tokens.extend(get_argument_tokens(extracted_pas.pas.explanations))
    tokens.sort()
    return tokens

def get_tokens_without_first_subject_without_punctuation(extracted_pas):
    tokens = []
    if (len(extracted_pas.pas.subjects) > 1):
        tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.subjects[1:len(extracted_pas.pas.subjects)]))

    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.predicates))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.objects))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.times))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.places))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.explanations))
    tokens.sort()
    return tokens

def print_argument(title, subtitle, tokens, arguments):
    print(title)
    for idx_argument, argument in enumerate(arguments):
        print(subtitle + str(idx_argument))
        for word in argument:
            print(tokens[word].name)
    print()

def print_result(extracted_pas):
    print_argument("Subjek", "Subjek ke: ", extracted_pas.tokens, extracted_pas.pas.subjects)
    print_argument("Predikat", "Predikat ke: ", extracted_pas.tokens, extracted_pas.pas.predicates)
    print_argument("Objek", "Objek ke: ", extracted_pas.tokens, extracted_pas.pas.objects)
    print_argument("Keterangan", "Keterangan ke: ", extracted_pas.tokens, extracted_pas.pas.explanations)
    print_argument("K. Waktu", "K. Waktu ke: ", extracted_pas.tokens, extracted_pas.pas.times)
    print_argument("K. Tempat", "K. Tempat ke: ", extracted_pas.tokens, extracted_pas.pas.places)

def print_clean_result(extracted_pas):
    # print_argument("Subjek", "Subjek ke: ", extracted_pas.tokens, extracted_pas.pas.subjects)
    print_argument("Subjek (clean)", "Subjek ke: ", extracted_pas.tokens, extracted_pas.clean_pas.subjects)
    
    # print_argument("Predikat", "Predikat ke: ", extracted_pas.tokens, extracted_pas.pas.predicates)
    print_argument("Predikat (clean)", "Predikat ke: ", extracted_pas.tokens, extracted_pas.clean_pas.predicates)
    
    # print_argument("Objek", "Objek ke: ", extracted_pas.tokens, extracted_pas.pas.objects)
    print_argument("Objek (clean)", "Objek ke: ", extracted_pas.tokens, extracted_pas.clean_pas.objects)
    
    # print_argument("Keterangan", "Keterangan ke: ", extracted_pas.tokens, extracted_pas.pas.explanations)
    print_argument("Keterangan (clean)", "Keterangan ke: ", extracted_pas.tokens, extracted_pas.clean_pas.explanations)
    
    # print_argument("K. Waktu", "K. Waktu ke: ", extracted_pas.tokens, extracted_pas.pas.times)
    print_argument("K. Waktu (clean)", "K. Waktu ke: ", extracted_pas.tokens, extracted_pas.clean_pas.times)
    
    # print_argument("K. Tempat", "K. Tempat ke: ", extracted_pas.tokens, extracted_pas.pas.places)
    print_argument("K. Tempat (clean)", "K. Tempat ke: ", extracted_pas.tokens, extracted_pas.clean_pas.places)

def min_start_arg(list_labels) :
    """
    Return the minimum element of a sequence.
    key_func is an optional one-argument ordering function.
    """
    
    minimum = list_labels[0]
    for item in list_labels :
        if item[1] < minimum[1] :
            minimum = item

    min_start = minimum[1]
    return min_start