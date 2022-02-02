from anytree import Node, RenderTree
import pickle, re, stanfordnlp
from models.sentence import Sentence
from models.token import Token
import networkx as nx

exception_pos_tags = ["PUNCT", "SYM", "X"]

def initialize_stanford_nlp():
    nlp = stanfordnlp.Pipeline(lang="id", treebank="id_gsd")
    return nlp

def print_tree(root):
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))


def preprocess(content):
    result = content.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").replace("–", "-") # replace non-ascii double and single quote
    result = re.sub(r"[^\x00-\x7f]", " ", result) # delete non ascii character
    # result = re.sub(r'"[^"]+"[^\."\n]*\.', '', result) # remove quoted sentence
    result = result.replace('"', '').replace("'", "") # remove quote
    result = re.sub(r'"[^"]+"', '', result) # remove quoted sentence
    result = re.sub(r"\.\s+((\([^\)]+\)\s+\[[^\]]+\])|(\([^\)]+\)))", ".", result) # delete author and recommendation at the end of paragraph
    result = result.replace("--", " ") # delete double dash
    return result

def preprocess_summary(content):
#     result = preprocess(content)
    result = content.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").replace("–", "-") # replace non-ascii double and single quote
    result = re.sub(r"[^\x00-\x7f]", " ", result) # delete non ascii character
    result = re.sub("(\d+)\s((\.)|(,)|(-)|(\/))\s(\d+)", r"\1\2\7", result) # digit with . and , and - and /
    result = result.replace("--", " ") # delete double dash
    result = result.replace('"', '').replace("'", "") # remove quote
    result = result.replace(" .", ".")
    result = result.replace(" ,", ",")
    result = result.replace(" - ", "-")
    # result = result.replace(" / ", "/")    
    # print(result)
    return result

def tokenize(doc, idx_topic, idx_news):
    sentences = []
    for idx_sentence, sentence in enumerate(doc.sentences):
        tokens = []
        token_words = []
        for word in sentence.words:
            tok = Node(Token(int(word.index), word.text, word.upos, word.dependency_relation, word.governor))
            tokens.append(tok)
            token_words.append(word.text)

        root = None
        for token in tokens:
            governor = token.name.governor
            if (governor > 0):
                token.parent = tokens[governor - 1]
            else:
                root = token

        sentence_obj = Sentence(idx_topic, idx_news, idx_sentence, sentence, tokens, root, len(doc.sentences))
        sentences.append(sentence_obj)

    return sentences

def tokenize_summary(doc):
    sentences = []
    for sentence in doc.sentences:
        tokens = []
        # print("sent: ")
        for word in sentence.words:
            # print(word.text)
            tokens.append(word.text)
        sentences.append(tokens)
    return sentences
    
def save_object(file_dir, obj):
    file_dest = file_dir + ".dat" 
    with open(file_dest, "wb") as f:
        pickle.dump(obj, f)

def load_object(file_dir):
    file_src = file_dir + ".dat"
    with open(file_src, "rb") as f:
        obj = pickle.load(f)
        return obj

def find_whole_word(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def save_graph(file_dir, obj):
    nx.write_gpickle(obj, file_dir + ".gpickle")

def load_graph(file_dir):
    return nx.read_gpickle(file_dir + ".gpickle")

def load_corpus(corpus_topics, corpus_pas, sentence_similarity_table, corpus_summaries_100, corpus_summaries_200, graph_docs):
    for corpus_name in (corpus_topics["test"] + corpus_topics["train"] + corpus_topics["validation"]): 
        corpus_pas[corpus_name] = load_object("../../temporary_data/corpus_pas/" + corpus_name)
        sentence_similarity_table[corpus_name] = load_object("../../temporary_data/sentence_similarity_table/" + corpus_name)
        corpus_summaries_100[corpus_name] = load_object("../../temporary_data/corpus_summaries_100/" + corpus_name)
        # corpus_summaries_200[corpus_name] = load_object("../../temporary_data/corpus_summaries_200/" + corpus_name)
        graph_docs[corpus_name] = load_graph("../../temporary_data/graph_docs/" + corpus_name)

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