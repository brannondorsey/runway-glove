import codecs
import faiss
import runway
import numpy as np
from runway import category, number, text

def get_label_dictionaries(labels_array):
    id_to_word = dict(zip(range(len(labels_array)), labels_array))
    word_to_id = dict((v,k) for k,v in id_to_word.items())
    return word_to_id, id_to_word

def build_word_vector_matrix(vector_file, n_words):
    '''Read a GloVe array from file and return its vectors and labels as arrays'''
    np_arrays = []
    labels_array = []

    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            sr = line.split()
            labels_array.append(sr[0])
            np_arrays.append(np.array([float(j) for j in sr[1:]]))
            if i == n_words - 1:
                return np.array(np_arrays, dtype=np.float32), labels_array

def find_nearest(excluded_words, vec, id_to_word, faiss_index, num_results):
    D, I = faiss_index.search(np.expand_dims(vec, axis=0), num_results + len(excluded_words))
    return [id_to_word[i] for i in I[0] if id_to_word[i] not in excluded_words]

def parse_arithmetic_expression(expr):
    split = expr.split()
    start_word = split[0]
    minus_words, plus_words = [], []
    for i, token in enumerate(split[1:]):
        if token == '+':
            plus_words.append(split[i + 2])
        elif token == '-':
            minus_words.append(split[i + 2])
    return start_word, minus_words, plus_words

def word_arithmetic(start_word, minus_words, plus_words, word_to_id, id_to_word, df, faiss_index, num_results=5):
    '''Returns a word string that is the result of the vector arithmetic'''
    try:
        start_vec  = df[word_to_id[start_word]]
        minus_vecs = [df[word_to_id[minus_word]] for minus_word in minus_words]
        plus_vecs  = [df[word_to_id[plus_word]] for plus_word in plus_words]
    except KeyError as err:
        return '{} not found in the dataset.'.format(err)

    result = start_vec

    if minus_vecs:
        for i, vec in enumerate(minus_vecs):
            result = result - vec

    if plus_vecs:
        for i, vec in enumerate(plus_vecs):
            result = result + vec

    excluded_words = [start_word] + minus_words + plus_words
    neighbors = find_nearest(excluded_words, result, id_to_word, faiss_index, num_results)
    return ', '.join(neighbors)

setup_options = {
    'word_vector_dimensions': category(
        choices=['50', '100', '200', '300'],
        default='100',
        description='The number of dimensions used to represent each word in the latent '\
            'space. Higher dimensions increase accuracy but take longer to run and use '\
            'more memory.'
    ),
    'number_of_words': number(
        min=100,
        max=400000,
        default=400000,
        description='The number of words in the corpus. More words create more variety '\
            ' but take longer to run.'
    )
}
@runway.setup(options=setup_options)
def setup(opts):
    dimensions = int(opts['word_vector_dimensions'])
    vector_file = 'data/glove/glove.6B.{}d.txt'.format(dimensions)
    df, labels_array = build_word_vector_matrix(vector_file, opts['number_of_words'])
    word_to_id, id_to_word = get_label_dictionaries(labels_array)
    faiss_index = faiss.IndexFlatL2(dimensions)
    faiss_index.add(df)
    print('is trained: {}'.format(faiss_index.is_trained))
    print('total: {}'.format(faiss_index.ntotal))
    return dict(faiss_index=faiss_index, df=df, word_to_id=word_to_id, id_to_word=id_to_word, labels_array=labels_array)

nearest_neighbor_description='Find words that are contextually similar to an input word.'
nearest_neighbor_inputs = {
    'word': text(description='The input word.'),
    'number_of_neighbors': number(
        min=1,
        max=100,
        default=10,
        description='The number of neighbors to return. Words closer to the input word '\
                    'appear earlier in the list')
}
nearest_neighbor_outputs = {
    'output': text(description='A list of neighbor words that are similar to the input word')
}
@runway.command('nearest_neighbor',
                description=nearest_neighbor_description,
                inputs=nearest_neighbor_inputs,
                outputs=nearest_neighbor_outputs)
def nearest_neighbor(model, args):
    word = str(args['word']).lower()
    try:
        word_vec = model['df'][model['word_to_id'][word]]
    except KeyError as err:
        return "Error: '{}' was not found in the dictionary.".format(word)
    ignored_words = [word]
    neighbors = find_nearest(ignored_words,
                             word_vec,
                             model['id_to_word'],
                             model['faiss_index'],
                             args['number_of_neighbors'])
    return ', '.join(neighbors)

word_arithmetic_description='Find words that are contextually similar to an input word.'
word_arithmetic_inputs = {
    'expression': text(description='An input expression consisting of words, +s, and -s'),
    'number_of_neighbors': number(
        min=1,
        max=100,
        default=10,
        description='The number of neighbors to return. Words closer to the input word '\
                    'appear earlier in the list')
}
word_arithmetic_outputs = {
    'output': text(description='A list of neighbor words that are near the evaluated word expression')
}
@runway.command('word_arithmetic',
                description=nearest_neighbor_description,
                inputs=word_arithmetic_inputs,
                outputs=word_arithmetic_outputs)
def word_arithmetic_(model, args):
    expression = str(args['expression']).lower()
    start_word, minus_words, plus_words = parse_arithmetic_expression(expression)
    return word_arithmetic(start_word,
                           minus_words,
                           plus_words,
                           model['word_to_id'],
                           model['id_to_word'],
                           model['df'],
                           model['faiss_index'],
                           args['number_of_neighbors'])

if __name__ == '__main__':
    runway.run()
