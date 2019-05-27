import sys
import utils
import faiss
import runway
import numpy as np
from runway import category, number, text
from scipy.spatial.distance import cosine

def find_nearest(excluded_words, vec, id_to_word, faiss_index, num_results):
    D, I = faiss_index.search(np.expand_dims(vec, axis=0), num_results + len(excluded_words))
    return [id_to_word[i] for i in I[0] if id_to_word[i] not in excluded_words]

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
    df, labels_array = utils.build_word_vector_matrix(vector_file, opts['number_of_words'])
    word_to_id, id_to_word = utils.get_label_dictionaries(labels_array)
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

@runway.command('nearest_neighbor', inputs=nearest_neighbor_inputs, outputs=nearest_neighbor_outputs)
def command(model, args):
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

if __name__ == '__main__':
    runway.run()
