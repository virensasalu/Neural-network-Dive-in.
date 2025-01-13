"""

The main code for the Strings-to-Vectors.
"""
#created an environment and ran the requirements.
from typing import Sequence, Any

import numpy as np


class Index:
    """
    Represents a mapping from a vocabulary (e.g., strings) to integers.
    """

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        #### YOUR CODE HERE ####
        #In this code I have created dictionaries to map vocabulary items to indices and vice versa.
        #Starting from a given index, it iterates over the vocabulary,
        #Assigning unique indices to each item and storing these mappings
        #in both vocab_to_index and index_to_vocab dictionaries.

        self.vocab_to_index = {}
        self.index_to_vocab = {}
        self.start = start
        current_index = start
        for item in vocab:
            if item not in self.vocab_to_index:
                self.vocab_to_index[item] = current_index
                self.index_to_vocab[current_index] = item
                current_index += 1

    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array of the object indexes.
        """

        #### YOUR CODE HERE ####
        #In this code, I created an array called indexes using NumPy.
        #It maps each object in object_seq to its corresponding index
        #from the vocab_to_index dictionary.
        #If an object isn't found in the dictionary,
        #I assign it a default index of self.start - 1.
        #Finally, the indexes array is returned.

        indexes = np.array([self.vocab_to_index.get(obj, self.start-1) for obj in object_seq])
        return indexes


    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array of the object indexes.
        """

        #### YOUR CODE HERE ####
        #In this code, I first created a fallback_index as one
        #less than the minimum index in vocab_to_index.
        #Then, I calculated the max_length of the sequences in object_seq_seq.
        #I initialized an index_matrix filled with fallback_index values
        #and sized according to the number of sequences and their maximum length.
        #For each sequence, I conveteed objects to their corresponding indices
        #using vocab_to_index, falling back to fallback_index when necessary,
        #and populated the matrix. Finally, the index_matrix is returned.

        fallback_index = min(self.vocab_to_index.values()) - 1
        max_length = max(len(seq) for seq in object_seq_seq)
        index_matrix = np.full((len(object_seq_seq), max_length), fallback_index)

        for i, seq in enumerate(object_seq_seq):
            indexes = [self.vocab_to_index.get(obj, fallback_index) for obj in seq]
            index_matrix[i, :len(indexes)] = indexes

        return index_matrix

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
        #### YOUR CODE HERE ####
        #In this code, I created a binary_vector initialized with zeros,
        #with a length equal to the size of vocab_to_index plus self.start.
        #I then iterated over the unique objects in object_seq. For each object,
        #I retrieved its corresponding index from vocab_to_index. If the index exists,
        #I set the corresponding position in binary_vector to 1.
        #Finally, the binary_vector is returned.

        binary_vector = np.zeros(len(self.vocab_to_index)+self.start, dtype=int)

        for obj in set(object_seq):
            index = self.vocab_to_index.get(obj)
            if index is not None:
                binary_vector[index] = 1

        return binary_vector

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
        #### YOUR CODE HERE ####
        #In this code, I initialized an empty list called output.
        #Then, for each sequence in object_seq_seq,
        #I converted the sequence to a binary vector using the objects_to_binary_vector method and
        #appended the result to the output list. Finally, I returned the output as a NumPy array.

        output = []
        for seq in object_seq_seq:
            output.append(self.objects_to_binary_vector(seq))

        return np.array(output)


    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """
        #### YOUR CODE HERE ####
        #Here, I created a dictionary index_to_vocab by inverting vocab_to_index,
        #mapping indices back to objects.
        #Then, I generaetd a list of objects by looking up each index in
        #index_vector using index_to_vocab,
        #returning None if an index isn't found.
        #I returned a list of objects, filtering out any None values.

        index_to_vocab = {index: object for object, index in self.vocab_to_index.items()}
        objects = [index_to_vocab.get(index, None) for index in index_vector]
        return [obj for obj in objects if obj is not None]

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """
        #### YOUR CODE HERE ####
        #Here the  code returns a list by applyig the
        #indexes_to_objects method to each sequence in index_matrix,
        #converting them to objects.
        #The transformed sequences are then collected and returned as a list.

        return [self.indexes_to_objects(seq) for seq in index_matrix]

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """
        #### YOUR CODE HERE ####
        #In this code, I find the indices of non-zero elements in vector and
        #retrieve their corresponding vocabulary items using index_to_vocab.
        # If an index is not found, "skipped" is used as a default value.
        #The resulting vocabulary items are returned as a list.

        nonzero_indexes = np.nonzero(vector)[0]
        return [self.index_to_vocab.get(idx, "skipped") for idx in nonzero_indexes]

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """
        #### YOUR CODE HERE ####
        #FInally, I created an empty list, object_sequences.
        #For each row in binary_matrix, I identified the indices of non-zero elements,
        #retrieved the corresponding vocabulary items using index_to_vocab,
        #and used "skiped" for any missing indices.
        #I then appended these items as a list to object sequences.
        #Finally, I returned the list of object sequences.

        object_sequences = []
        for row in binary_matrix:
            nonzero_indexes = np.nonzero(row)[0]
            objects = [self.index_to_vocab.get(idx, "skipped") for idx in nonzero_indexes]
            object_sequences.append(objects)
        return object_sequences
