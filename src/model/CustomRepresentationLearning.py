import numpy as np
import torch

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.language_representation.representation_model import batch_iterable, mean_across_all_tokens, \
    concat_all_tokens
from tqdm import tqdm


class CustomRepresentationModel(RepresentationModel):
    '''
    This is a customized version of the RepresentationModel class from simpletransformers.
    This version implements an encoding method that returns the embedding for the [CLS] token,
    plus a few performance optimization.
    '''
    def encode_sentences(self, text_list, combine_strategy=None, batch_size=32):
        """
        Generates list of contextual word or sentence embeddings using the model passed to class constructor
        :param text_list: list of text sentences
        :param combine_strategy: strategy for combining word vectors, supported values: None, "mean", "concat", or int value to select a specific embedding (e.g. 0 for [CLS] or -1 for the last one)
        :param batch_size
        :return: list of lists of sentence embeddings(if `combine_strategy=None`) OR list of sentence embeddings(if `combine_strategy!=None`)
        """

        self.model.to(self.device)

        batches = batch_iterable(text_list, batch_size=batch_size)
        embeddings = list()
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(batches, total=np.ceil(len(text_list)/batch_size)):
                input_ids_tensor = self._tokenize(batch)
                input_ids_tensor = input_ids_tensor.to(self.device)
                token_vectors = self.model(input_ids=input_ids_tensor)
                if combine_strategy is not None:
                    if type(combine_strategy)==int:
                        embedding_func = lambda x: x[:,combine_strategy,:]
                    else:
                        embedding_func_mapping = {"mean": mean_across_all_tokens, "concat": concat_all_tokens}
                        try:
                            embedding_func = embedding_func_mapping[combine_strategy]
                        except KeyError:
                            raise ValueError(
                                "Provided combine_strategy is not valid." "supported values are: 'concat', 'mean' and None."
                            )
                    batch_embeddings = embedding_func(token_vectors).cpu().detach().numpy() 
                    embeddings.append(batch_embeddings)
                else:
                    batch_embeddings = token_vectors.detach()
                    embeddings.append(batch_embeddings.cpu().detach().numpy())
        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
