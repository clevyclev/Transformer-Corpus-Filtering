import torch.nn as nn
import torch.nn.functional as F
from fairseq.data import Dictionary
from fairseq.models import (
	FairseqDecoder,
	FairseqLanguageModel,
	register_model,
	register_model_architecture,
)
from fairseq.models.transformer import (
	DEFAULT_MIN_PARAMS_TO_WRAP, Embedding, TransformerDecoder
)
from fairseq.models.transformer_lm import (
	TransformerLanguageModelConfig
)


@register_model('npi_transformer')
class NpiTransformer(FairseqLanguageModel):
	def __init__(self, decoder):
		super().__init__(decoder)

	@classmethod
	def build_model(cls, args, task):
		"""Build a new model instance."""

		embed_tokens = cls.build_embedding(
			args, task.source_dictionary, args.decoder_input_dim
		)
		print(embed_tokens)
		print(args)

		decoder = TransformerDecoder(
			args, task.target_dictionary, embed_tokens, no_encoder_attn=True
		)
		return cls(decoder)

	@classmethod
	def build_embedding(cls, args, dictionary, embed_dim, path=None):
		embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
		return embed_tokens

	@staticmethod
	def add_args(parser):
		# Models can override this method to add new command-line arguments.
		# Here we'll add a new command-line argument to configure the
		# dimensionality of the hidden state.
		parser.add_argument(
			'--decoder-layers', type=int, metavar='N'
		)
		parser.add_argument(
			'--decoder-input-dim', type=int, metavar='N'
		)
		parser.add_argument(
			'--decoder-ffn-embed-dim', type=int, metavar='N'
		)
		parser.add_argument(
			'--decoder-embed-dim', type=int, metavar='N'
		)
		parser.add_argument(
			'--decoder-attention-heads', type=int, metavar='N'
		)
