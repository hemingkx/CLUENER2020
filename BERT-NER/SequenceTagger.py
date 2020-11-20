from transformers.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence


class BertForSequenceTagging(BertPreTrainedModel):
	def __init__(self, config):
		super(BertForSequenceTagging, self).__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()

	def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
				position_ids=None, inputs_embeds=None, head_mask=None):
		input_ids, input_token_starts = input_data
		#print("input_ids", input_ids.shape)
		#print("input_token_starts", input_token_starts.shape)
		#print("attention_mask", attention_mask.shape)
		#print("labels", labels.shape)
		outputs = self.bert(input_ids,
							attention_mask=attention_mask,
							token_type_ids=token_type_ids,
							position_ids=position_ids,
							head_mask=head_mask,
							inputs_embeds=inputs_embeds)
		sequence_output = outputs[0]
		#print("sequence_output", sequence_output.shape)

		#### 'X' label Issue Start ####
		# obtain original token representations from sub_words representations (by selecting the first sub_word)
		origin_sequence_output = [
			layer[starts.nonzero().squeeze(1)]
			for layer, starts in zip(sequence_output, input_token_starts)]
		padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
		#print("padded_sequence_output", padded_sequence_output.shape)
		padded_sequence_output = self.dropout(padded_sequence_output)
		#### 'X' label Issue End ####

		logits = self.classifier(padded_sequence_output)

		outputs = (logits,)
		if labels is not None:
			loss_mask = labels.gt(-1)
			loss_fct = CrossEntropyLoss()
			# Only keep active parts of the loss
			if loss_mask is not None:
				active_loss = loss_mask.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)[active_loss]
				active_labels = labels.view(-1)[active_loss]
				loss = loss_fct(active_logits, active_labels)
			else:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs  # (loss), scores
