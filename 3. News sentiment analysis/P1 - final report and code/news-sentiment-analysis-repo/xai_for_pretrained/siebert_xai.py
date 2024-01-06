import torch
from captum.attr import LayerIntegratedGradients, visualization, LimeBase
import torch.nn.functional as F
from captum._utils.models import SkLearnLasso

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(inputs, model, position_ids=None, attention_mask=None):
    output = model(inputs, position_ids=position_ids, attention_mask=attention_mask, )
    return output.logits


def construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
    return token_type_ids, ref_token_type_ids


def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def ig_attr(sentiment_pipeline, text, attr_label, return_convergence_delta=False):
    model = sentiment_pipeline.model
    tokenizer = sentiment_pipeline.tokenizer

    ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence

    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id,
                                                                cls_token_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)
    lig = LayerIntegratedGradients(predict, model.roberta.embeddings)

    attributions, delta = lig.attribute(inputs=input_ids,
                                        target=attr_label,
                                        baselines=ref_input_ids,
                                        additional_forward_args=(model, position_ids, attention_mask),
                                        n_steps=10,
                                        return_convergence_delta=True)

    if return_convergence_delta:
        return attributions, delta
    return attributions


def visualize_ig_attr(sentiment_pipeline, text, attributions, delta, attr_label, true_label):
    model = sentiment_pipeline.model
    tokenizer = sentiment_pipeline.tokenizer

    ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence

    attributions_sum = summarize_attributions(attributions)
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id,
                                                                cls_token_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    scores = predict(input_ids, model, attention_mask=attention_mask, position_ids=position_ids)
    vis = visualization.VisualizationDataRecord(
        word_attributions=attributions_sum,
        pred_prob=torch.max(torch.softmax(scores[0], dim=0)),
        pred_class=torch.argmax(scores[0]),
        true_class=true_label,
        attr_class=str(attr_label),
        attr_score=attributions_sum.sum(),
        raw_input_ids=all_tokens,
        convergence_score=delta)
    return visualization.visualize_text([vis])


def lime_attr(sentiment_pipeline, text, attr_label):
    model = sentiment_pipeline.model
    tokenizer = sentiment_pipeline.tokenizer

    def forward_func(text):
        try:
            out = model(text, attention_mask=torch.ones_like(text)).logits
        except IndexError:
            # perturbed to be 0 length
            out = torch.zeros(1, 2)  # match the shape of logits
        return out

    # encode text indices into latent representations & calculate cosine similarity
    def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
        original_emb = model.roberta.embeddings(original_inp).sum(dim=1)
        perturbed_emb = model.roberta.embeddings(perturbed_inp).sum(dim=1)
        distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
        return torch.exp(-1 * (distance ** 2) / 2)

    # binary vector where each word is selected independently and uniformly at random
    def bernoulli_perturb(text, **kwargs):
        probs = torch.ones_like(text) * 0.5
        return torch.bernoulli(probs).long()

    # remove absenst token based on the intepretable representation sample
    def interp_to_input(interp_sample, original_input, **kwargs):
        return original_input[interp_sample.bool()].view(original_input.size(0), -1)

    lasso_lime_base = LimeBase(
        forward_func,
        interpretable_model=SkLearnLasso(alpha=0.05),
        similarity_func=exp_embedding_cosine_distance,
        perturb_func=bernoulli_perturb,
        perturb_interpretable_space=True,
        from_interp_rep_transform=interp_to_input,
        to_interp_rep_transform=None
    )
    attrs = lasso_lime_base.attribute(
        torch.tensor(tokenizer.encode(text)).unsqueeze(0),
        target=attr_label,
        n_samples=100,
        show_progress=True
    ).squeeze(0)

    return attrs


def visualize_lime_attr(sentiment_pipeline, text, attributions, attr_label, true_label):
    tokenizer = sentiment_pipeline.tokenizer
    model = sentiment_pipeline.model

    ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence

    input_ids, ref_input_ids, _ = construct_input_ref_pair(tokenizer,text, ref_token_id, sep_token_id, cls_token_id)
    position_ids, _ = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    scores = predict(input_ids, model, attention_mask=attention_mask, position_ids=position_ids)

    vis = visualization.VisualizationDataRecord(
        word_attributions=attributions,
        pred_prob=torch.max(torch.softmax(scores[0], dim=0)),
        pred_class=torch.argmax(scores[0]),
        true_class=true_label,
        attr_class=str(attr_label),
        attr_score=attributions.sum(),
        raw_input_ids=all_tokens,
        convergence_score=None)

    print('\033[1m', 'Visualizations', '\033[0m')
    return visualization.visualize_text([vis])
