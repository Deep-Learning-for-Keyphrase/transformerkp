from dataclasses import dataclass, field


def aggregate_kp_confidence_score(scores=None, score_method=None):
    assert scores and score_method
    if score_method == "avg":
        return float(sum(scores) / len(scores))
    elif score_method == "first":
        return scores[0]
    elif score_method == "max":
        return max(scores)


def extract_kp_from_tags(
    token_ids, tags, tokenizer, scores=None, score_method=None, return_offsets=False
):
    if score_method:
        assert len(tags) == len(
            scores
        ), "Score is not none and len of score is not equal to tags"
    all_kps = []
    all_kps_score = []
    all_kps_offset = []
    current_kp = []
    current_score = []
    current_kp_offset = []

    for i, (id, tag) in enumerate(zip(token_ids, tags)):
        if tag == "O" and len(current_kp) > 0:  # current kp ends
            if score_method:
                confidence_score = aggregate_kp_confidence_score(
                    scores=current_score, score_method=score_method
                )
                current_score = []
                all_kps_score.append(confidence_score)

            all_kps.append(current_kp)
            current_kp = []
            if return_offsets:
                all_kps_offset.append(current_kp_offset)
                current_kp_offset = []
        elif tag == "B":  # a new kp starts
            if len(current_kp) > 0:
                if score_method:
                    confidence_score = aggregate_kp_confidence_score(
                        scores=current_score, score_method=score_method
                    )
                    all_kps_score.append(confidence_score)
                all_kps.append(current_kp)
                all_kps_offset.append(current_kp_offset)
            current_kp = []
            current_score = []
            current_kp.append(id)
            if return_offsets:
                current_kp_offset = []
                current_kp_offset.append(i)
            if score_method:
                current_score.append(scores[i])
        elif tag == "I":  # it is part of current kp so just append
            current_kp.append(id)
            if return_offsets:
                current_kp_offset.append(i)
            if score_method:
                current_score.append(scores[i])
    if len(current_kp) > 0:  # check for the last KP in sequence
        all_kps.append(current_kp)
        if return_offsets:
            all_kps_offset.append(current_kp_offset)
        if score_method:
            confidence_score = aggregate_kp_confidence_score(
                scores=current_score, score_method=score_method
            )
            all_kps_score.append(confidence_score)
    all_kps = tokenizer.batch_decode(
        all_kps,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    final_kps, final_score, final_offsets = [], [], []
    kps_set = {}
    for i, kp in enumerate(all_kps):
        if kp.lower() not in kps_set:
            final_kps.append(kp.lower())
            kps_set[kp.lower()] = -1
            if return_offsets:
                final_offsets.append((min(all_kps_offset[i]), max(all_kps_offset[i])))
            if score_method:
                kps_set[kp.lower()] = all_kps_score[i]
                final_score.append(all_kps_score[i])

    if score_method and not return_offsets:
        assert len(final_kps) == len(
            final_score
        ), "len of kps and score calculated is not same"
        return final_kps, final_score
    if score_method and return_offsets:
        assert len(final_kps) == len(
            final_score
        ), "len of kps and score calculated is not same"
        return final_kps, final_offsets, final_score

    return final_kps, None
