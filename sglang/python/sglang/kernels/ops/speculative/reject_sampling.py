import triton
import triton.language as tl


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        offset_draft = (
            (pid * stride_dp_b)
            + (cur_prob_row * stride_dp_s)
            + (draft_token * stride_dp_v)
        )

        p = tl.load(TargetProbs + offset_prob)
        q = tl.load(DraftProbs + offset_draft)

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    # Pass 1: Sum
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_ptr = tp_base_ptr + v_offsets * stride_tp_v
        p_val = tl.load(p_ptr, mask=mask, other=0.0)

        if all_drafts_accepted:
            val = p_val
        else:
            q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
            q_val = tl.load(q_ptr, mask=mask, other=0.0)
            # Treat NaN q (degenerate draft rows) as 0: residual falls back to p.
            q_val = tl.where(q_val == q_val, q_val, 0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)

        norm_sum += tl.sum(val)

    # Pass 2: CDF. Degenerate residual (norm_sum == 0, i.e. p == q everywhere on
    # rejection) leaves the cumsum at 0 <= target_u, so final_token falls back to
    # VOCAB_SIZE - 1; acceptable since this case is numerically near-impossible.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Same NaN-q guard as pass 1.
                q_val = tl.where(q_val == q_val, q_val, 0.0)
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]

    grid = (batch_size,)
    speculative_sampling_classic_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
    )


@triton.jit
def speculative_sampling_from_logits_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetLogits,
    DraftLogits,
    TargetLogNorm,
    DraftLogNorm,
    Temperatures,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tl_b,
    stride_tl_s,
    stride_tl_v,
    stride_dl_b,
    stride_dl_s,
    stride_dl_v,
    stride_tn_b,
    stride_tn_s,
    stride_dn_b,
    stride_dn_s,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Exact chain rejection while reconstructing p/q from logits on demand."""
    pid = tl.program_id(0)
    temperature = tl.load(Temperatures + pid).to(tl.float32)
    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base)
    tl.store(AcceptIndex + pid * stride_idx_b, root_global_idx)
    last_accepted_global_idx = root_global_idx
    num_accept = 0
    cur_prob_row = 0
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
        target_logit = tl.load(
            TargetLogits
            + pid * stride_tl_b
            + cur_prob_row * stride_tl_s
            + draft_token * stride_tl_v
        ).to(tl.float32)
        draft_logit = tl.load(
            DraftLogits
            + pid * stride_dl_b
            + cur_prob_row * stride_dl_s
            + draft_token * stride_dl_v
        ).to(tl.float32)
        target_norm = tl.load(
            TargetLogNorm + pid * stride_tn_b + cur_prob_row * stride_tn_s
        )
        draft_norm = tl.load(
            DraftLogNorm + pid * stride_dn_b + cur_prob_row * stride_dn_s
        )
        p = tl.exp(target_logit / temperature - target_norm)
        q = tl.exp(draft_logit / temperature - draft_norm)
        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)
            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx
            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    target_norm = tl.load(
        TargetLogNorm + pid * stride_tn_b + cur_prob_row * stride_tn_s
    )
    safe_draft_row = tl.minimum(cur_prob_row, NUM_SLOTS - 2)
    draft_norm = tl.load(
        DraftLogNorm + pid * stride_dn_b + safe_draft_row * stride_dn_s
    )
    norm_sum = 0.0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE
        target_values = tl.load(
            TargetLogits
            + pid * stride_tl_b
            + cur_prob_row * stride_tl_s
            + v_offsets * stride_tl_v,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        p = tl.exp(target_values / temperature - target_norm)
        if all_drafts_accepted:
            values = p
        else:
            draft_values = tl.load(
                DraftLogits
                + pid * stride_dl_b
                + safe_draft_row * stride_dl_s
                + v_offsets * stride_dl_v,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            q = tl.exp(draft_values / temperature - draft_norm)
            values = tl.maximum(p - q, 0.0)
        norm_sum += tl.sum(tl.where(mask, values, 0.0))

    target_u = coin_final * norm_sum
    cumulative = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE
            target_values = tl.load(
                TargetLogits
                + pid * stride_tl_b
                + cur_prob_row * stride_tl_s
                + v_offsets * stride_tl_v,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            p = tl.exp(target_values / temperature - target_norm)
            if all_drafts_accepted:
                values = p
            else:
                draft_values = tl.load(
                    DraftLogits
                    + pid * stride_dl_b
                    + safe_draft_row * stride_dl_s
                    + v_offsets * stride_dl_v,
                    mask=mask,
                    other=-float("inf"),
                ).to(tl.float32)
                q = tl.exp(draft_values / temperature - draft_norm)
                values = tl.maximum(p - q, 0.0)
            block_prefix = tl.cumsum(tl.where(mask, values, 0.0), axis=0)
            total_prefix = cumulative + block_prefix
            candidates_mask = total_prefix > target_u
            has_match = tl.max(candidates_mask, axis=0)
            if has_match:
                final_token = v_start + tl.argmax(
                    candidates_mask.to(tl.int32), axis=0
                )
                found = 1
            cumulative += tl.sum(tl.where(mask, values, 0.0))

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_from_logits_triton(
    *,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_logits,
    draft_logits,
    target_log_norm,
    draft_log_norm,
    temperatures,
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_logits.shape[-1]
    speculative_sampling_from_logits_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_logits,
        draft_logits,
        target_log_norm,
        draft_log_norm,
        temperatures,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_logits.stride(0),
        target_logits.stride(1),
        target_logits.stride(2),
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_logits.stride(2),
        target_log_norm.stride(0),
        target_log_norm.stride(1),
        draft_log_norm.stride(0),
        draft_log_norm.stride(1),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        num_warps=8,
    )
