"""Lastde for very short texts. With at least 8 shifted tokens the authors' get_lastde is used unchanged.
Below that, the multiscale window is capped at tau_prime = min(5, T - 3) (embed size 3, epsilon 10T and the
formula unchanged), since a coarse sequence needs embed_size + 1 entries to form an adjacent-orbit cosine.
In CATS this affects one very short test text."""
def lastde_score(native, fast_mde, logits, labels):
    length = logits.shape[1]
    if length >= 8:
        return native(logits, labels)
    embed_size = 3
    # a coarse sequence needs at least embed_size+1 entries to form an adjacent-orbit cosine
    tau_prime = min(5, length - embed_size)
    if tau_prime < 2:
        raise ValueError("Lastde requires at least two feasible scales")
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    likelihood = logits.log_softmax(-1).gather(-1, labels)
    denominator = fast_mde.get_tau_multiscale_DE(
        ori_data=likelihood, embed_size=embed_size, epsilon=10 * length,
        tau_prime=tau_prime)
    return (likelihood.mean(dim=1) / denominator).cpu().item()
