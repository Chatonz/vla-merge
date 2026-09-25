"""Deterministic sampling for the two calibration passes."""

FLOWS = [0, 5, 9]


def row_indices(nrows, cap, flow, request_slot, *, cameras=1, camera=0, last_only=False):
    """Split a request/module row quota over generation calls and camera views."""
    if nrows < 1 or cap < 1 or flow not in FLOWS or (not 0 <= camera < cameras):
        raise ValueError("Invalid row-selection arguments")
    quota = min(cap, nrows * cameras)
    tokens = [0] if quota == 1 else [(nrows * cameras - 1) * i // (quota - 1) for i in range(quota)]
    result = []
    for i, token in enumerate(tokens):
        stage = 9 if last_only else FLOWS[(i + request_slot) % 3]
        if stage == flow and token // nrows == camera:
            result.append(token % nrows)
    return result
