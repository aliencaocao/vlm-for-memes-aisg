def embed_ipython(default_rank: int | None=None):
    # Usage: for f in embed_ipython(0): f()
    from IPython import embed
    from torch import distributed as dist
    if default_rank is not None:
        if dist.get_rank() == default_rank:
            yield embed
        dist.barrier()

    def select_rank(ws = dist.get_world_size()):
        p = f'(distdbg) select rank [0-{ws-1}] (or empty to exit): '
        try:
            s = [None if dist.get_rank() else input(p)]
        except EOFError:
            return None
        dist.broadcast_object_list(s)
        return int(s[0]) if s[0].isdigit() else None

    while s := select_rank():
        if dist.get_rank() == int(s):
            yield embed
        dist.barrier()
