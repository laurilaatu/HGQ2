def str_float_int_or_none(s: str):
    if s == 'None':
        return None
    if s == 'True':
        return True
    if s == 'False':
        return False
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    if s.isdigit():
        return int(s)
    try:
        return float(s)
    except ValueError:
        pass
    raise ValueError(f'Invalid value: {s}')


def parse_string_to_args(string: str):
    chunks = string.split(',')
    args, kwargs = [], {}
    _pos_arg_ended = False
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk == '':
            raise SyntaxError(f'Invalid syntax: {string}')
        if '=' in chunk:
            _pos_arg_ended = True
            k, v = chunk.split('=')
            k, v = k.strip(), v.strip()
            if k == '' or v == '':
                raise SyntaxError(f'Invalid syntax: {string}')
            kwargs[k] = v
        else:
            assert not _pos_arg_ended, 'Positional argument after keyword argument'
            args.append(chunk)

    args = [str_float_int_or_none(a) for a in args]
    kwargs = {k: str_float_int_or_none(v) for k, v in kwargs.items()}
    return args, kwargs
