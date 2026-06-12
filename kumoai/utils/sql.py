def quote_ident(ident: str, char: str = '"') -> str:
    r"""Quotes a SQL identifier."""
    return char + ident.replace(char, char + char) + char
