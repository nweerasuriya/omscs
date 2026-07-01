class Token:
    """
    A general classification for a player's token for participation in a game that designates
    the agents placeholder on the playing field.
    """
    def __init__(self, value: str):
        self._value = value.upper()

    def value(self) -> str:
        """
        Returns the designation for this token.
        :return: string value of this token
        """
        return self._value

    def __eq__(self, other):
        return self.value() == other.value()

    def __hash__(self):
        return hash(self.value())
