import math


def print_table(header: str, lines: list[str], separator: str = "#") -> None:
    """
        Prints a prettified table with the given header and lines.
    """
    length = math.ceil((max([len(line) for line in lines]+[len(header)])-len(header))/2)*2+len(header)+2
    print(int((length-len(header))/2)*separator+header+int((length-len(header))/2)*separator)
    for line in lines:
        if line == "":
            print(length*separator)
        else:
            print(separator+line+(length-len(line)-2)*" "+separator)