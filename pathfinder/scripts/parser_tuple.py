import argparse

# Funcion solo para transformar texto a un tupla
# Ej: "10,10,1" -> (10,10,1)

def parse_tuple(arg):
    try:
        # Split the input by commas, strip whitespace, and convert to integers
        return tuple(map(int, (item.strip() for item in arg.split(','))))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be in the form x,y,z,...")
