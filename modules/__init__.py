import os
__all__ = []

for file in os.listdir("./modules"):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:-3]
        __all__.append(module)
