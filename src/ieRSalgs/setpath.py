import os

def setpath(subdirectory = './data/'):
    working_path = os.path.join(os.path.dirname(__file__), subdirectory)
    return working_path
  