# helpers.py
#
# 15 November 2023

def translate_dict(dico):
    """Helper function to load model weights (for visualizing in notebooks)

    Simply removes the "module." at the beginning of all the keys"""
    newdico = {}
    length = len('module.')
    for k,v in dico.items():
        k_new = k[length:]
        newdico[k_new] = v
    return newdico
