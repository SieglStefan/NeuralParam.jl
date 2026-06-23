### General utility functions
###
### Helper functions used across



# Extract vertical layer k from a series of fields
extract_layer(layer, f) = [i[:, layer] for i in f]
