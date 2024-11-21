from importlib.metadata import metadata

# Weights zip urls
ZIP_URLS = dict([meta.split(', ') for meta in metadata('totalspineseg').get_all('Project-URL') if meta.startswith('Dataset')])
