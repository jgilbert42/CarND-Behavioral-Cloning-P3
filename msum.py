import argparse
import pprint as pp
from keras.models import load_model

parser = argparse.ArgumentParser(description='Print Model Summary')
parser.add_argument(
    'model',
    type=str,
    help='Path to model h5 file. Model should be on the same path.'
)
args = parser.parse_args()

model = load_model(args.model)
model.summary()

pp.pprint(model.get_config())

