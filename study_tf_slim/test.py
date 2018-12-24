# -*- coding: utf-8 -* -
from tensorflow.contrib.slim import nets

inception = nets.inception
model = inception.inception_v2(None)


