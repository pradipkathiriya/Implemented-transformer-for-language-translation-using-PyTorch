import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizer import Tokenizer
from tokenizer.models import WordLevel
from tokenizer.pre_tokenizer import Whitespace


