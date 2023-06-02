import torch
import random
import pickle
import pandas as pd
import os
import numpy as np
import datasets
import transformers
from transformers import pipeline, set_seed, get_scheduler, GPTNeoForCausalLM, AutoModelForCausalLM, GPT2TokenizerFast, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import Subset, IterableDataset, DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split, KFold
from random import sample
from torch.optim import AdamW
#don't load datasets.IterableDataset, not sent to cuda
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets
from argparse import Namespace
import json
import requests
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as F
import shutil
import re
import json
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb
from accelerate import Accelerator
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from argparse import Namespace
import argparse
import csv
import itertools
import subprocess
#from torch.nn.utils import clip_grad_norm_

if(torch.cuda.is_available()):
    print("cuda available")
    device = 'cuda'
else:
    print("cuda not available")
    device = 'cpu'
