import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import threading
import queue
import uuid
import datetime
import os
import torch
from collections import deque, Counter
from facenet_pytorch import InceptionResnetV1
from geometry_utils import normalize_geometry

from gtts import gTTS
import pygame
