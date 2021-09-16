from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "Baseline"

_C.MODEL.FREEZE_LAYERS = ['']

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.DEPTH = "50x"
_C.MODEL.BACKBONE.LAST_STRIDE = 1
# Backbone feature dimension
_C.MODEL.BACKBONE.FEAT_DIM = 2048
# Normalization method for the convolution layers.
_C.MODEL.BACKBONE.NORM = "BN"
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = False
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = False
# If use Non-local block in backbone
_C.MODEL.BACKBONE.WITH_NL = False
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = True
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''
_C.MODEL.BACKBONE.REMOVE = (3, 4)
# ---------------------------------------------------------------------------- #
# Transformer options
# ---------------------------------------------------------------------------- #
_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.NAME = "TransformerHead"
_C.MODEL.TRANSFORMER.FEATURE_DIM = 2048 # input feature dim
_C.MODEL.TRANSFORMER.EMBED_DIM = 768
_C.MODEL.TRANSFORMER.POS_DROP = 0.0 # 0
_C.MODEL.TRANSFORMER.PATH_DROP = 0.1 # 0.1
_C.MODEL.TRANSFORMER.DEPTH = 12
_C.MODEL.TRANSFORMER.NUM_HEADS = 3
_C.MODEL.TRANSFORMER.ENABLE_POS = False

_C.MODEL.TRANSFORMER.PART_STR = ("4x1",)
_C.MODEL.TRANSFORMER.PART_H = 4
_C.MODEL.TRANSFORMER.PART_V = 1
_C.MODEL.TRANSFORMER.INPUT_SIZE = (512,64,32)

_C.MODEL.TRANSFORMER_S = CN()
_C.MODEL.TRANSFORMER_S.NAME = "SpatialTransformerHead"
_C.MODEL.TRANSFORMER_S.FEATURE_DIM = 2048 # input feature dim
_C.MODEL.TRANSFORMER_S.EMBED_DIM = 768
_C.MODEL.TRANSFORMER_S.POS_DROP = 0.0 # 0
_C.MODEL.TRANSFORMER_S.PATH_DROP = 0.1 # 0.1
_C.MODEL.TRANSFORMER_S.DEPTH = 12
_C.MODEL.TRANSFORMER_S.NUM_HEADS = 3
_C.MODEL.TRANSFORMER_S.PART_STR = "8x4"
_C.MODEL.TRANSFORMER_S.INPUT_SIZE = (512,64,32)
_C.MODEL.TRANSFORMER_S.ENABLE_POS = False
_C.MODEL.TRANSFORMER_S.GAP = False
_C.MODEL.TRANSFORMER_S.ATTN_DROP = 0.0
_C.MODEL.TRANSFORMER_S.RET_ALL = False
_C.MODEL.TRANSFORMER_S.LOCAL_PART = 4
_C.MODEL.TRANSFORMER_S.CE_SCALE = 1.0
_C.MODEL.TRANSFORMER_T = CN()
_C.MODEL.TRANSFORMER_T.NAME = "TemporalTransformerHead"
_C.MODEL.TRANSFORMER_T.EMBED_DIM = 768
_C.MODEL.TRANSFORMER_T.POS_DROP = 0.0 # 0
_C.MODEL.TRANSFORMER_T.PATH_DROP = 0.1 # 0.1
_C.MODEL.TRANSFORMER_T.DEPTH = 12
_C.MODEL.TRANSFORMER_T.NUM_HEADS = 3
_C.MODEL.TRANSFORMER_T.ENABLE_POS = False
_C.MODEL.TRANSFORMER_T.GAP = False
_C.MODEL.TRANSFORMER_T.ATTN_DROP = 0.0
_C.MODEL.TRANSFORMER_T.RET_ALL = False
_C.MODEL.TRANSFORMER_T.RET_ATTN = False
# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.HASH_LAYER_DIMS = []
_C.MODEL.HEADS.HASH_LAYER_NUM = 1
_C.MODEL.HEADS.NAME = "EmbeddingHead"
# Normalization method for the convolution layers.
_C.MODEL.HEADS.NORM = "BN"
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = 0
# Embedding dimension in head
_C.MODEL.HEADS.EMBEDDING_DIM = 0
_C.MODEL.HEADS.TEMP = 1.0
# If use BNneck in embedding
_C.MODEL.HEADS.WITH_BNNECK = True
# Triplet feature using feature before(after) bnneck
_C.MODEL.HEADS.NECK_FEAT = "before"  # options: before, after
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "avgpool"

# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "linear"  # "arcSoftmax" or "circleSoftmax"

# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.15
_C.MODEL.HEADS.SCALE = 128

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = '("CrossEntropyLoss",)'

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_C.MODEL.LOSSES.CE.EPSILON = 0.0
_C.MODEL.LOSSES.CE.ALPHA = 0.2
_C.MODEL.LOSSES.CE.SCALE = 1.0

# Focal Loss options
_C.MODEL.LOSSES.FL = CN()
_C.MODEL.LOSSES.FL.ALPHA = 0.25
_C.MODEL.LOSSES.FL.GAMMA = 2
_C.MODEL.LOSSES.FL.SCALE = 1.0

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI.MARGIN = 0.3
_C.MODEL.LOSSES.TRI.NORM_FEAT = False
_C.MODEL.LOSSES.TRI.HARD_MINING = True
_C.MODEL.LOSSES.TRI.SCALE = 1.0

# Circle Loss options
_C.MODEL.LOSSES.CIRCLE = CN()
_C.MODEL.LOSSES.CIRCLE.MARGIN = 0.25
_C.MODEL.LOSSES.CIRCLE.GAMMA = 128
_C.MODEL.LOSSES.CIRCLE.SCALE = 1.0

# Cosface Loss options
_C.MODEL.LOSSES.COSFACE = CN()
_C.MODEL.LOSSES.COSFACE.MARGIN = 0.25
_C.MODEL.LOSSES.COSFACE.GAMMA = 128
_C.MODEL.LOSSES.COSFACE.SCALE = 1.0

# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5

# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10

# Random color jitter
_C.INPUT.CJ = CN()
_C.INPUT.CJ.ENABLED = False
_C.INPUT.CJ.PROB = 0.5
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1

# Auto augmentation
_C.INPUT.DO_AUTOAUG = False
_C.INPUT.AUTOAUG_PROB = 0.0

_C.INPUT.DO_RANDAUG = False
_C.INPUT.RANDAUG_PROB = 0.5
_C.INPUT.RANDAUG_STR = "rand-m9-mstd0.5"

# Augmix augmentation
_C.INPUT.DO_AUGMIX = False
_C.INPUT.AUGMIX_PROB = 0.0

# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.VALUE = [0.596*255, 0.558*255, 0.497*255]
# Random Patch
_C.INPUT.RPT = CN()
_C.INPUT.RPT.ENABLED = False
_C.INPUT.RPT.PROB = 0.5
_C.INPUT.MIX_TRACKLET = False
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("Market1501",)
# List of the dataset names for testing
_C.DATASETS.TESTS = ("Market1501",)
# Combine trainset and testset joint training
_C.DATASETS.COMBINEALL = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# P/K Sampler for data loading
_C.DATALOADER.PK_SAMPLER = True
# Naive sampler which don't consider balanced identity sampling
_C.DATALOADER.NAIVE_WAY = True
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.SEQ_LEN = 8
_C.DATALOADER.TEST_SEQ_LEN = 0
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.PERSON_NUMBER = 2900
_C.DATALOADER.PERSON_NUMBER_TEST = 3060

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# AUTOMATIC MIXED PRECISION
_C.SOLVER.FP16_ENABLED = False

# Optimizer
_C.SOLVER.OPT = "Adam"

_C.SOLVER.MAX_EPOCH = 120

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1.
_C.SOLVER.HEADS_LR_FACTOR = 1.
_C.SOLVER.TRANS_LR_FACTOR = 1.

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_NORM=0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.
_C.SOLVER.WEIGHT_DECAY_TRANS = 0.0005

# Multi-step learning rate options
_C.SOLVER.SCHED = "MultiStepLR"

_C.SOLVER.DELAY_EPOCHS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [30, 55]

# Cosine annealing learning rate options
_C.SOLVER.ETA_MIN_LR = 1e-7

# Warmup options
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

# Backbone freeze iters
_C.SOLVER.FREEZE_ITERS = 0

# FC freeze iters
_C.SOLVER.FREEZE_FC_ITERS = 0


# SWA options
# _C.SOLVER.SWA = CN()
# _C.SOLVER.SWA.ENABLED = False
# _C.SOLVER.SWA.ITER = 10
# _C.SOLVER.SWA.PERIOD = 2
# _C.SOLVER.SWA.LR_FACTOR = 10.
# _C.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
# _C.SOLVER.SWA.LR_SCHED = False

_C.SOLVER.CHECKPOINT_PERIOD = 20

# Number of images per batch across all machines.
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 5.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.NEIGHBOR_RANGE = 100
_C.TEST.ERASE_NUM = 64
_C.TEST.SEQ_TEST = False
_C.TEST.TIMES = 1
_C.TEST.HASH_RATE = 0.3
_C.TEST.EVAL_PERIOD = 20
_C.TEST.VERIF = False
# Number of images per batch in one process.
_C.TEST.IMS_PER_BATCH = 64
_C.TEST.METRIC = "cosine"
_C.TEST.ROC_ENABLED = False
_C.TEST.FLIP_ENABLED = False

# Average query expansion
_C.TEST.AQE = CN()
_C.TEST.AQE.ENABLED = False
_C.TEST.AQE.ALPHA = 3.0
_C.TEST.AQE.QE_TIME = 1
_C.TEST.AQE.QE_K = 5

# Re-rank
_C.TEST.RERANK = CN()
_C.TEST.RERANK.ENABLED = False
_C.TEST.RERANK.K1 = 20
_C.TEST.RERANK.K2 = 6
_C.TEST.RERANK.LAMBDA = 0.3

# Precise batchnorm
_C.TEST.PRECISE_BN = CN()
_C.TEST.PRECISE_BN.ENABLED = False
_C.TEST.PRECISE_BN.DATASET = 'Market1501'
_C.TEST.PRECISE_BN.NUM_ITER = 300

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
