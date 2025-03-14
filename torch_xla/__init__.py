import logging
import os
import re
import tempfile

import torch
import _XLAC
from ._internal import tpu

logging.basicConfig()
logger = logging.getLogger(__name__)


def _set_missing_flags(flags, sets):
  for name, defval in sets:
    insert = True
    for fval in flags:
      m = re.match(r'(--)?([^=]+)', fval)
      if m and m.group(2) == name:
        insert = False
        break
    if insert:
      flags.append('--{}={}'.format(name, defval))
  return flags


def _setup_xla_flags():
  flags = os.environ.get('XLA_FLAGS', '').split(' ')
  flags = _set_missing_flags(flags, (('xla_cpu_enable_fast_math', 'false'),))
  flags = _set_missing_flags(
      flags, (('xla_gpu_simplify_all_fp_conversions', 'false'),))
  flags = _set_missing_flags(flags,
                             (('xla_gpu_force_compilation_parallelism', '8'),))
  os.environ['XLA_FLAGS'] = ' '.join(flags)


def _setup_libtpu_flags():
  flags = os.environ.get('LIBTPU_INIT_ARGS', '').split(' ')
  # This flag will rerun the latency hidding scheduler if the default
  # shared memory limit 95% leads to OOM. Each rerun will choose a value
  # 0.9x of the previous run, and the number of rerun is set to 1 now.
  # Shared memory limit refers to --xla_tpu_scheduler_percent_shared_memory_limit.
  # Lower shared memory limit means less communiation and computation overlapping,
  # and thus worse performance.
  flags = _set_missing_flags(flags,
                             (('xla_latency_hiding_scheduler_rerun', '1'),))

  # This flag will prevent AllGather decomposition into AllReduce by the
  # compiler when async AllGather is enabled. Decomposed AllGathers are
  # persisted in-memory and shared between the forward and backward passes,
  # which can result in the entire model's parameters being in device memory.
  # However, regular AllGathers are instead rematerialized in the backward pass,
  # and when they are async this incurs little overhead but significantly
  # improves device memory usage.
  flags = _set_missing_flags(
      flags, (('xla_tpu_prefer_async_allgather_to_allreduce', 'true'),))

  if tpu.version() == 5:
    default_v5_flags = {
        # TODO(jonbolin): Tune these flags for async collective fusion - v5
        # requires continuation fusion to run async collectives.
        'xla_enable_async_all_gather': 'true',
        'xla_enable_async_collective_permute': 'true',
    }
    flags = _set_missing_flags(flags, default_v5_flags.items())

  os.environ['LIBTPU_INIT_ARGS'] = ' '.join(flags)


def _setup_default_env():
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')

  if tpu.num_available_chips() > 0:
    _setup_libtpu_flags()

    os.environ.setdefault('ALLOW_MULTIPLE_LIBTPU_LOAD', '1')
    os.environ.setdefault('TPU_ML_PLATFORM', 'PyTorch/XLA')

    if tpu.version() == 4:
      os.environ.setdefault('TPU_MEGACORE', 'megacore_dense')


_fd, _tmp_fname = -1, ''


def _setup_debug_env():
  fd, tmp_fname = tempfile.mkstemp('.ptxla', text=True)
  os.environ.setdefault('XLA_FNTRACKER_FILE', tmp_fname)
  return fd, tmp_fname


def _summarize_fn_tracker():
  if not _tmp_fname:
    return
  from .debug.frame_parser_util import process_frames
  process_frames(_tmp_fname)
  os.close(_fd)
  os.remove(_tmp_fname)


def _aws_ec2_inf_trn_init():
  try:
    from torch_neuronx import xla
  except ImportError:
    return
  else:
    xla.init()


def _setup_tpu_vm_library_path() -> bool:
  """Returns true if $TPU_LIBRARY_PATH is set or can be inferred.

  We load libtpu.so in the following order of precedence:

  1. User-set $TPU_LIBRARY_PATH
  2. libtpu.so included in torch_xla/lib
  3. libtpu-nightly pip package

  Sets $PTXLA_TPU_LIBRARY_PATH if path is inferred by us to prevent conflicts
  with other frameworks. This env var will be removed in a future version.
  """
  if 'TPU_LIBRARY_PATH' in os.environ:
    return True

  module_path = os.path.dirname(__file__)
  bundled_libtpu_path = os.path.join(module_path, 'lib/libtpu.so')
  if os.path.isfile(bundled_libtpu_path) and not os.getenv('TPU_LIBRARY_PATH'):
    logger.info('Using bundled libtpu.so (%s)', bundled_libtpu_path)
    os.environ['PTXLA_TPU_LIBRARY_PATH'] = bundled_libtpu_path
    return True

  try:
    import libtpu
    os.environ['PTXLA_TPU_LIBRARY_PATH'] = libtpu.get_library_path()
    return True
  except ImportError:
    return False


# These needs to be called before the _XLAC module is loaded.
_setup_default_env()
_setup_xla_flags()
if int(os.environ.get('PT_XLA_DEBUG', '0')):
  _fd, _tmp_fname = _setup_debug_env()

if os.environ.get('TF_CPP_MIN_LOG_LEVEL') == '0':
  logger.setLevel(logging.INFO)

import atexit
from ._patched_functions import _apply_patches
from .version import __version__

_found_libtpu = _setup_tpu_vm_library_path()

# Setup Neuron library for AWS EC2 inf/trn instances.
_aws_ec2_inf_trn_init()


def _prepare_to_exit():
  try:
    _XLAC._prepare_to_exit()
    if int(os.environ.get('PT_XLA_DEBUG', '0')):
      _summarize_fn_tracker()
  except Exception as e:
    logging.error(
        "Caught an exception when exiting the process. Exception: ", exc_info=e)
    # Due to https://bugs.python.org/issue27035, simply raising an exception in the atexit callback does not set the exit code correctly. That is why we need to set the exit code explicitly.
    # Using `exit(1)` does not set a correct exit code because it is useful for the interactive interpreter shell and should not be used in programs and it works by raising an exception. (https://docs.python.org/3/library/constants.html#exit)
    # sys.exit(1) does not set a correct exit code because it also raises an exception.
    os._exit(1)


def _init_xla_lazy_backend():
  _XLAC._init_xla_lazy_backend()


atexit.register(_prepare_to_exit)
_apply_patches()
_init_xla_lazy_backend()

# This is to temporarily disable the automtic dynamic shape in PyTorch Dynamo,
# which was enabled by https://github.com/pytorch/pytorch/pull/103623.
# While we come up with a long term fix, we'll set this flag to False to
# keep PyTorch/XLA CI healthy.
# TODO @wonjoo come up with a long term fix in Dynamo.
torch._dynamo.config.automatic_dynamic_shapes = False

from .stablehlo import save_as_stablehlo, save_torch_model_as_stablehlo

from .experimental import plugins

if os.getenv('XLA_REGISTER_INSTALLED_PLUGINS') == '1':
  plugins.use_dynamic_plugins()
  plugins.register_installed_plugins()
