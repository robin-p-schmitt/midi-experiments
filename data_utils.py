import subprocess
import os
import h5py
import numpy as np
import pretty_midi
from mido import KeySignatureError
from midi2audio import FluidSynth
from matplotlib import pyplot as plt
import librosa
from multiprocessing import Process
from typing import Optional, List, Tuple, Callable
import time

DATA_BASE_PATH = 'data'

LAKH_DATA_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
LAKH_DATA_PACKED_NAME = 'lmd_full.tar.gz'
LAKH_DATA_PACKED_PATH = os.path.join(DATA_BASE_PATH, LAKH_DATA_PACKED_NAME)
LAKH_DATA_UNPACKED_PATH = os.path.join(DATA_BASE_PATH, 'lmd_full')
LAKH_HDF_FILE_PATH = os.path.join(DATA_BASE_PATH, 'lakh_played_notes.h5')
LAKH_MONOPHONIC_HDF_FILE_PATH = os.path.join(DATA_BASE_PATH, 'lakh_played_notes_monophonic.h5')

MAESTRO_DATA_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_DATA_PACKED_NAME = "maestro-v3.0.0-midi.zip"
MAESTRO_DATA_PACKED_PATH = os.path.join(DATA_BASE_PATH, MAESTRO_DATA_PACKED_NAME)
MAESTRO_DATA_UNPACKED_PATH = os.path.join(DATA_BASE_PATH, 'maestro-v3.0.0')
MAESTRO_HDF_FILE_PATH = os.path.join(DATA_BASE_PATH, 'maestro_played_notes.h5')
MAESTRO_MONOPHONIC_HDF_FILE_PATH = os.path.join(DATA_BASE_PATH, 'maestro_played_notes_monophonic.h5')

EXAMPLE_PIANO_ROLLS_PATH = os.path.join(DATA_BASE_PATH, 'example_piano_rolls')

VALID_DATASET_NAMES = ("LAKH", "MAESTRO")

FRAMES_PER_BAR = 16
NUM_BARS = 4
MAX_NUM_PROCESSED_FILES = 40_000
START_NOTE = 21
NUM_NOTES = 88
SILENT_IDX = NUM_NOTES
MAX_NUM_NOTES_PLAYED_AT_ONCE = 5


def download_dataset(data_url: str, data_gz_path: str):
  print('Downloading dataset...')
  os.makedirs(DATA_BASE_PATH, exist_ok=True)
  subprocess.run(['wget', data_url, '-O', data_gz_path])
  print('Downloaded dataset.')


def unpack_dataset(data_gz_path: str):
  print('Unpacking dataset...')
  if os.path.split(data_gz_path)[-1].endswith('.zip'):
    subprocess.run(['unzip', data_gz_path, '-d', DATA_BASE_PATH])
  else:
    assert os.path.split(data_gz_path)[-1].endswith('.tar.gz'), "Expected zip or tar.gz file."
    subprocess.run(['tar', 'xvzf', data_gz_path, '-C', DATA_BASE_PATH])
  print('Unpacked dataset.')


def extract_piano_roll_from_instrument(
        instrument: pretty_midi.Instrument,
        sampling_freq: int,
) -> Tuple[Optional[np.ndarray], int]:
  """
  Extract the piano roll for the target instrument, binarize, and roll it.
  :param instrument:
  :param sampling_freq:
  :return: Tuple of piano roll and length
  """
  # Extract the piano roll for the target instrument
  piano_roll = instrument.get_piano_roll(fs=sampling_freq)[START_NOTE:START_NOTE + NUM_NOTES]  # (NUM_NOTES, None)
  if np.all(piano_roll == 0):
    # Skip empty piano rolls
    return None, 0

  # roll array such that the instrument starts in the first time step
  first_non_zero_col = np.nonzero(piano_roll)[1][0]
  piano_roll = np.roll(piano_roll, -first_non_zero_col)
  # clip the time to the last non-zero column
  seq_len = int(np.nonzero(piano_roll)[1][-1]) + 1
  piano_roll = piano_roll[:, :seq_len]
  piano_roll[piano_roll > 0] = 1  # binarize the piano roll

  # convert piano roll of shape (NUM_NOTES, seq_len) to array with shape (seq_len, MAX_NUM_NOTES_PLAYED_AT_ONCE),
  # where the second dimension contains the indices of the played notes at each time step
  if np.all(np.sum(piano_roll, axis=0) > MAX_NUM_NOTES_PLAYED_AT_ONCE):
    # Skip piano rolls with more than MAX_NUM_NOTES_PLAYED_AT_ONCE notes played at once
    return None, 0

  # return transposed piano roll (seq_len, NUM_NOTES) and seq_len
  return piano_roll.T, seq_len


def compress_piano_roll(piano_roll: np.ndarray) -> np.ndarray:
  """

  :param piano_roll: np.array of shape (seq_len, NUM_NOTES)
  :return:
  """
  seq_len = piano_roll.shape[0]
  # identify indices of played notes
  played_indices = np.argwhere(piano_roll == 1)  # Find (time_frame, note_index) pairs
  # initialize the output array. set non-played notes to -1
  played_notes = np.full((seq_len, MAX_NUM_NOTES_PLAYED_AT_ONCE), -1, dtype=np.int8)

  # populate the played_notes array
  time_frames, note_indices = played_indices[:, 0], played_indices[:, 1]
  note_counts = np.bincount(time_frames, minlength=seq_len)  # Count notes per time frame
  # create a mask to ensure we only consider up to MAX_NUM_NOTES_PLAYED_AT_ONCE notes per time frame
  valid_mask = np.hstack([np.arange(c) < MAX_NUM_NOTES_PLAYED_AT_ONCE for c in note_counts])
  # assign notes to the correct positions
  played_notes[time_frames[valid_mask], np.arange(valid_mask.sum()) % MAX_NUM_NOTES_PLAYED_AT_ONCE] = note_indices[valid_mask]

  return played_notes


def decompress_played_notes(played_notes: np.ndarray) -> np.ndarray:
  """

  :param played_notes: np.array of shape (seq_len, MAX_NUM_NOTES_PLAYED_AT_ONCE)
  :return:
  """
  seq_len = played_notes.shape[0]
  # True for played notes, False for padding
  valid_mask = played_notes != -1  # [seq_len, MAX_NUM_NOTES_PLAYED_AT_ONCE]
  # flattened array of valid notes
  rows = played_notes[valid_mask]  # [num_notes_played]
  # flattened array of time steps -> repeat each time step MAX_NUM_NOTES_PLAYED_AT_ONCE times since we allow up to
  # MAX_NUM_NOTES_PLAYED_AT_ONCE notes per time step
  cols = np.repeat(np.arange(seq_len), MAX_NUM_NOTES_PLAYED_AT_ONCE)[valid_mask.flatten()]

  piano_roll = np.zeros((NUM_NOTES, seq_len))
  # set the played notes to 1
  piano_roll[rows, cols] = 1

  return piano_roll.T


def get_instrument_by_target_instrument(
        instruments: List[pretty_midi.Instrument],
        target_instrument: Optional[str]
) -> Optional[pretty_midi.Instrument]:
  assert target_instrument is not None, "Target instrument must be provided."
  # only process midi files which include target instrument
  instruments = [instrument for instrument in instruments if target_instrument in instrument.name]
  if len(instruments) > 0:
    return instruments[0]
  else:
    return None


def get_single_instrument(
        instruments: List[pretty_midi.Instrument],
        target_instrument: Optional[str] = None
) -> Optional[pretty_midi.Instrument]:
  # only process midi files which include target instrument
  if len(instruments) == 1:
    return instruments[0]
  else:
    return None


def build_hdf_file(
        hdf_file_path: str,
        directories: List[str],
        target_instrument: Optional[str],
        get_instrument_func: Callable
):
  """
  Build an HDF file containing the piano rolls for the target instrument. If the target instrument is not found
  in a MIDI file, the file is skipped.

  Piano rolls are binarized, i.e., the velocity is set to 1 if the note is played at a time step, and 0 otherwise.
  Piano rolls are rolled such that the first time step contains the first played note.

  :param hdf_file_path: Path to the HDF file
  :param directories: List of directories containing MIDI files
  :param target_instrument: Name of the target instrument
  :param get_instrument_func: Function to extract the target instrument from a list of instruments
  :return:
  """

  if os.path.exists(hdf_file_path):
    print(f"File {hdf_file_path} already exists. Skipping...")
    return

  with h5py.File(hdf_file_path, "w") as hf:
    # Create an expandable dataset for piano rolls
    # we store all frames of all seqs along the first dimension and use a separate dataset for lengths
    hf.create_dataset(
      "played_notes",
      shape=(0, MAX_NUM_NOTES_PLAYED_AT_ONCE),
      maxshape=(None, MAX_NUM_NOTES_PLAYED_AT_ONCE),  # Unlimited growth along the first dimension
      dtype=np.int8
    )
    hf.create_dataset(
      "seq_lens",
      shape=(0,),
      maxshape=(None,),
      dtype=np.int32
    )
    hf.create_dataset(
      "sampling_freq",
      shape=(0,),
      maxshape=(None,),
      dtype=np.int32
    )

    # Process MIDI files and append to the dataset
    num_processed_files = 0
    for directory in directories:
      for dirpath, dirnames, filenames in os.walk(directory):
        if num_processed_files >= MAX_NUM_PROCESSED_FILES:
          break
        for midi_filename in filenames:
          num_processed_files += 1
          midi_path = os.path.join(
            dirpath,
            # midi_filename can be bytes or str
            midi_filename if isinstance(midi_filename, str) else midi_filename.decode('utf-8')
          )

          if num_processed_files % 1000 == 0:
            print(f"Processed {num_processed_files} midi files for {hdf_file_path}")
          if num_processed_files >= MAX_NUM_PROCESSED_FILES:
            break

          # Load the MIDI file
          try:
            pm = pretty_midi.PrettyMIDI(midi_path)
          # some files are known to be corrupted, so we skip them
          except (OSError, EOFError, KeySignatureError, ValueError, KeyError, IndexError, ZeroDivisionError) as e:
            print(f"Error loading {midi_path} for {hdf_file_path}: {e}")
            continue

          instrument = get_instrument_func(pm.instruments, target_instrument)
          if instrument is None:
            continue

          # Extract the piano roll for the target instrument
          sampling_freq = int(1 / (pm.get_beats()[1] / 4))
          if sampling_freq > 10:
            # Skip files with high sampling rates
            continue

          piano_roll, seq_len = extract_piano_roll_from_instrument(instrument, sampling_freq)
          if piano_roll is None:
            continue

          # compress piano roll
          played_notes = compress_piano_roll(piano_roll)

          # sanity check to ensure the decompression works
          # reconstructed_piano_roll = decompress_played_notes(played_notes)
          # assert np.all(piano_roll == reconstructed_piano_roll), "Decompression failed"

          # Resize and append the piano roll to the dataset
          # Resize the dataset to accommodate the new data
          hf["played_notes"].resize((hf["played_notes"].shape[0] + seq_len, MAX_NUM_NOTES_PLAYED_AT_ONCE))
          hf["played_notes"][-seq_len:] = played_notes  # Add new played notes
          # append the length
          hf["seq_lens"].resize((hf["seq_lens"].shape[0] + 1,))
          hf["seq_lens"][-1] = seq_len
          # append the sampling rate
          hf["sampling_freq"].resize((hf["sampling_freq"].shape[0] + 1,))
          hf["sampling_freq"][-1] = sampling_freq

    # Verify the final dataset
    print(f"Final dataset shape for {hdf_file_path}:", hf["played_notes"].shape)


def merge_hdf_files(output_file, hdf_files):
  """
  Merge multiple HDF5 files into a single file.
  """
  with h5py.File(output_file, "w") as hf_out:
    hf_out.create_dataset(
      "played_notes",
      shape=(0, MAX_NUM_NOTES_PLAYED_AT_ONCE),
      maxshape=(None, MAX_NUM_NOTES_PLAYED_AT_ONCE),
      dtype=np.int8
    )
    hf_out.create_dataset(
      "seq_lens",
      shape=(0,),
      maxshape=(None,),
      dtype=np.int32
    )
    hf_out.create_dataset(
      "sampling_freq",
      shape=(0,),
      maxshape=(None,),
      dtype=np.int32
    )

    for hdf_file in hdf_files:
      with h5py.File(hdf_file, "r") as hf_in:
        played_notes = hf_in["played_notes"]
        hf_out["played_notes"].resize(
          (hf_out["played_notes"].shape[0] + played_notes.shape[0], MAX_NUM_NOTES_PLAYED_AT_ONCE)
        )
        hf_out["played_notes"][-played_notes.shape[0]:] = played_notes[:]
        hf_out["seq_lens"].resize(
          (hf_out["seq_lens"].shape[0] + hf_in["seq_lens"].shape[0],)
        )
        hf_out["seq_lens"][-hf_in["seq_lens"].shape[0]:] = hf_in["seq_lens"][:]
        hf_out["sampling_freq"].resize(
          (hf_out["sampling_freq"].shape[0] + hf_in["sampling_freq"].shape[0],)
        )
        hf_out["sampling_freq"][-hf_in["sampling_freq"].shape[0]:] = hf_in["sampling_freq"][:]

  print("Merged HDF files into:", output_file)


def build_hdf_file_multi(
        data_unpacked_path: str,
        hdf_file_path: str,
        target_instrument: str,
        get_instrument_func: Callable,
        num_cores: int = 1,
):
  assert num_cores <= os.cpu_count(), "Number of cores exceeds the number of available cores."

  # Divide the subdirectories among processes
  subdirectories = [
    os.path.join(data_unpacked_path, d) for d in os.listdir(data_unpacked_path) if
    os.path.isdir(os.path.join(data_unpacked_path, d))
  ]
  subdirectories_split = np.array_split(subdirectories, num_cores)

  processes = []
  temp_hdf_files = []

  # Create one process per core
  for i, subdirs in enumerate(subdirectories_split):
    temp_hdf_file = f"{DATA_BASE_PATH}/piano_rolls_{i}.h5"
    temp_hdf_files.append(temp_hdf_file)
    process = Process(target=build_hdf_file, args=(temp_hdf_file, subdirs, target_instrument, get_instrument_func))
    processes.append(process)
    process.start()

  # Wait for all processes to complete
  for process in processes:
    process.join()

  # Merge all temporary HDF files
  merge_hdf_files(hdf_file_path, temp_hdf_files)

  # Cleanup temporary files
  for temp_file in temp_hdf_files:
    os.remove(temp_file)


def convert_polyphonic_hdf_to_monophonic(hdf_file_path: str, output_file_path: str):
  with h5py.File(output_file_path, "w") as hf_out:
    hf_out.create_dataset(
      "played_notes",
      shape=(0,),
      maxshape=(None,),
      dtype=np.int8
    )
    hf_out.create_dataset(
      "seq_lens",
      shape=(0,),
      maxshape=(None,),
      dtype=np.int32
    )
    hf_out.create_dataset(
      "sampling_freq",
      shape=(0,),
      maxshape=(None,),
      dtype=np.int32
    )

    with h5py.File(hdf_file_path, "r") as hf_in:
      played_notes = hf_in["played_notes"][:]
      # Convert the polyphonic played notes to monophonic by taking the highest note (pitch) played at each time step
      monophonic_played_notes = played_notes.max(axis=1)
      # use special index in case that no note is played
      monophonic_played_notes[monophonic_played_notes == -1] = SILENT_IDX
      hf_out["played_notes"].resize((monophonic_played_notes.shape[0],))
      hf_out["played_notes"][:] = monophonic_played_notes

      hf_out["seq_lens"].resize((hf_in["seq_lens"].shape[0],))
      hf_out["seq_lens"][:] = hf_in["seq_lens"][:]
      hf_out["sampling_freq"].resize((hf_in["sampling_freq"].shape[0],))
      hf_out["sampling_freq"][:] = hf_in["sampling_freq"][:]


def save_example_piano_rolls(hdf_file_path: str, dataset_name: str, num_examples: int = 5, monophonic: bool = False):
  example_directory = os.path.join(EXAMPLE_PIANO_ROLLS_PATH, dataset_name.lower())
  if os.path.exists(example_directory):
    print(f"Example piano rolls already exist for {dataset_name}. Skipping...")
    return
  else:
    print(f"Saving example piano rolls for {dataset_name}...")
  os.makedirs(example_directory, exist_ok=True)

  with h5py.File(hdf_file_path, "r") as hf:
    played_notes = hf["played_notes"]
    seq_lens = hf["seq_lens"]
    sampling_freq = hf["sampling_freq"]

    for i in range(num_examples):
      seq_len_i = seq_lens[i]
      sampling_freq_i = sampling_freq[i]

      start_idx = sum(seq_lens[:i])
      played_notes_i = played_notes[start_idx:start_idx + seq_len_i]

      if monophonic:
        piano_roll_i = np.eye(NUM_NOTES + 1)[played_notes_i]
      else:
        piano_roll_i = decompress_played_notes(played_notes_i)
      visualize_piano_roll(piano_roll_i.T, sampling_freq_i, f"{example_directory}/piano_roll_{i}")
      piano_roll_array_to_wav(piano_roll_i.T[:, :200], f"{example_directory}/piano_roll_{i}", sampling_freq_i)


def prepare_data(dataset_name: str = "lakh", num_workers: int = 1):
  dataset_name = dataset_name.upper()
  assert dataset_name in VALID_DATASET_NAMES, "Invalid dataset name."

  data_url = eval(f"{dataset_name}_DATA_URL")
  data_gz_path = eval(f"{dataset_name}_DATA_PACKED_PATH")
  data_unpacked_path = eval(f"{dataset_name}_DATA_UNPACKED_PATH")
  hdf_file_path = eval(f"{dataset_name}_HDF_FILE_PATH")
  monophonic_hdf_file_path = eval(f"{dataset_name}_MONOPHONIC_HDF_FILE_PATH")

  if any([os.path.exists(path) for path in (data_gz_path, data_unpacked_path, hdf_file_path)]):
    print('Dataset already downloaded.')
  else:
    download_dataset(data_url=data_url, data_gz_path=data_gz_path)

  if any([os.path.exists(path) for path in (data_unpacked_path, hdf_file_path)]):
    print('Dataset already unpacked.')
  else:
    unpack_dataset(data_gz_path=data_gz_path)

  if any([os.path.exists(path) for path in (hdf_file_path,)]):
    print('HDF already generated.')
  else:
    if dataset_name == "LAKH":
      target_instrument = "Bass"
      get_instrument_func = get_instrument_by_target_instrument
    else:
      target_instrument = None
      get_instrument_func = get_single_instrument
    build_hdf_file_multi(
      data_unpacked_path=data_unpacked_path,
      hdf_file_path=hdf_file_path,
      num_cores=num_workers,
      target_instrument=target_instrument,
      get_instrument_func=get_instrument_func
    )

  if os.path.exists(monophonic_hdf_file_path):
    print('Monophonic HDF already generated.')
  else:
    convert_polyphonic_hdf_to_monophonic(hdf_file_path, monophonic_hdf_file_path)

  save_example_piano_rolls(hdf_file_path, dataset_name)
  save_example_piano_rolls(monophonic_hdf_file_path, f"{dataset_name}_monophonic", monophonic=True)


def piano_roll_array_to_wav(piano_roll: np.ndarray, file_path: str, sampling_freq: int):
  """

  :param piano_roll: numpy array with shape (NUM_NOTES, time_steps)
  :param file_path:
  :param sampling_freq:
  :return:
  """
  fs = FluidSynth(sample_rate=44_100, sound_font="/usr/share/sounds/sf2/TimGM6mb.sf2")

  # Create a PrettyMIDI object
  pm = pretty_midi.PrettyMIDI()

  # Create an instrument (e.g., Piano, program number 0)
  instrument = pretty_midi.Instrument(program=0)

  # Iterate through the piano roll and add notes
  for note_number, row in enumerate(piano_roll):
    note_number += START_NOTE  # Adjust for MIDI pitch alignment (A0 starts at 21)
    nonzero_indices = np.where(row > 0)[0]  # Get time steps with non-zero velocity
    if len(nonzero_indices) == 0:
      continue
    start_idx = nonzero_indices[0]
    for idx in range(1, len(nonzero_indices)):
      if nonzero_indices[idx] != nonzero_indices[idx - 1] + 1:
        # Found the end of a note
        end_idx = nonzero_indices[idx - 1]
        start_time = start_idx * 1 / sampling_freq  # Convert time step to seconds
        end_time = (end_idx + 1) * 1 / sampling_freq
        velocity = int(row[start_idx])  # Use velocity from the piano roll
        note = pretty_midi.Note(
          velocity=velocity,
          pitch=note_number,
          start=start_time,
          end=end_time,
        )
        instrument.notes.append(note)
        start_idx = nonzero_indices[idx]
    # Add the last note
    end_idx = nonzero_indices[-1]
    start_time = start_idx * 1 / sampling_freq
    end_time = (end_idx + 1) * 1 / sampling_freq
    velocity = int(row[start_idx])
    note = pretty_midi.Note(
      velocity=velocity,
      pitch=note_number,
      start=start_time,
      end=end_time,
    )
    instrument.notes.append(note)

  # Add the instrument to the PrettyMIDI object
  pm.instruments.append(instrument)

  # Write the MIDI file
  pm.write(f"{file_path}.mid")
  # skip the wav file for now because it is not playing any sound for some reason
  # TODO: fix this
  # fs.midi_to_audio(f"{file_path}.mid", f'{file_path}.wav')


def visualize_piano_roll(piano_roll: np.ndarray, sampling_freq: int, filename: Optional[str] = None):
  """

  :param piano_roll: numpy array with shape (NUM_NOTES, time_steps)
  :param sampling_freq:
  :param filename:
  :return:
  """
  if not filename:
    filename = "piano_rolls"

  # Plot the piano roll
  plt.figure(figsize=(12, 6))
  librosa.display.specshow(
    piano_roll,
    y_axis="cqt_note",
    x_axis="time",
    cmap="Greys",
    sr=sampling_freq,
    hop_length=1,
  )
  plt.title(f"Piano Roll Visualization")
  plt.xlabel("Time (seconds)")
  plt.ylabel("Note")
  plt.colorbar(label="Velocity")

  ax = plt.gca()
  ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(unit='s'))

  # plt.show()
  plt.savefig(filename)
