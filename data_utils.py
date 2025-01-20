import subprocess
import os
import h5py
import numpy as np
import pretty_midi
from mido import KeySignatureError
from midi2audio import FluidSynth
from matplotlib import pyplot as plt
import librosa
from multiprocessing import Process, Queue
from typing import Optional, List, Tuple, Callable
import time

DATA_BASE_PATH = 'data'
DATA_GZ_NAME = 'lmd_full.tar.gz'
DATA_GZ_PATH = os.path.join(DATA_BASE_PATH, DATA_GZ_NAME)
DATA_UNPACKED_PATH = os.path.join(DATA_BASE_PATH, 'lmd_full')
HDF_FILE_PATH = os.path.join(DATA_BASE_PATH, 'piano_rolls.h5')
SAMPLING_FREQ = 100

MAX_NUM_TIME_STEPS = 2_000
TARGET_INSTRUMENT = "Bass"
MAX_NUM_PROCESSED_FILES = 100
START_NOTE = 21
NUM_NOTES = 88


def download_dataset():
  if any([os.path.exists(path) for path in (DATA_GZ_PATH, DATA_UNPACKED_PATH, HDF_FILE_PATH)]):
    print('Dataset already downloaded.')
    return

  print('Downloading dataset...')
  os.makedirs(DATA_BASE_PATH, exist_ok=True)
  subprocess.run(['wget', 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz', '-O', DATA_GZ_PATH])
  print('Downloaded dataset.')


def unpack_dataset():
  if any([os.path.exists(path) for path in (DATA_UNPACKED_PATH, HDF_FILE_PATH)]):
    print('Dataset already unpacked.')
    return

  print('Unpacking dataset...')
  subprocess.run(['tar', 'xvzf', DATA_GZ_PATH, '-C', DATA_BASE_PATH])
  print('Unpacked dataset.')


def extract_piano_roll_from_instrument(instrument: pretty_midi.Instrument):
  """
  Extract the piano roll for the target instrument, binarize, and roll it.
  :param instrument:
  :return:
  """
  # Extract the piano roll for the target instrument
  piano_roll = instrument.get_piano_roll(fs=SAMPLING_FREQ)[START_NOTE:START_NOTE + NUM_NOTES]  # (NUM_NOTES, MAX_NUM_TIME_STEPS)
  if np.all(piano_roll == 0) or piano_roll.shape[1] < MAX_NUM_TIME_STEPS:
    # Skip empty piano rolls and those shorter than MAX_NUM_TIME_STEPS
    return None

  # roll array such that the instrument starts in the first time step
  first_non_zero_col = np.nonzero(piano_roll)[1][0]
  piano_roll = np.roll(piano_roll, -first_non_zero_col)
  # clip the time to time_steps
  piano_roll = piano_roll[:, :MAX_NUM_TIME_STEPS]
  piano_roll[piano_roll > 0] = 1  # binarize the piano roll
  return piano_roll


def extract_piano_roll_from_instrument_v2(instrument: pretty_midi.Instrument) -> Tuple[Optional[np.ndarray], int]:
  """
  Extract the piano roll for the target instrument, binarize, and roll it.
  :param instrument:
  :return: Tuple of piano roll and length
  """
  # Extract the piano roll for the target instrument
  piano_roll = instrument.get_piano_roll(fs=SAMPLING_FREQ)[START_NOTE:START_NOTE + NUM_NOTES]  # (NUM_NOTES, None)
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

  # return transposed piano roll and seq_len
  return piano_roll.T, seq_len


def build_hdf_file(hdf_file_path: str, directories: List[str], queue: Queue):
  """
  Build an HDF file containing the piano rolls for the target instrument. If the target instrument is not found
  in a MIDI file, the file is skipped.

  Piano rolls are binarized, i.e., the velocity is set to 1 if the note is played at a time step, and 0 otherwise.
  Piano rolls are also resized to have MAX_NUM_TIME_STEPS time steps and are rolled such that the first time step
  contains the first played note.
  :return:
  """

  if os.path.exists(hdf_file_path):
    print(f"File {hdf_file_path} already exists. Skipping...")
    return

  with h5py.File(hdf_file_path, "w") as hf:
    # Create an expandable dataset for piano rolls
    hf.create_dataset(
      "piano_rolls",
      shape=(0, NUM_NOTES, MAX_NUM_TIME_STEPS),
      maxshape=(None, NUM_NOTES, MAX_NUM_TIME_STEPS),  # Unlimited growth along the first dimension
      dtype=np.float32
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

          # only process midi files which include target instrument
          instruments = [instrument for instrument in pm.instruments if TARGET_INSTRUMENT in instrument.name]
          if len(instruments) > 0:
            instrument = instruments[0]
          else:
            continue

          # Extract the piano roll for the target instrument
          piano_roll = extract_piano_roll_from_instrument(instrument)
          if piano_roll is None:
            continue

          # Resize and append the piano roll to the dataset
          # Resize the dataset to accommodate the new data
          hf["piano_rolls"].resize((hf["piano_rolls"].shape[0] + 1, NUM_NOTES, MAX_NUM_TIME_STEPS))
          hf["piano_rolls"][-1] = piano_roll  # Add new piano roll

    # Verify the final dataset
    print(f"Final dataset shape for {hdf_file_path}:", hf["piano_rolls"].shape)


def build_hdf_file_v2(hdf_file_path: str, directories: List[str], queue: Queue):
  """
  Build an HDF file containing the piano rolls for the target instrument. If the target instrument is not found
  in a MIDI file, the file is skipped.

  Piano rolls are binarized, i.e., the velocity is set to 1 if the note is played at a time step, and 0 otherwise.
  Piano rolls are also resized to have MAX_NUM_TIME_STEPS time steps and are rolled such that the first time step
  contains the first played note.
  :return:
  """

  if os.path.exists(hdf_file_path):
    print(f"File {hdf_file_path} already exists. Skipping...")
    return

  with h5py.File(hdf_file_path, "w") as hf:
    # Create an expandable dataset for piano rolls
    # we store all frames of all seqs along the first dimension and use a separate dataset for lengths
    hf.create_dataset(
      "piano_rolls",
      shape=(0, NUM_NOTES),
      maxshape=(None, NUM_NOTES),  # Unlimited growth along the first dimension
      dtype=np.float32
    )
    hf.create_dataset(
      "seq_lens",
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

          # only process midi files which include target instrument
          instruments = [instrument for instrument in pm.instruments if TARGET_INSTRUMENT in instrument.name]
          if len(instruments) > 0:
            instrument = instruments[0]
          else:
            continue

          # Extract the piano roll for the target instrument
          piano_roll, seq_len = extract_piano_roll_from_instrument_v2(instrument)
          if piano_roll is None:
            continue

          # Resize and append the piano roll to the dataset
          # Resize the dataset to accommodate the new data
          hf["piano_rolls"].resize((hf["piano_rolls"].shape[0] + seq_len, NUM_NOTES))
          hf["piano_rolls"][-seq_len:] = piano_roll  # Add new piano roll
          # append the length
          hf["seq_lens"].resize((hf["seq_lens"].shape[0] + 1,))
          hf["seq_lens"][-1] = seq_len

    # Verify the final dataset
    print(f"Final dataset shape for {hdf_file_path}:", hf["piano_rolls"].shape)


def merge_hdf_files(output_file, hdf_files):
  """
  Merge multiple HDF5 files into a single file.
  """
  with h5py.File(output_file, "w") as hf_out:
    hf_out.create_dataset(
      "piano_rolls",
      shape=(0, NUM_NOTES, MAX_NUM_TIME_STEPS),
      maxshape=(None, NUM_NOTES, MAX_NUM_TIME_STEPS),
      dtype=np.float32
    )
    for hdf_file in hdf_files:
      with h5py.File(hdf_file, "r") as hf_in:
        piano_rolls = hf_in["piano_rolls"]
        hf_out["piano_rolls"].resize(
          (hf_out["piano_rolls"].shape[0] + piano_rolls.shape[0], NUM_NOTES, MAX_NUM_TIME_STEPS))
        hf_out["piano_rolls"][-piano_rolls.shape[0]:] = piano_rolls[:]
  print("Merged HDF files into:", output_file)


def merge_hdf_files_v2(output_file, hdf_files):
  """
  Merge multiple HDF5 files into a single file.
  """
  with h5py.File(output_file, "w") as hf_out:
    hf_out.create_dataset(
      "piano_rolls",
      shape=(0, NUM_NOTES),
      maxshape=(None, NUM_NOTES),
      dtype=np.float32
    )
    hf_out.create_dataset(
      "seq_lens",
      shape=(0,),
      maxshape=(None,),
      dtype=np.int32
    )

    for hdf_file in hdf_files:
      with h5py.File(hdf_file, "r") as hf_in:
        piano_rolls = hf_in["piano_rolls"]
        hf_out["piano_rolls"].resize(
          (hf_out["piano_rolls"].shape[0] + piano_rolls.shape[0], NUM_NOTES)
        )
        hf_out["piano_rolls"][-piano_rolls.shape[0]:] = piano_rolls[:]
        hf_out["seq_lens"].resize(
          (hf_out["seq_lens"].shape[0] + hf_in["seq_lens"].shape[0],)
        )
        hf_out["seq_lens"][-hf_in["seq_lens"].shape[0]:] = hf_in["seq_lens"][:]

  print("Merged HDF files into:", output_file)


def build_hdf_file_multi(
        num_cores: int,
        build_hdf_file_func: Callable = build_hdf_file_v2,
        merge_hdf_files_func: Callable = merge_hdf_files_v2
):
  if any([os.path.exists(path) for path in (HDF_FILE_PATH,)]):
    print(f"HDF dataset already exists.")
    return

  assert num_cores <= os.cpu_count(), "Number of cores exceeds the number of available cores."

  # Divide the subdirectories among processes
  subdirectories = [
    os.path.join(DATA_UNPACKED_PATH, d) for d in os.listdir(DATA_UNPACKED_PATH) if
    os.path.isdir(os.path.join(DATA_UNPACKED_PATH, d))
  ]
  subdirectories_split = np.array_split(subdirectories, num_cores)

  processes = []
  queue = Queue()
  temp_hdf_files = []

  # Create one process per core
  for i, subdirs in enumerate(subdirectories_split):
    temp_hdf_file = f"{DATA_BASE_PATH}/piano_rolls_{i}.h5"
    temp_hdf_files.append(temp_hdf_file)
    process = Process(target=build_hdf_file_func, args=(temp_hdf_file, subdirs, queue))
    processes.append(process)
    process.start()

  # Wait for all processes to complete
  for process in processes:
    process.join()

  # Merge all temporary HDF files
  merge_hdf_files_func(HDF_FILE_PATH, temp_hdf_files)

  # Cleanup temporary files
  for temp_file in temp_hdf_files:
    os.remove(temp_file)


def piano_roll_array_to_wav(piano_roll: np.ndarray, file_path: str, fs: FluidSynth):
  # Create a PrettyMIDI object
  pm = pretty_midi.PrettyMIDI()

  # Create an instrument (e.g., Piano, program number 0)
  instrument = pretty_midi.Instrument(program=0)

  # Iterate through the piano roll and add notes
  for note_number, row in enumerate(piano_roll):
    note_number += 21  # Adjust for MIDI pitch alignment (A0 starts at 21)
    nonzero_indices = np.where(row > 0)[0]  # Get time steps with non-zero velocity
    if len(nonzero_indices) == 0:
      continue
    start_idx = nonzero_indices[0]
    for idx in range(1, len(nonzero_indices)):
      if nonzero_indices[idx] != nonzero_indices[idx - 1] + 1:
        # Found the end of a note
        end_idx = nonzero_indices[idx - 1]
        start_time = start_idx * 1 / SAMPLING_FREQ  # Convert time step to seconds
        end_time = (end_idx + 1) * 1 / SAMPLING_FREQ
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
    start_time = start_idx * 0.01
    end_time = (end_idx + 1) * 0.01
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
  fs.midi_to_audio(f"{file_path}.mid", f'{file_path}.wav')


def visualize_piano_roll(piano_roll: np.ndarray, i=0):
  # Plot the piano roll
  plt.figure(figsize=(12, 6))
  librosa.display.specshow(
    piano_roll,
    y_axis="cqt_note",
    x_axis="time",
    cmap="Greys",
    sr=SAMPLING_FREQ,
  )
  plt.title(f"Piano Roll Visualization")
  plt.xlabel("Time (frames)")
  plt.ylabel("Note")
  plt.colorbar(label="Velocity")
  # plt.show()
  plt.savefig(f"piano_rolls_{i}")
