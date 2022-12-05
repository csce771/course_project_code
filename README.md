# Application of Artificial Neural Networks (ANN) to Automatic Speech Recognition (ASR) on a Novel Dataset created using YouTube
## Usage
### Download Dataset
*We provide an example dataset from a numberphile YouTube video that is small; however we describe how to use YouTube videos of your own choosing.*
1. Download the YouTube video of your choosing using the "Download" button below the YouTube video (requires YouTube subscription).
2. Extract the audio from the YouTube video in wav format using the popular ffmpeg command-line tool.
3. Name the audio "audio.wav".
4. Download the companion script from the YouTube video by clicking the 3 dots under the video player, highlighting the entire transcript, and pasting into a file named "script.txt".
5. Put "audio.wav" and "script.txt" in a subdirectory of this one named of your choosing.
6. Open "main.ipynb" and replace "video_name" with the name of your subdirectory.
7. Run both cells in order.
8. You should have "metadata.csv" and "phrases/*.wav" in your subdirectory.
9. You are ready to train and evaluate a model.
### Train and Evaluate a Model
1. In "main.py" adjust "metadata_path" and "wavs_path" according to the name of your subdirectory containing your phrases.
2. Run main.py.
3. You might have to adjust batch sizes according to the memory capacity of your machine.