#!/bin/bash

# Initializes the project's structure with train, test, and valid dirs.
# Each directory contains a file listing the video and its labels, as well as some dummy images.
# Run this script to call `python main.py' locally (not from the VM).

# NOTE: Requires imagemagick. Try:
# brew install imagemagick
#   OR
# sudo apt-get install imagemagick

read -p "Would you like to set up dummy training data? [Do NOT run this on the VM]:" -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

# Can use obama-360x240.png
echo -n "Enter the path to an image to be used for all frames [leave empty to use a random image]: "
read frame

echo "Setting up training data..."
parentdir="$(dirname "$PWD")"

modes=( train test valid )
COUNTER=0
for mode in "${modes[@]}"
do
	# Create the top-level directory.
	DIR=${parentdir}/$mode
	mkdir -p $DIR
	SUBDIR=$DIR/$DIR

	# Create the labels list file for each video example.
	LIST_FILE=${DIR}/${mode}_list.txt

	NUM_SIGNERS=5
	for ((i=1;i<=NUM_SIGNERS;i++)); do
		SIGNER_ID=$(printf "%03d\n" $i)

		# Sample a random number of videos (1 to 5) for each signer.
		NUM_VIDEOS=$(( ( RANDOM % 5 )  + 1 ))

		for ((j=1;j<=NUM_VIDEOS;j++)); do
			VIDEO_ID=$(printf "%05d\n" $j)

			M_VIDEO_DIR="${DIR}/${mode}/${SIGNER_ID}/M_${VIDEO_ID}"
			K_VIDEO_DIR="${DIR}/${mode}/${SIGNER_ID}/K_${VIDEO_ID}"
			M_VIDEO_PATH="${M_VIDEO_DIR}.avi"
			K_VIDEO_PATH="${K_VIDEO_DIR}.avi"

			mkdir -p $M_VIDEO_DIR
			mkdir -p $K_VIDEO_DIR

			# Store a random label for the video frames (1 to 10).
			LABEL=$(( ( RANDOM % 10 )  + 1 ))
			echo $M_VIDEO_PATH $K_VIDEO_PATH $LABEL >> $LIST_FILE

			# Save a random number of fake frames (1 to 30) for each video.
			NUM_FRAMES=$(( ( RANDOM % 30 )  + 1 ))
			for ((k=1;k<=NUM_FRAMES;k++)); do
				M_VIDEO_FRAME_PNG="${M_VIDEO_DIR}/M_${VIDEO_ID}_${k}.png"
				K_VIDEO_FRAME_PNG="${K_VIDEO_DIR}/K_${VIDEO_ID}_${k}.png"

				if [[ $frame = *[!\ ]* ]]; then
					# Copy the frame image over.
					cp $frame $M_VIDEO_FRAME_PNG
					cp $frame $K_VIDEO_FRAME_PNG
				else
					# Save randomly generated PNGs.
				 	mx=320;my=240;head -c "$((3*mx*my))" /dev/urandom | convert -depth 8 -size "${mx}x${my}" RGB:- $M_VIDEO_FRAME_PNG
					mx=320;my=240;head -c "$((3*mx*my))" /dev/urandom | convert -depth 8 -size "${mx}x${my}" RGB:- $K_VIDEO_FRAME_PNG
				fi
			done
			let COUNTER=COUNTER+2
		done
	done
done

echo "Created ${COUNTER} total videos."
