#!/bin/sh

## Convert images sequence to MP4 video
ffmpeg -framerate 12 -start_number 1 -i /data/render/drone_building_background/%04d.png -c:v libx264 -r 24 -pix_fmt yuv420p out.mp4
