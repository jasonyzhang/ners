ffmpeg -i 40_Church\ of\ St.\ Paraskevi.gif -c:v libx264 -preset slow -crf 20 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p,reverse" 40_Church\ of\ St.\ Paraskevi.mp4
