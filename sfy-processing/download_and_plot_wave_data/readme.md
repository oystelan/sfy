# Installation and simulation instruction.

1. Clone this repository https://github.com/oystelan/sfy
2. go to directory sfy/sfy-processing/download_and_plot_wave_data/
3. follow the sfy python installation instructions in sfy_module.pdf. Make sure to use THIS forked version of SFY and not the original from Gaute, as it contains several needed bugfixes.
4. When installation is complete, test that you are able to download (hint: 
sfydata axl ts UIO_MEK_<your_buoy_name> --file <outfilename>.nc ). Note that if on windows, you need to add SFY_SERVER, SFY_READ_TOKEN and SFY_DATA_CACHE to system environmental variables are restart before it will work
5. Install UTC time app on your iphone. This is useful to keep track of time.
6. i have prepared a special script for the competition named dnv_competion_script.py It is fairly self explaining and should be what you need for the competition. enjoy!

br Oystein