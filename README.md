[Link to overview (2 pages)](SummaryReport.pdf)

I also wrote a pretty in depth (but not scientifically formal) report that details everything I tried, mathematical formulation, and have some images and results. This is not formal academic paper writing, but like a description of what is going on.

[Link to in depth report](InDepthReport.pdf)


[Link to video](https://youtu.be/oWuA4C3rU-4)



To run this code, first download [k-Wave](http://www.k-wave.org/), MATLAB R2025b, Parallel Computing Toolbox, Signal Processing Toolbox

To generate the data, first run `matlab -batch utils/run.m`
You can modify the contents of run.m to specific the size of your dataset, what RNG seed you use and the output path.

Next, create a python virtual environment and install all packages specified in `requirements.txt`.

From there, you can modify `train.py` for either the FNO or the WNO to train the model. For evaluation, save the image and run `utils/view_image.m` in MATLAB to evaluate how well the model creates a beamformed image.


