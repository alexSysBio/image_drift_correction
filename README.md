# image_drift_correction 🔬🪛

This reository includes a collection of functions written in Python to correct the drift in time-lapse microscopy images.
The drift between consecutive frames is calculated by applying the 'phase_cross_correlation' function from the Scikit-image library (https://scikit-image.org/docs/0.25.x/api/skimage.registration.html#skimage.registration.phase_cross_correlation).
Then the cumulative drift is calculated. The use can choose to smnooth the cumulative drift by applying:
1. A moving average
2. A polynomial function of nth degree
3. A univariate smoothing spline

The smoothed or raw cumulative drifts are used to adjust the maximum possible croppig frame around the microscopy images.
An example of the drift correction is shown below. The uncorrected movie is on the left, and the drif-corrected time-lapse movie is shown on the right:
![movie_example.gif](https://github.com/alexSysBio/image_drift_correction/blob/main/correction_example.gif)

An example of using the functions provided in the repository is included in the 'drift_correction_user_guide.py' script.
