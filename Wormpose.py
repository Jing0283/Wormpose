import os
from wormpose.config import default_paths
from wormpose.dataset.loader import load_dataset

from ipython_utils import ImagesViewer
import matplotlib.pyplot as plt
import tempfile
from wormpose.demo.synthetic_simple_visualizer import SyntheticSimpleVisualizer
from ipython_utils import ImagesViewer, display_as_slider

from wormpose.demo.real_simple_visualizer import RealSimpleVisualizer
from ipython_utils import ImagesViewer, display_as_slider

import numpy as np
import h5py
import matplotlib.pyplot as plt
from wormpose.commands import calibrate
from ipython_utils import ImagesViewer

from wormpose.commands import generate
from ipywidgets import FloatProgress
from IPython.display import display

from glob import glob
import os
from wormpose.config import default_paths
from wormpose.machine_learning import tfrecord_file
from ipython_utils import ImagesViewer
import tensorflow as tf
from wormpose.commands import train

from wormpose.commands import predict


def main():
    # We have different loaders for different datasets, we use "sample_data" for the tutorial data,
    # replace with "tierpsy" for Tierpsy tracker data, or with your custom dataset loader name
    dataset_loader = "sample_data"

    # Set the path to the dataset,
    # for Tierpsy tracker data this will be the root path of a folder containing subfolders for each videos
    dataset_path = "your dataset root path"
    print(dataset_path)
    dataset_root_name = os.path.basename(os.path.normpath(dataset_path))
    project_dir = os.path.join(default_paths.WORK_DIR, dataset_root_name)

    # Set if the worm is lighter than the background in the image
    # in the sample data, the worm is darker so we set this variable to False
    worm_is_lighter = False

    # This function loads the dataset
    # optional fields: there is an optional resize parameter to resize the images
    # also you can select specific videos from the dataset instead of loading them all
    dataset = load_dataset(dataset_loader, dataset_path, worm_is_lighter=worm_is_lighter)
    print(dataset)
    # visualize the raw dataset images
    MAX_FRAMES = 100
    img_viewer = ImagesViewer()

    video_name = dataset.video_names[0]
    with dataset.frames_dataset.open(video_name) as frames:
        for frame in frames[:MAX_FRAMES]:
            img_viewer.add_image(frame)

    img_viewer.view_as_slider()

    # visualise synthetic images
    synth_viz = SyntheticSimpleVisualizer(dataset_loader,
                                          dataset_path,
                                          worm_is_lighter=worm_is_lighter).generate()
    img_viewer, img_viewer_plot = ImagesViewer(), ImagesViewer()
    num_images = 50

    print(f"Viewing {num_images} synthetic images.")
    tempdir = tempfile.gettempdir()
    for i in range(num_images):
        synth_image, theta = next(synth_viz)

        plt.plot(theta)
        plt.ylabel("theta (rad)")
        plt.xlabel("body segment")
        plot_path = os.path.join(tempdir, f"theta_{i}.png")
        plt.savefig(plot_path)
        plt.clf()
        img_viewer_plot.add_image_filename(plot_path)

        img_viewer.add_image(synth_image)

    display_as_slider(img_viewer, img_viewer_plot)

    # visualise frame preprocessing
    viz = RealSimpleVisualizer(dataset_loader,
                               dataset_path,
                               worm_is_lighter=worm_is_lighter).generate()
    orig_img_viewer, processed_img_viewer = ImagesViewer(), ImagesViewer()

    max_viz = 100
    print(f"Displaying the first {max_viz} frames : original and processed.")

    for _ in range(max_viz):
        orig_image, processed_image = next(viz)
        orig_img_viewer.add_image(orig_image)
        processed_img_viewer.add_image(processed_image)

    display_as_slider(orig_img_viewer, processed_img_viewer)

    print(
        f"The processed images are all set to the size: ({processed_image.shape[0]}px, {processed_image.shape[1]}px).")

    # calibration
    video_name, result_file = next(calibrate(dataset_loader,
                                             dataset_path,
                                             worm_is_lighter=worm_is_lighter,
                                             save_images=True))

    VIEW_SCORES = 5

    img_viewer = ImagesViewer()
    with h5py.File(result_file, "r") as f:
        scores = f['scores'][()]
        real_images = f['real_images']
        synthetic_images = f['synth_images']

        plt.hist(scores, bins=np.arange(0.5, 1, 0.01),
                 weights=np.ones_like(scores) / len(scores))
        plt.xlabel("image similarity")
        plt.title(f"Distribution of image scores for known frames\n (video: {video_name})")
        plt.show()

        sorted_scores = np.argsort(scores)
        step = int(len(sorted_scores) / VIEW_SCORES)
        sorted_selection_index = [sorted_scores[0]] + sorted_scores[step:-step:step].tolist() + [sorted_scores[-1]]

        for index in sorted_selection_index:
            im = np.hstack([real_images[index], synthetic_images[index]])
            img_viewer.add_image(im)

    img_viewer.view_as_list(legends=scores[sorted_selection_index])

    # build training and evaluation dataset

    fp = FloatProgress(min=0., max=1.)
    display(fp)

    gen_progress = generate(dataset_loader,
                            dataset_path,
                            worm_is_lighter=worm_is_lighter,
                            num_train_samples=1000)

    for progress_value in gen_progress:
        fp.value = progress_value

    # Check generated tf-record files
    def view_tfrecord(filename, theta_dims=100, max_viz=100):
        img_viewer = ImagesViewer()
        for index, record in enumerate(tfrecord_file.read(filename, theta_dims)):
            if index >= max_viz:
                break
            image_data = record[0].numpy()
            img_viewer.add_image(image_data)
        print(f"Reading: \'{filename}\' ({index} first frames)")

        img_viewer.view_as_slider()

    train_records = list(sorted(glob(os.path.join(project_dir,
                                                  default_paths.TRAINING_DATA_DIR,
                                                  default_paths.SYNTH_TRAIN_DATASET_NAMES.format(index='*')))))
    print(f"Training tfrecord files: {len(train_records)} files.")
    if len(train_records) > 0:
        view_tfrecord(train_records[0])
    eval_record = list(glob(os.path.join(project_dir,
                                         default_paths.TRAINING_DATA_DIR,
                                         default_paths.REAL_EVAL_DATASET_NAMES.format(index='*'))))
    if len(eval_record) > 0:
        view_tfrecord(eval_record[0])

    # train
    if tf.test.gpu_device_name() == '':
        print("Warning, no GPU available for training, this will be very slow.")

    train(dataset_path, epochs=10)

    # predict
    use_pretrained_network = True
    if use_pretrained_network:
        model_path = "D:\\JingALing\\Wormpose\\models\\sample_data\\trained_model.hdf5"
    else:
        model_path = None

    predict(dataset_path=dataset_path,
            score_threshold=0.7,
            model_path=model_path)


if __name__ == '__main__':
    main()
